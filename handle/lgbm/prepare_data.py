# handle/lgbm/prepare_data.py
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import pymysql
import os
from sklearn.model_selection import train_test_split

def load_data():
    conn = pymysql.connect(
        host="localhost", user="root", password="123456", database="traffic", charset="utf8mb4"
    )
    # 添加 coerce_float=True 避免类型问题
    df = pd.read_sql("SELECT * FROM station_hourly_flow ORDER BY timestamp, station_id", conn)
    conn.close()
    
    # 确保timestamp是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 加载天气数据（确保timestamp类型一致）
    weather_df = pd.read_csv('./handle/lgbm/nyc_weather_hourly.csv', parse_dates=['timestamp'])
    df = pd.merge(df, weather_df, on='timestamp', how='left')
    
    # 加载节假日数据（关键修复点）
    holidays_df = pd.read_csv('./handle/lgbm/us_holidays_2025.csv', parse_dates=['date'])  # 添加parse_dates
    holidays_df['is_holiday'] = 1
    
    # 统一date列为datetime64[ns]类型
    df['date'] = pd.to_datetime(df['timestamp'].dt.date)  # 显式转换为datetime64
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])  # 确保也是datetime64
    
    # 合并数据
    df = pd.merge(df, holidays_df[['date', 'is_holiday', 'holiday_name']], on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    
    return df

def build_samples(df):
    # 从 timestamp 中提取时间特征
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['date'] = df['timestamp'].dt.date.astype(str)

    # 保存原始数据（包含新特征）
    os.makedirs('./handle/lgbm/data', exist_ok=True)
    df.to_csv('./handle/lgbm/lgbm_raw_samples.csv', index=False)
    print("样本构建完成，已保存至 ./handle/lgbm/lgbm_raw_samples.csv")

if __name__ == "__main__":
    df = load_data()
    build_samples(df)