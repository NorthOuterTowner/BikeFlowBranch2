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
    df = pd.read_sql("SELECT * FROM station_hourly_flow ORDER BY timestamp, station_id", conn)
    conn.close()
    return df

def build_samples(df):
    # 从 timestamp 中提取时间特征
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['date'] = df['timestamp'].dt.date.astype(str)

    # 特征构造：将每个站点、每个时间戳的数据作为一个样本
    feature_cols = ['station_id', 'hour', 'weekday']
    X = pd.get_dummies(df[feature_cols], columns=['station_id'], drop_first=False)
    y_inflow = df['inflow']
    y_outflow = df['outflow']

    # 保存训练集
    os.makedirs('./handle/lgbm/data', exist_ok=True)
    X.to_csv('./handle/lgbm/data/X.csv', index=False)
    y_inflow.to_csv('./handle/lgbm/data/y_inflow.csv', index=False)
    y_outflow.to_csv('./handle/lgbm/data/y_outflow.csv', index=False)
    df.to_csv('./handle/lgbm/lgbm_raw_samples.csv', index=False)
    print("样本构建完成，已保存至 ./handle/lgbm/data/")

if __name__ == "__main__":
    df = load_data()
    build_samples(df)
