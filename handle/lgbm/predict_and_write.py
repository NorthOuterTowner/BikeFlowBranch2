import pandas as pd
import pymysql
import joblib
from datetime import datetime
import numpy as np
import warnings

# 忽略DtypeWarning警告（可选）
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

def get_connection():
    return pymysql.connect(host='localhost', user='root', password='123456', 
                         database='traffic', charset='utf8')
def get_station_capacities(conn):
    """获取站点容量信息"""
    query = "SELECT station_id, capacity FROM station_info"
    capacities = pd.read_sql(query, conn)
    return capacities.set_index('station_id')['capacity'].to_dict()
def calculate_stock(df, capacities):
    """
    计算库存预测
    基于流入流出预测，确保库存不为0或负数
    """
    # 按站点分组计算
    grouped = df.groupby('station_id')
    
    stock_values = []
    for station_id, group in grouped:
        capacity = capacities.get(station_id, 20)  # 默认容量20
        
        # 按时间排序
        group = group.sort_values('timestamp')
        
        # 初始化库存(使用容量的一半或最后已知库存)
        initial_stock = max(1, capacity // 2)
        
        # 计算库存变化
        stock = [initial_stock]
        for i in range(1, len(group)):
            new_stock = stock[i-1] + group.iloc[i]['pred_inflow'] - group.iloc[i]['pred_outflow']
            stock.append(max(1, min(new_stock, capacity)))  # 确保在[1, 容量]之间
        
        stock_values.extend(stock)
    
    return stock_values
def add_features(df):
    """添加时间、滞后、节假日、天气特征，适用于 LightGBM"""

    # 确保 timestamp 是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 排序
    df = df.sort_values(['station_id', 'timestamp'])

    # 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype('int8')

    # 滞后特征
    for lag in [1, 2, 3]:
        df[f'inflow_lag_{lag}'] = df.groupby('station_id')['inflow'].shift(lag)
        df[f'outflow_lag_{lag}'] = df.groupby('station_id')['outflow'].shift(lag)

    # ---------- 合并节假日 ----------
    try:
        holidays_df = pd.read_csv(
            './handle/lgbm/us_holidays_2025.csv',
            parse_dates=['date'],
            dtype={'is_holiday': 'int8'}
        )
        # 统一日期格式
        df['merge_date'] = df['timestamp'].dt.normalize()
        holidays_df['date'] = holidays_df['date'].dt.normalize()

        df = pd.merge(
            df,
            holidays_df[['date', 'is_holiday']].rename(columns={'date': 'merge_date'}),
            on='merge_date',
            how='left'
        )

        df['is_holiday'] = df['is_holiday'].fillna(0).astype('int8')
        df.drop(columns='merge_date', inplace=True)

    except Exception as e:
        print(f"[警告] 节假日合并失败：{e}")
        df['is_holiday'] = 0

    # ---------- 合并天气 ----------
    try:
        weather_df = pd.read_csv(
            './handle/lgbm/nyc_weather_hourly.csv',
            parse_dates=['timestamp'],
            dtype={'temp': 'float32', 'prcp': 'float32', 'wspd': 'float32'}
        )

        df = pd.merge(df, weather_df, on='timestamp', how='left')

        # 确保三个字段都存在，即使未成功合并也补上
        for col in ['temp', 'prcp', 'wspd']:
            if col not in df.columns:
                print(f"[警告] 缺失字段 {col}，填充默认值 0.0")
                df[col] = 0.0
            df[col] = df[col].fillna(df[col].median()).astype('float32')

    except Exception as e:
        print(f"[警告] 天气数据合并失败：{e}")
        df['temp'] = df['prcp'] = df['wspd'] = 0.0

    return df

def predict_and_write():
    conn = None
    try:
        conn = get_connection()
        
        # 1. 加载数据和编码器
        df = pd.read_csv(
            './handle/lgbm/lgbm_raw_samples.csv',
            parse_dates=['timestamp'],
            dtype={
                'station_id': 'str',
                'inflow': 'float32',
                'outflow': 'float32'
            }
        )
        
        # 检查必要列是否存在
        required_cols = ['station_id', 'timestamp', 'inflow', 'outflow']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
        
        encoder = joblib.load('./handle/lgbm/station_id_encoder.pkl')
        
        # 2. 应用特征工程
        df = add_features(df)
        
        # 只保留6月12日及以后的记录
        cutoff_date = pd.to_datetime('2025-06-12')
        df = df[df['timestamp'] >= cutoff_date]
        if df.empty:
            raise ValueError("没有符合时间条件的数据（需晚于2025-06-12）")
        
        # 3. 确保与训练时相同的预处理
        df['station_id_encoded'] = encoder.transform(df['station_id'].astype(str))
        df['hour'] = df['timestamp'].dt.hour
        
        # 4. 准备特征
        expected_features = [
            'station_id_encoded', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
            'temp', 'prcp', 'wspd',
            'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
            'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3'
        ]
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"缺少必要特征列: {missing_features}")
        
        X = df[expected_features]
        
        # 5. 加载模型并预测
        model_in = joblib.load('./handle/lgbm/inflow_model_tuned.pkl')
        model_out = joblib.load('./handle/lgbm/outflow_model_tuned.pkl')
        
        df['pred_inflow'] = np.maximum(0, model_in.predict(X))  # 确保非负
        df['pred_outflow'] = np.maximum(0, model_out.predict(X))  # 确保非负
        
        # 6. 计算库存预测
        capacities = get_station_capacities(conn)
        df['stock'] = calculate_stock(df, capacities)
        
        # 7. 写入数据库
        cursor = conn.cursor()
        
        # 准备批量插入数据
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append((
                str(row['station_id']),
                row['timestamp'].date(),
                int(row['hour']),
                float(row['pred_inflow']),
                float(row['pred_outflow']),
                int(row['stock']),  # 确保为整数
                datetime.now()
            ))
        
        # 批量执行（提高性能）
        sql = """
        REPLACE INTO station_hourly_status 
        (station_id, date, hour, inflow, outflow, stock, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(sql, data_to_insert)
        conn.commit()
        
        print(f"成功写入 {len(df)} 条预测数据（包含库存预测）")
        print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"库存范围: {df['stock'].min()}~{df['stock'].max()}")
        
    except Exception as e:
        print(f"[错误] 预测写入失败: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    predict_and_write()