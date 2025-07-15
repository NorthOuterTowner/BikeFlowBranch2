import pandas as pd
import pymysql
import joblib
from datetime import datetime

def get_connection():
    return pymysql.connect(host='localhost', user='root', password='123456', 
                         database='traffic', charset='utf8')

def add_features(df):
    """特征工程函数"""
    # 排序
    df = df.sort_values(['station_id', 'timestamp'])
    
    # 构造时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 构造滞后特征
    for lag in [1, 2, 3]:
        df[f'inflow_lag_{lag}'] = df.groupby('station_id')['inflow'].shift(lag)
        df[f'outflow_lag_{lag}'] = df.groupby('station_id')['outflow'].shift(lag)
    
    return df

def predict_and_write():
    try:
        # 1. 加载数据和编码器
        df = pd.read_csv('./handle/lgbm/lgbm_raw_samples.csv', parse_dates=['timestamp'])
        encoder = joblib.load('./handle/lgbm/station_id_encoder.pkl')
        
        # 2. 应用特征工程
        df = add_features(df)
        
        # 3. 确保与训练时相同的预处理
        df['station_id_encoded'] = encoder.transform(df['station_id'])
        df['hour'] = df['timestamp'].dt.hour
        
        # 4. 准备特征
        features = ['station_id_encoded', 'hour', 'dayofweek', 'is_weekend',
                   'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
                   'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3']
        X = df[features]
        
        # 5. 加载模型并预测
        model_in = joblib.load('./handle/lgbm/inflow_model.pkl')
        model_out = joblib.load('./handle/lgbm/outflow_model.pkl')
        
        df['pred_inflow'] = model_in.predict(X)
        df['pred_outflow'] = model_out.predict(X)
        
        # 6. 写入数据库
        conn = get_connection()
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            sql = """
            REPLACE INTO station_hourly_status 
            (station_id, date, hour, inflow, outflow, stock, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                row['station_id'],  # 原始station_id
                row['timestamp'].date(), 
                row['hour'],
                row['pred_inflow'], 
                row['pred_outflow'],
                0,  # stock设为0
                datetime.now()  # 当前时间作为updated_at
            ))
        
        conn.commit()
        print(f"成功写入 {len(df)} 条预测数据")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    predict_and_write()