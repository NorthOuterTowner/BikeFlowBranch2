# -*- coding: utf-8 -*-
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.system('chcp 65001 > nul')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import json
import logging
from datetime import datetime

# 自定义 JSON 编码器
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')
except Exception as e:
    logger.error(f"数据库连接失败: {e}")
    exit(1)

try:
    status_df = pd.read_sql("SELECT * FROM station_hourly_status", engine)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT station_id, capacity FROM station_info"))
        rows = result.mappings().fetchall()
        if not rows:
            logger.error("station_info 表为空，请检查数据")
            exit(1)
        station_capacities = {row['station_id']: row['capacity'] if row['capacity'] is not None else 20 for row in rows}
    stations = status_df['station_id'].unique()
except Exception as e:
    logger.error(f"加载数据失败: {e}")
    exit(1)

if len(status_df) == 0:
    logger.error("station_hourly_status 表为空，请先运行 writetosql_updated.py")
    exit(1)

def generate_offline_rl_data(status_df, station_capacities, num_samples=10000):
    status_df['datetime'] = pd.to_datetime(status_df['date']) + pd.to_timedelta(status_df['hour'], unit='h')
    agg_status_df = status_df.groupby(['station_id', pd.Grouper(key='datetime', freq='3h')]).agg({
        'stock': 'mean',
        'inflow': 'sum',
        'outflow': 'sum',
        'hour': 'first'
    }).reset_index()
    agg_status_df['date'] = agg_status_df['datetime'].dt.date.astype(str)
    agg_status_df['time_slot'] = agg_status_df['datetime'].dt.hour
    agg_status_df['hour'] = agg_status_df['hour'].astype(int)

    offline_data = []
    bikes_options = [0, 5, 10]
    for i in range(num_samples):
        sample = agg_status_df.sample(1).iloc[0]
        noise_scale = 0.1  # 表示最大10%的扰动
        sample['inflow'] *= np.random.uniform(1 - noise_scale, 1 + noise_scale)
        sample['outflow'] *= np.random.uniform(1 - noise_scale, 1 + noise_scale)
        station_id = sample['station_id']
        capacity = station_capacities.get(station_id, 20)
        logger.debug(f"Sample types: stock={type(sample['stock'])}, inflow={type(sample['inflow'])}, hour={type(sample['hour'])}")
        state = [
            float(sample['stock']),
            float(sample['inflow']),
            float(sample['outflow']),
            int(sample['hour'])
        ]
        action = {"from_station": None, "to_station": None, "bikes": 0}
        if sample['stock'] > 5:
            target_station = np.random.choice(stations)
            if target_station not in stations:
                logger.warning(f"Invalid to_station {target_station}")
                target_station = np.random.choice(stations)
            bikes = np.random.choice(bikes_options[1:])
            action = {"from_station": station_id, "to_station": target_station, "bikes": bikes}
        elif sample['stock'] < 8:
            from_station = np.random.choice(stations)
            if from_station != station_id:
                bikes = np.random.choice([5, 10])
                action = {"from_station": from_station, "to_station": station_id, "bikes": bikes}
        demand_satisfied = 1.0 if 10 <= sample['stock'] <= capacity else 0.0
        cost = 0.02 * action['bikes']
        balance_reward = 0.3 * (1 - abs(sample['stock'] - capacity / 2) / (capacity / 2))
        penalty = 0.5 if sample['stock'] < 5 or sample['stock'] > capacity - 5 else 0.0

        # 新增：库存低（stock < 5）时如果没有调度（bikes == 0），给予额外惩罚
        extra_penalty = 0.0
        if sample['stock'] < 5 and action['bikes'] == 0:
            extra_penalty = 1.0  # 你可以根据实验调为 0.8 或 1.0

        reward = 0.5 * demand_satisfied - cost + balance_reward - penalty - extra_penalty

        reward = min(max(reward, -1.0), 1.0)
        next_stock = sample['stock'] + sample['inflow'] - sample['outflow'] - action['bikes']
        next_stock = max(0, min(next_stock, capacity))
        next_state = [
            float(next_stock),
            float(sample['inflow']),
            float(sample['outflow']),
            int((sample['hour'] + 3) % 24)
        ]
        offline_data.append({
            "station_id": str(station_id),
            "date": str(sample['date']),
            "hour": int(sample['hour']),
            "state": json.dumps(state, cls=NpEncoder),
            "action": json.dumps(action, cls=NpEncoder),
            "reward": float(reward),
            "next_state": json.dumps(next_state, cls=NpEncoder),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        if (i + 1) % 1000 == 0:
            logger.info(f"已生成 {i + 1} 条离线数据")
    return offline_data

try:
    offline_data = generate_offline_rl_data(status_df, station_capacities)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE offline_rl_data"))
        logger.info("已清空 offline_rl_data 表")
        
        insert_sql = """
        INSERT INTO offline_rl_data
        (station_id, date, hour, state, action, reward, next_state, created_at)
        VALUES (:station_id, :date, :hour, :state, :action, :reward, :next_state, :created_at)
        """
        batch_size = 1000
        for i in range(0, len(offline_data), batch_size):
            batch = offline_data[i:i + batch_size]
            conn.execute(text(insert_sql), parameters=[{
                "station_id": d['station_id'],
                "date": d['date'],
                "hour": d['hour'],
                "state": d['state'],
                "action": d['action'],
                "reward": d['reward'],
                "next_state": d['next_state'],
                "created_at": d['created_at']
            } for d in batch])
            conn.commit()
            if (i + batch_size) % 5000 == 0:
                logger.info(f"已插入 {min(i + batch_size, len(offline_data))} 条记录")
except Exception as e:
    logger.error(f"插入数据失败: {e}")
    exit(1)

try:
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM offline_rl_data")).scalar()
    logger.info(f"生成离线数据集: {count} 条记录")
except Exception as e:
    logger.error(f"验证插入失败: {e}")
    exit(1)