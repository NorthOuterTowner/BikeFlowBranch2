import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import json
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLAlchemy 连接
try:
    engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')
except Exception as e:
    logger.error(f"数据库连接失败: {e}")
    exit(1)

# 加载预测数据和容量
try:
    status_df = pd.read_sql("SELECT * FROM station_hourly_status", engine)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT station_id, capacity FROM station_info"))
        station_capacities = {row[0]: row[1] if row[1] is not None else 20 for row in result.fetchall()}
    stations = status_df['station_id'].unique()
except Exception as e:
    logger.error(f"加载数据失败: {e}")
    exit(1)

# 检查 status_df 是否为空
if len(status_df) == 0:
    logger.error("station_hourly_status 表为空，请先运行 writetosql_updated.py")
    exit(1)

# 模拟离线数据集
def generate_offline_data(status_df, station_capacities, num_samples=10000):
    offline_data = []
    for i in range(num_samples):
        sample = status_df.sample(1).iloc[0]
        station_id = sample['station_id']
        capacity = station_capacities.get(station_id, 20)
        # 转换为 Python 原生类型
        state = [
            int(sample['stock']),  # numpy.int64 -> int
            int(sample['inflow']),
            int(sample['outflow']),
            int(sample['hour'])
        ]
        action = {"from_station": None, "to_station": None, "bikes": 0}
        if sample['stock'] > 10:
            target_station = np.random.choice(stations)
            action = {"from_station": station_id, "to_station": target_station, "bikes": 5}
        demand_satisfied = 1.0 if 5 <= sample['stock'] <= capacity else 0.0
        cost = 0.1 * action['bikes']
        reward = 0.5 * demand_satisfied - cost - (0.2 if sample['stock'] < 5 else 0.0)
        next_stock = sample['stock'] + sample['inflow'] - sample['outflow'] - action['bikes']
        next_stock = max(0, min(next_stock, capacity))
        next_state = [
            int(next_stock),  # numpy.int64 -> int
            int(sample['inflow']),
            int(sample['outflow']),
            int((sample['hour'] + 1) % 24)
        ]
        offline_data.append({
            "station_id": str(station_id),
            "date": str(sample['date']),  # 确保 date 是字符串
            "hour": int(sample['hour']),  # numpy.int64 -> int
            "state": json.dumps(state),
            "action": json.dumps(action),
            "reward": float(reward),  # 确保为 float
            "next_state": json.dumps(next_state),
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        if (i + 1) % 1000 == 0:
            logger.info(f"已生成 {i + 1} 条离线数据")
    return offline_data

# 生成并保存数据
try:
    offline_data = generate_offline_data(status_df, station_capacities)
    with engine.connect() as conn:
        # 清空 offline_rl_data 表
        conn.execute(text("TRUNCATE TABLE offline_rl_data"))
        logger.info("已清空 offline_rl_data 表")
        
        # 批量插入（分批提交，每 1000 条）
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
            logger.info(f"已插入 {min(i + batch_size, len(offline_data))} 条记录")
except Exception as e:
    logger.error(f"数据插入失败: {e}")
    exit(1)

# 验证插入
try:
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM offline_rl_data")).scalar()
    logger.info(f"生成离线数据集: {count} 条记录")
except Exception as e:
    logger.error(f"验证插入失败: {e}")
    exit(1)