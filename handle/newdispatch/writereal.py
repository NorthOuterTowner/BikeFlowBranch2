'''
6.12-6.14的真实数据初始化
'''
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# ---------- 配置数据库 ----------
user = 'zq'
password = '123456'
host = 'localhost'
database = 'traffic'
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# ---------- 获取站点容量 ----------
def load_station_capacities():
    df = pd.read_sql("SELECT station_id, capacity FROM station_info", engine)
    return dict(zip(df['station_id'], df['capacity']))

# ---------- 初始化真实库存 ----------
def init_real_stock_from_flow():
    print("[初始化] 开始生成 6.12 ~ 6.14 的真实库存数据...")

    capacities = load_station_capacities()
    default_capacity = 20

    start_date = datetime(2025, 6, 12)
    end_date = datetime(2025, 6, 15)  # 注意是 < 6.15，含6.14

    # 获取所有站点
    all_stations = list(capacities.keys())

    # 生成完整时间戳列表
    full_timestamps = [start_date + timedelta(hours=h) for h in range(72)]  # 3天 * 24小时

    # 原始流量数据
    df_raw = pd.read_sql(f"""
        SELECT station_id, timestamp, inflow, outflow 
        FROM station_hourly_flow
        WHERE timestamp >= '{start_date.strftime("%Y-%m-%d")}'
          AND timestamp < '{end_date.strftime("%Y-%m-%d")}'
    """, engine)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

    # 删除原有记录
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM station_real_data
            WHERE date >= '2025-06-12' AND date <= '2025-06-14'
        """))

    result_rows = []
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 输出统计信息
    print(f"\n[统计] 总站点数（station_info）：{len(all_stations)}")
    print(f"[统计] 实际出现的站点数（station_hourly_flow）：{df_raw['station_id'].nunique()}")
    print(f"[统计] 时间点数量（全局唯一时间戳数量）：{df_raw['timestamp'].nunique()}")
    print(f"[统计] 预期记录数（站点数 × 时间点数）：{len(all_stations) * len(full_timestamps)}")

    # 每个站点处理
    for sid in all_stations:
        cap = capacities.get(sid, default_capacity)
        stock = cap // 2
        stock_by_time = {}

        # 筛选该站点的原始数据
        df_station = df_raw[df_raw['station_id'] == sid].set_index('timestamp')

        for ts in full_timestamps:
            date_str = ts.date()
            hour = ts.hour

            if hour == 0:
                stock = cap // 2  # 每天0点重置

            inflow = 0
            outflow = 0

            if ts in df_station.index:
                inflow = df_station.loc[ts]['inflow']
                outflow = df_station.loc[ts]['outflow']

            stock += inflow - outflow
            stock = max(0, min(stock, cap))

            result_rows.append({
                "station_id": sid,
                "date": date_str,
                "hour": hour,
                "stock": int(stock),
                "update_time": update_time
            })

    df_result = pd.DataFrame(result_rows)
    df_result.to_sql("station_real_data", engine, if_exists='append', index=False)

    print(f"[完成] 写入 {len(df_result)} 条 6.12 ~ 6.14 真实库存记录")

# ---------- 执行 ----------
if __name__ == '__main__':
    init_real_stock_from_flow()
