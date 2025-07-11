import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import numpy as np

# ---------- 配置数据库 ----------
user = 'zq'
password = '123456'
host = 'localhost'
database = 'traffic'

try:
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
except Exception as e:
    print(f"[错误] 数据库连接失败：{e}")
    sys.exit(1)

# ---------- 读取站点容量 ----------
def load_station_capacities():
    df = pd.read_sql("SELECT station_id, capacity FROM station_info", engine)
    return dict(zip(df['station_id'], df['capacity']))

# ---------- 获取当前库存 ----------
def get_current_stock(date_str, hour):
    sql = """
    SELECT station_id, stock FROM station_real_data
    WHERE date = :date AND hour = :hour
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"date": date_str, "hour": hour})
        return {row['station_id']: row['stock'] for row in result.mappings()}

# ---------- 获取未来 inflow/outflow ----------
def get_future_flow(start_time, end_time):
    sql = f"""
    SELECT station_id, SUM(inflow) AS inflow, SUM(outflow) AS outflow
    FROM station_hourly_status
    WHERE CONCAT(date, ' ', LPAD(hour, 2, '0'), ':00:00') >= '{start_time}'
      AND CONCAT(date, ' ', LPAD(hour, 2, '0'), ':00:00') < '{end_time}'
    GROUP BY station_id
    """
    df = pd.read_sql(sql, engine)
    return df

# ---------- 计算距离（可扩展） ----------
def compute_distance(sid1, sid2):
    return 1  # 简化为常数，可拓展为地理计算或图计算

# ---------- 生成调度图并求解 ----------
def plan_dispatch(df, capacities):
    G = nx.DiGraph()
    surplus_nodes, shortage_nodes = [], []

    for _, row in df.iterrows():
        sid = row['station_id']
        cap = capacities.get(sid, 50)
        min_thres, max_thres = 0.2 * cap, 0.8 * cap
        expected = row['stock'] + row['inflow'] - row['outflow']

        print(f"[检查] 站点 {sid} | cap={cap} | inflow={row['inflow']} outflow={row['outflow']} stock={row['stock']} ➔ 预计库存={expected:.1f}")

        if expected < min_thres:
            shortage = max(0, int(min_thres - expected))
            if shortage > 0:
                shortage_nodes.append((sid, shortage))
                print(f" → 缺车站点: {sid} 需要 {shortage} 辆")
        elif expected > max_thres:
            surplus = max(0, int(expected - max_thres))
            if surplus > 0:
                surplus_nodes.append((sid, surplus))
                print(f" → 多车站点: {sid} 可调出 {surplus} 辆")

    print(f"\n[构建图] 调出站点：{len(surplus_nodes)} 个，调入站点：{len(shortage_nodes)} 个")

    total_supply = sum(q for _, q in surplus_nodes)
    total_demand = sum(q for _, q in shortage_nodes)
    balanced_qty = min(total_supply, total_demand)

    new_surplus, remain = [], balanced_qty
    for sid, qty in surplus_nodes:
        q = min(qty, remain)
        if q > 0:
            new_surplus.append((sid, q))
            remain -= q
        if remain <= 0:
            break

    new_shortage, remain = [], balanced_qty
    for sid, qty in shortage_nodes:
        q = min(qty, remain)
        if q > 0:
            new_shortage.append((sid, q))
            remain -= q
        if remain <= 0:
            break

    for sid, qty in new_surplus:
        G.add_node(sid, demand=-qty)
    for sid, qty in new_shortage:
        G.add_node(sid, demand=qty)

    for from_id, from_qty in new_surplus:
        for to_id, to_qty in new_shortage:
            qty = min(from_qty, to_qty)
            if qty > 0:
                cost = compute_distance(from_id, to_id)
                G.add_edge(from_id, to_id, capacity=qty, weight=cost)
                print(f" → 边：{from_id} ➔ {to_id} 容量={qty} 成本={cost}")

    if len(G.edges) == 0:
        print("没有生成任何搬运边，图为空，返回空结果")
        return []

    try:
        flow_dict = nx.network_simplex(G)[1]
    except Exception as e:
        print(f"[错误] 最小费用流求解失败：{e}")
        return []

    actions = []
    for u, v_dict in flow_dict.items():
        for v, bikes in v_dict.items():
            if bikes > 0:
                actions.append({'from': u, 'to': v, 'bikes': bikes})
                print(f"搬运计划：{u} ➔ {v} 搬 {int(bikes)} 辆")

    return actions

# ---------- 写入数据库 ----------
def save_schedule_to_db(date_str, hour, actions):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM station_schedule 
                WHERE date = :date AND hour = :hour
            """), {"date": date_str, "hour": hour})

            for action in actions:
                conn.execute(text("""
                    INSERT INTO station_schedule 
                        (date, hour, start_id, end_id, bikes, updated_at)
                    VALUES 
                        (:date, :hour, :start_id, :end_id, :bikes, NOW())
                """), {
                    "date": date_str,
                    "hour": hour,
                    "start_id": action["from"],
                    "end_id": action["to"],
                    "bikes": action["bikes"]
                })
    except Exception as e:
        print(f"[错误] 写入调度结果失败：{e}")

# ---------- 主调度函数 ----------
def run_scheduler_for_timepoint(date_str, hour):
    print(f"\n[开始调度] 时间点：{date_str} {hour:02d} 点")
    capacities = load_station_capacities()
    current_stock = get_current_stock(date_str, hour)

    start_time = f"{date_str} {str(hour).zfill(2)}:00:00"
    end_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S") + timedelta(hours=3)
    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    df = get_future_flow(start_time, end_time)
    df['stock'] = df['station_id'].map(current_stock).fillna(0)

    if df.empty:
        print(f"[警告] 时间段内无数据：{start_time} ~ {end_time}")
        return

    actions = plan_dispatch(df, capacities)

    print(f"[调度完成] 调度动作数：{len(actions)}")
    save_schedule_to_db(date_str, hour, actions)

# ---------- 示例入口 ----------
# patch.py 最后加上这段
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='调度日期，例如 2025-06-13')
    parser.add_argument('--hour', type=int, required=True, help='小时，例如 9 表示 09:00')
    args = parser.parse_args()

    run_scheduler_for_timepoint(args.date, args.hour)
