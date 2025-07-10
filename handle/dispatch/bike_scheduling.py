import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine, text
import json
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# 价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# 调度器
class OfflineRLScheduler:
    def __init__(self, policy_net, value_net, adj_matrix, station_capacities, stations):
        self.policy_net = policy_net
        self.value_net = value_net
        self.adj_matrix = adj_matrix
        self.station_capacities = station_capacities
        self.stations = list(stations)  # 转换为列表以支持 index 方法
    
    def train(self, offline_data, gamma=0.99, epochs=5):
        optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=0.001)
        optimizer_value = optim.Adam(self.value_net.parameters(), lr=0.001)
        for epoch in range(epochs):
            total_loss = 0
            for _, row in offline_data.iterrows():
                try:
                    state = torch.tensor(json.loads(row['state']), dtype=torch.float32)  # [4]
                    action_dict = json.loads(row['action'])
                    reward = torch.tensor(row['reward'], dtype=torch.float32)
                    next_state = torch.tensor(json.loads(row['next_state']), dtype=torch.float32)
                    
                    # 动作索引：0 表示不调度，1 到 len(stations) 表示目标站点
                    action_idx = 0 if action_dict['bikes'] == 0 else self.stations.index(action_dict['to_station']) + 1
                    action = torch.zeros(len(self.stations) + 1, dtype=torch.float32)
                    action[action_idx] = 1.0  # one-hot 编码
                    
                    # 计算价值损失
                    next_action_probs = self.policy_net(next_state)
                    next_action_idx = torch.argmax(next_action_probs).item()
                    next_action = torch.zeros(len(self.stations) + 1, dtype=torch.float32)
                    next_action[next_action_idx] = 1.0
                    target = reward + gamma * self.value_net(next_state, next_action).detach()
                    
                    predicted = self.value_net(state, action)
                    value_loss = (target - predicted) ** 2
                    
                    optimizer_value.zero_grad()
                    value_loss.backward()
                    optimizer_value.step()
                    
                    # 重新计算 policy_net 输出
                    action_probs = self.policy_net(state)
                    policy_loss = -self.value_net(state, action_probs).mean()
                    
                    optimizer_policy.zero_grad()
                    policy_loss.backward()
                    optimizer_policy.step()
                    
                    total_loss += value_loss.item()
                except Exception as e:
                    logger.error(f"处理数据行失败: {e}, row: {row.to_dict()}")
                    continue
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(offline_data):.4f}")
    
    def schedule(self, state, station_id):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state)
        action_idx = torch.argmax(action_probs).item()
        if action_idx == 0:
            return {"from_station": None, "to_station": None, "bikes": 0}
        else:
            station_idx = self.stations.index(station_id)
            neighbor_indices = np.where(self.adj_matrix[station_idx] > 0)[0]
            if len(neighbor_indices) == 0:
                return {"from_station": None, "to_station": None, "bikes": 0}
            target_idx = neighbor_indices[np.random.choice(len(neighbor_indices))]
            target_station = self.stations[target_idx]
            return {"from_station": station_id, "to_station": target_station, "bikes": 5}

# 动态更新库存
def update_inventory(status_df, schedule_actions, station_capacities):
    inventory = {sid: status_df[status_df['station_id'] == sid]['stock'].iloc[0] 
                 for sid in status_df['station_id'].unique()}
    updated_status = status_df.copy()
    for _, action in schedule_actions.iterrows():
        action_data = json.loads(action['schedule_action']) if isinstance(action['schedule_action'], str) else action['schedule_action']
        if action_data['bikes'] > 0:
            from_station = action_data['from_station']
            to_station = action_data['to_station']
            bikes = action_data['bikes']
            # 更新库存
            inventory[from_station] = max(0, inventory[from_station] - bikes)
            inventory[to_station] = min(station_capacities.get(to_station, 20), 
                                       inventory[to_station] + bikes)
            # 更新 status_df
            updated_status.loc[(updated_status['station_id'] == from_station) & 
                              (updated_status['date'] == action['date']) & 
                              (updated_status['hour'] == int(action['hour'].split(':')[0])), 
                              'stock'] = inventory[from_station]
            updated_status.loc[(updated_status['station_id'] == to_station) & 
                              (updated_status['date'] == action['date']) & 
                              (updated_status['hour'] == int(action['hour'].split(':')[0])), 
                              'stock'] = inventory[to_station]
    return updated_status

# 可视化
def plot_inventory_forecast(status_df, station_id):
    df = status_df[status_df['station_id'] == station_id]
    times = [f"{row['hour']:02d}:00" for _, row in df.iterrows()]
    inventories = [row['stock'] for _, row in df.iterrows()]
    plt.plot(times, inventories, marker='o', label=f"Station {station_id}")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title(f"Inventory Forecast for {station_id}")
    plt.legend()
    plt.show()

# 主程序
def main():
    # SQLAlchemy 连接
    try:
        engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        exit(1)
    
    # 加载数据
    try:
        with engine.connect() as conn:
            status_df = pd.read_sql("SELECT * FROM station_hourly_status", conn)
            offline_data = pd.read_sql("SELECT * FROM offline_rl_data", conn)
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        exit(1)
    
    try:
        adj_matrix = np.load("./handle/forecast/adj_matrix.npy")
    except Exception as e:
        logger.error(f"加载邻接矩阵失败: {e}")
        exit(1)
    
    stations = status_df['station_id'].unique()
    
    # 加载站点容量
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT station_id, capacity FROM station_info WHERE capacity IS NOT NULL"))
            station_capacities = {row[0]: row[1] for row in result.fetchall()}
        default_capacity = 20
        for sid in stations:
            if sid not in station_capacities:
                station_capacities[sid] = default_capacity
    except Exception as e:
        logger.error(f"加载站点容量失败: {e}")
        exit(1)
    
    # 初始化模型
    state_dim = 4  # 库存、流入、流出、小时
    action_dim = len(stations) + 1  # 包括不调度
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim, action_dim)
    scheduler = OfflineRLScheduler(policy_net, value_net, adj_matrix, station_capacities, stations)
    
    # 训练
    try:
        scheduler.train(offline_data)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        exit(1)
    
    # 生成调度方案
    schedule_results = []
    for station_id in stations:
        df = status_df[status_df['station_id'] == station_id]
        for _, row in df.iterrows():
            state = [row['stock'], row['inflow'], row['outflow'], row['hour']]
            action = scheduler.schedule(state, station_id)
            result = {
                "station_id": station_id,
                "date": str(row['date']),
                "hour": f"{row['hour']:02d}:00",
                "inflow": int(row['inflow']),
                "outflow": int(row['outflow']),
                "stock": int(row['stock']),
                "schedule_action": action,
                "alert": {
                    "low_inventory_warning": row['stock'] < 5,
                    "critical_time": f"{row['hour']:02d}:00" if row['stock'] < 5 else None
                }
            }
            schedule_results.append(result)
    
    # 转换为 DataFrame
    schedule_df = pd.DataFrame(schedule_results)
    
    # 更新库存
    try:
        updated_status_df = update_inventory(status_df, schedule_df, station_capacities)
    except Exception as e:
        logger.error(f"更新库存失败: {e}")
        exit(1)
    
    # 保存调度结果
    # 保存调度结果
    try:
        with engine.connect() as conn:
            insert_sql = """
            INSERT INTO station_schedule
            (station_id, date, hour, schedule_action, alert, updated_at)
            VALUES (:station_id, :date, :hour, :schedule_action, :alert, :updated_at)
            """
            insert_values = []
            for r in schedule_results:
                insert_values.append({
                    "station_id": r['station_id'],
                    "date": r['date'],
                    "hour": int(r['hour'].split(':')[0]),
                    "schedule_action": json.dumps(r['schedule_action']) if isinstance(r['schedule_action'], dict) else r['schedule_action'],
                    "alert": json.dumps(r['alert']) if isinstance(r['alert'], dict) else r['alert'],
                    "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            conn.execute(text(insert_sql), parameters=insert_values)

            # 更新 station_hourly_status 的库存
            update_sql = """
            UPDATE station_hourly_status
            SET stock = :stock
            WHERE station_id = :station_id AND date = :date AND hour = :hour
            """
            update_values = [{
                "stock": row['stock'],
                "station_id": row['station_id'],
                "date": row['date'],
                "hour": row['hour']
            } for _, row in updated_status_df.iterrows()]
            conn.execute(text(update_sql), parameters=update_values)
            conn.commit()
    except Exception as e:
        logger.error(f"保存调度结果失败: {e}")
        exit(1)

    
    # 保存 JSON
    try:
        with open("station_schedule.json", "w", encoding='utf-8') as f:
            json.dump(schedule_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存 JSON 文件失败: {e}")
        exit(1)
    
    # 可视化（示例：第一个站点）
    try:
        plot_inventory_forecast(updated_status_df, stations[0])
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        exit(1)

if __name__ == "__main__":
    main()