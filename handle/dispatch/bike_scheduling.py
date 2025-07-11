# -*- coding: utf-8 -*-
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
from torch.optim.lr_scheduler import LambdaLR
import uuid
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.system('chcp 65001 > nul')
os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bike_scheduling.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 状态归一化
def normalize_state(state, capacity=20):
    stock, inflow, outflow, hour = state
    return [
        stock / capacity,
        inflow / 30.0,
        outflow / 30.0,
        hour / 23.0
    ]

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.bikes_fc = nn.Linear(128, 3)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        bikes_probs = torch.softmax(self.bikes_fc(x), dim=-1)
        return action_probs, bikes_probs

# 价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action, bikes):
        x = torch.cat([state, action, bikes.unsqueeze(-1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# 调度器
class OfflineRLScheduler:
    def __init__(self, policy_net, value_net, adj_matrix, station_capacities, stations):
        self.policy_net = policy_net
        self.value_net = value_net
        self.adj_matrix = adj_matrix
        self.station_capacities = station_capacities
        self.stations = list(stations)
        self.bikes_options = [0, 5, 10]
        self.ema_target = None
    
    def train(self, offline_data, gamma=0.99, epochs=100, batch_size=32):
        optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        optimizer_value = optim.Adam(self.value_net.parameters(), lr=0.00005, weight_decay=1e-5)
        warm_up_epochs = 10
        lr_lambda = lambda epoch: min((epoch + 1) / warm_up_epochs, 1.0)
        scheduler_policy = LambdaLR(optimizer_policy, lr_lambda)
        scheduler_value = LambdaLR(optimizer_value, lr_lambda)
        
        # 奖励归一化
        rewards = offline_data['reward'].values
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-6
        offline_data['reward'] = (rewards - reward_mean) / reward_std
        logger.info(f"Reward stats: mean={reward_mean:.4f}, std={reward_std:.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
        
        # 过滤无效数据
        valid_data = offline_data[offline_data['action'].apply(
            lambda x: json.loads(x)['to_station'] in self.stations or json.loads(x)['bikes'] == 0
        )]
        logger.info(f"Filtered offline_data: {len(valid_data)} valid rows out of {len(offline_data)}")
        if len(valid_data) < 1000:
            logger.error("有效训练数据不足，请运行 generate_rl_data.py")
            exit(1)
        
        batch_size = max(16, min(64, len(valid_data) // 100))
        logger.info(f"Dynamic batch_size: {batch_size}")
        
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for epoch in range(epochs):
            total_loss = 0
            invalid_count = 0
            for i in range(0, len(valid_data), batch_size):
                batch = valid_data[i:i + batch_size]
                valid_rows = []
                for _, row in batch.iterrows():
                    try:
                        action_dict = json.loads(row['action'])
                        if not isinstance(action_dict, dict) or 'bikes' not in action_dict or 'to_station' not in action_dict:
                            logger.warning(f"Invalid action format: {row['action']}, skipping row")
                            invalid_count += 1
                            continue
                        if action_dict['bikes'] != 0 and action_dict['to_station'] not in self.stations:
                            logger.warning(f"Invalid to_station: {action_dict['to_station']}, skipping row")
                            invalid_count += 1
                            continue
                        valid_rows.append((row, action_dict))
                    except Exception as e:
                        logger.warning(f"解析 action 失败: {e}, row: {row['action']}")
                        invalid_count += 1
                        continue
                
                if not valid_rows:
                    continue
                
                states, actions, rewards, next_states, action_indices, bikes_indices = [], [], [], [], [], []
                for row, action_dict in valid_rows:
                    try:
                        state = json.loads(row['state'])
                        states.append(normalize_state(state))
                        action_idx = 0 if action_dict['bikes'] == 0 else self.stations.index(action_dict['to_station']) + 1
                        action = torch.zeros(len(self.stations) + 1, dtype=torch.float32)
                        action[action_idx] = 1.0
                        actions.append(action)
                        rewards.append(row['reward'])
                        next_state = json.loads(row['next_state'])
                        next_states.append(normalize_state(next_state))
                        action_indices.append(action_idx)
                        bikes_indices.append(self.bikes_options.index(action_dict['bikes']))
                    except Exception as e:
                        logger.warning(f"处理数据行失败: {e}, row: {row.to_dict()}")
                        invalid_count += 1
                        continue
                
                if not states:
                    continue
                
                states = torch.tensor(states, dtype=torch.float32)  # (batch_size, state_dim)
                actions = torch.stack(actions)  # (batch_size, action_dim)
                rewards = torch.tensor(rewards, dtype=torch.float32)  # (batch_size,)
                next_states = torch.tensor(next_states, dtype=torch.float32)  # (batch_size, state_dim)
                action_idx = torch.tensor(action_indices, dtype=torch.long)
                bikes_idx = torch.tensor(bikes_indices, dtype=torch.long)
                
                try:
                    next_action_probs, next_bikes_probs = self.policy_net(next_states)
                    next_action_idx = torch.argmax(next_action_probs, dim=1)
                    next_bikes_idx = torch.argmax(next_bikes_probs, dim=1)
                    next_actions = torch.zeros_like(next_action_probs)
                    next_actions.scatter_(1, next_action_idx.unsqueeze(1), 1.0)
                    next_bikes = torch.tensor([self.bikes_options[idx] for idx in next_bikes_idx], dtype=torch.float32)
                    
                    targets = rewards + gamma * self.value_net(next_states, next_actions, next_bikes).squeeze(-1).detach()
                    ema_alpha = 0.9
                    if self.ema_target is None:
                        self.ema_target = targets.mean().item()
                    self.ema_target = ema_alpha * self.ema_target + (1 - ema_alpha) * targets.mean().item()
                    targets = targets.clamp(min=self.ema_target - 10, max=self.ema_target + 10)
                    
                    action_probs, bikes_probs = self.policy_net(states)
                    logger.debug(f"action_probs shape: {action_probs.shape}, action_idx: {action_idx}")
                    
                    predicted = self.value_net(states, actions, torch.tensor([self.bikes_options[idx] for idx in bikes_idx], dtype=torch.float32)).squeeze(-1)
                    value_loss = ((targets - predicted) ** 2).mean()
                    
                    optimizer_value.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.1)
                    optimizer_value.step()
                    
                    advantage = (targets - predicted).detach()
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
                    policy_loss = -torch.log(action_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)) * advantage
                    policy_loss = policy_loss.mean()
                    
                    optimizer_policy.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.1)
                    optimizer_policy.step()
                    
                    total_loss += value_loss.item() * len(valid_rows)
                except Exception as e:
                    logger.error(f"批量训练失败: {e}")
                    continue
            
            scheduler_policy.step()
            scheduler_value.step()
            avg_loss = total_loss / len(valid_data)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Invalid rows skipped: {invalid_count}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    def schedule(self, state, station_id):
        state = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)
        action_probs, bikes_probs = self.policy_net(state)
        action_idx = torch.argmax(action_probs, dim=1).item()
        bikes_idx = torch.argmax(bikes_probs, dim=1).item()
        bikes = self.bikes_options[bikes_idx]
        logger.debug(f"Station {station_id}: action_probs={action_probs.tolist()}, bikes_probs={bikes_probs.tolist()}, action_idx={action_idx}, bikes={bikes}")
        if action_idx == 0 or bikes == 0:
            return {"from_station": None, "to_station": None, "bikes": 0}
        else:
            station_idx = self.stations.index(station_id)
            neighbor_indices = np.where(self.adj_matrix[station_idx] > 0)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != station_idx]
            if len(neighbor_indices) == 0:
                logger.warning(f"Station {station_id} has no valid neighbors in adj_matrix")
                return {"from_station": None, "to_station": None, "bikes": 0}
            valid_neighbors = neighbor_indices + 1  # 动作索引从 1 开始
            if action_idx not in valid_neighbors:
                action_idx = 0
                bikes = 0
                logger.debug(f"Invalid action_idx {action_idx} for {station_id}, resetting to no action")
            if action_idx == 0:
                return {"from_station": None, "to_station": None, "bikes": 0}
            target_idx = neighbor_indices[np.random.choice(len(neighbor_indices))]
            target_station = self.stations[target_idx]
            return {"from_station": station_id, "to_station": target_station, "bikes": bikes}

# 动态更新库存
def update_inventory(status_df, schedule_actions, station_capacities):
    inventory = {sid: status_df[status_df['station_id'] == sid]['stock'].iloc[0] 
                 for sid in status_df['station_id'].unique()}
    updated_status = status_df.copy()
    for _, action in schedule_actions.iterrows():
        action_data = json.loads(action['schedule_action']) if isinstance(action['schedule_action'], str) else action['schedule_action']
        if action_data['bikes'] > 0:
            from_station = action_data.get('from_station')
            to_station = action_data.get('to_station')
            bikes = action_data['bikes']
            inventory[from_station] = max(0, inventory.get(from_station, 0) - bikes)
            inventory[to_station] = min(station_capacities.get(to_station, 20), 
                                       inventory.get(to_station, 0) + bikes)
            start_hour = int(action['hour'].split(':')[0])
            for h in range(start_hour, start_hour + 3):
                updated_status.loc[(updated_status['station_id'] == from_station) & 
                                   (updated_status['date'] == action['date']) & 
                                   (updated_status['hour'] == h), 
                                   'stock'] = inventory[from_station]
                updated_status.loc[(updated_status['station_id'] == to_station) & 
                                   (updated_status['date'] == action['date']) & 
                                   (updated_status['hour'] == h), 
                                   'stock'] = inventory[to_station]
    return updated_status

# 可视化
def plot_inventory_forecast(status_df, station_id):
    df = status_df[status_df['station_id'] == station_id]
    times = [f"{row['hour']:02d}:00" for _, row in df.iterrows()]
    inventories = [row['stock'] for _, row in df.iterrows()]
    plt.figure(figsize=(10, 6))
    plt.plot(times, inventories, marker='o', label=f'Station {station_id}')
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title(f"Inventory Forecast for {station_id}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 汇总调度方案
def print_schedule_summary(schedule_df, date, hour):
    df = schedule_df[(schedule_df['date'] == date) & (schedule_df['hour'] == f"{int(hour):02d}:00")]
    logger.info(f"\n调度选项 - {date} {hour}:00:")
    valid_count = len(df[df['schedule_action'].apply(lambda x: json.loads(x).get('bikes', 0) > 0 if isinstance(x, str) else x.get('bikes', 0) > 0)])
    logger.info(f"有效调度记录数: {valid_count}")
    if not schedule_df.empty:
        logger.info("调度详情（前10条）:")
        logger.info(schedule_df[['from_station', 'to_station', 'bikes', 'datetime']].head(10).to_string(index=False))
    for _, row in df.iterrows():
        action = json.loads(row['schedule_action']) if isinstance(row['schedule_action'], str) else row['schedule_action']
        if action.get('bikes', 0) > 0:
            logger.info(f"- 从 {action['from_station']} 运 {action['bikes']} 辆车到 {action['to_station']}")

# 验证调度结果
def validate_schedule(schedule_df, stations):
    invalid_actions = schedule_df[schedule_df['schedule_action'].apply(
        lambda x: (json.loads(x).get('from_station') not in stations and json.loads(x).get('from_station') is not None) or
                  (json.loads(x).get('to_station') not in stations and json.loads(x).get('to_station') is not None)
        if isinstance(x, str) else
        (x.get('from_station') not in stations and x.get('from_station') is not None) or
        (x.get('to_station') not in stations and x.get('to_station') is not None)
    )]
    if not invalid_actions.empty:
        logger.warning(f"检测到无效调度记录: {invalid_actions.shape[0]} 条")

def diagnose_undispatched(schedule_df):
    df = schedule_df.copy()
    
    # 提取调度动作中的 bike 数量
    df['bikes'] = df['schedule_action'].apply(
        lambda x: json.loads(x).get('bikes', 0) if isinstance(x, str) else x.get('bikes', 0)
    )
    
    # 筛选出库存<5 的行
    low_stock_df = df[df['stock'] < 5]
    
    # 其中未调度的
    no_action_df = low_stock_df[low_stock_df['bikes'] == 0]

    total_low_stock = len(low_stock_df)
    total_no_action = len(no_action_df)

    print(f"诊断报告：")
    print(f"总共有 {total_low_stock} 条记录库存小于5")
    print(f"其中有 {total_no_action} 条记录没有调度动作（bikes == 0）")
    
    if total_low_stock > 0:
        rate = total_no_action / total_low_stock * 100
        print(f"未调度占比为：{rate:.2f}%")
        if rate > 50:
            print("说明模型在库存低时未能主动调度，可能需要调整策略或奖励函数")
        else:
            print("多数低库存情况有响应调度，策略基本合理")
    else:
        print("无库存过低的记录，模型输出“无调度”是合理的")

    # 也可以返回这部分记录方便查看
    return no_action_df

# 主程序
def main():
    run_id = str(uuid.uuid4())
    try:
        engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')  # Update to correct DB name
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        exit(1)
    
    try:
        with engine.connect() as conn:
            status_df = pd.read_sql("SELECT * FROM station_hourly_status", conn)
            offline_data = pd.read_sql("SELECT * FROM offline_rl_data WHERE hour IN (0, 3, 6, 12, 15, 21)", conn)
            count = conn.execute(text("SELECT COUNT(*) FROM offline_rl_data")).fetchone()[0]
            if count < 10000:
                logger.warning(f"离线数据记录: {count}, 建议重新运行 generate_rl_data.py")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        exit(1)
    
    # Rest of the main function remains unchanged
    try:
        adj_matrix = np.load("./handle/forecast/adj_matrix.npy")
        np.fill_diagonal(adj_matrix, 0)
        logger.info("已移除 adj_matrix 自环")
    except Exception as e:
        logger.error(f"加载邻接矩阵失败: {e}")
        exit(1)
    
    stations = status_df['station_id'].unique()
    
    try:
        with engine.connect() as conn:
            result_df = pd.read_sql_query("SELECT * FROM station_info", conn)
            if result_df.empty:
                logger.error("station_info 表为空，请检查数据")
                exit(1)
            station_capacities = result_df.set_index('station_id')['capacity'].to_dict()
            logger.info(f"station_capacities 类型是：{type(station_capacities)}")
        default_capacity = 20
        for sid in stations:
            if not sid in station_capacities:
                station_capacities[sid] = default_capacity
    except Exception as e:
        logger.error(f"加载站点容量失败: {e}")
        exit(1)
    
    state_dim = 4
    action_dim = len(stations) + 1
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim, action_dim)
    scheduler = OfflineRLScheduler(policy_net, value_net, adj_matrix, station_capacities, stations)
    
    try:
        scheduler.train(offline_data)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        exit(1)
    
    status_df['datetime'] = pd.to_datetime(status_df['date']) + pd.to_timedelta(status_df['hour'], unit='h')
    agg_status = status_df.groupby(['station_id', pd.Grouper(key='datetime', freq='3h')]).agg({'stock': 'mean', 'inflow': 'sum', 'outflow': 'sum', 'hour': 'first'}).reset_index()
    agg_status['date'] = agg_status['datetime'].dt.date.astype(str)
    agg_status['time_slot'] = agg_status['datetime'].dt.hour
    
    schedule_results = []
    for station_id in stations:
        df = agg_status[agg_status['station_id'] == station_id]
        for _, row in df.iterrows():
            try:
                state = [row['stock'], row['inflow'], row['outflow'], row['hour']]
                action = scheduler.schedule(state, station_id)
                result = {
                    "station_id": station_id,
                    "date": str(row['date']),
                    "hour": f"{row['hour']:02d}:00",
                    "inflow": float(row['inflow']),
                    "outflow": float(row['outflow']),
                    "stock": float(row['stock']),
                    "schedule_action": action,
                    "alert": {
                        "low_inventory_alert": row['stock'] < 5,
                        "critical_time": f"{row['hour']:02d}:00" if row['stock'] < 5 else None
                    }
                }
                schedule_results.append(result)
            except Exception as e:
                logger.error(f"生成调度失败: {station_id}: {e}")
                continue
    
    schedule_df = pd.DataFrame(schedule_results)
    validate_schedule(schedule_df, stations)
    
    try:
        updated_status_df = update_inventory(status_df, schedule_df, station_capacities)
    except Exception as e:
        logger.error(f"更新库存失败: {e}")
        exit(1)
    
    try:
        with engine.connect() as conn:
            logger.info("插入 station_schedule，保留现有数据")
            insert_sql = """
            INSERT INTO station_schedule
            (station_id, date, hour, schedule_action, alert, updated_at)
            VALUES (:station_id, :date, :hour, :schedule_action, :alert, :updated_at)
            ON DUPLICATE KEY UPDATE
                schedule_action = VALUES(schedule_action),
                alert = VALUES(alert),
                updated_at = VALUES(updated_at)
            """
            values = []
            for r in schedule_results:
                values.append({
                    "station_id": r['station_id'],
                    "date": r['date'],
                    "hour": int(r['hour'].split(':')[0]),
                    "schedule_action": json.dumps(r['schedule_action']) if isinstance(r['schedule_action'], dict) else r['schedule_action'],
                    "alert": json.dumps(r['alert']) if isinstance(r['alert'], dict) else r['alert'],
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            conn.execute(text(insert_sql), parameters=values)
            update_sql = """
            UPDATE station_hourly_status
            SET stock = :stock
            WHERE station_id = :station_id AND date = :date AND hour = :hour
            """
            update_values = [{
                "stock": row['stock'],
                "station_id": row['station_id'],
                "date": str(row['date']),
                "hour": int(row['hour'])
            } for _, row in updated_status_df.iterrows()]
            conn.execute(text(update_sql), parameters=update_values)
            conn.commit()
    except Exception as e:
        logger.error(f"保存调度结果失败: {e}")
        exit(1)
    
    try:
        with open(f"station_schedule_{run_id}.json", "w", encoding='utf-8') as f:
            json.dump(schedule_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存 JSON 文件失败: {e}")
        exit(1)
    
    try:
        print_schedule_summary(schedule_df, "2025-06-12", 6)
    except Exception as e:
        logger.error(f"打印调度方案错误: {e}")
    
    try:
        plot_inventory_forecast(updated_status_df, stations[0])
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        exit(1)

    diagnose_undispatched(schedule_df)

if __name__ == '__main__':
    main()