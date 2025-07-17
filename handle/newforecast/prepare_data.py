import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import json

# 数据库连接（MySQL）
engine = create_engine("mysql+pymysql://zq:123456@localhost:3306/traffic")

# 读取站点信息
df_stations = pd.read_sql("SELECT station_id, lat, lng FROM station_info", engine)
stations = df_stations['station_id'].tolist()
station_idx = {sid: i for i, sid in enumerate(stations)}
num_stations = len(stations)

# 读取流量数据（2025-01-01 起）
df_flow = pd.read_sql("SELECT station_id, timestamp, inflow, outflow FROM station_hourly_flow WHERE timestamp >= '2025-01-01'", engine)
df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'])
df_flow = df_flow.set_index(['timestamp', 'station_id'])

# 创建完整时间索引（小时级）
start_time = datetime(2025, 1, 1)
end_time = df_flow.index.get_level_values(0).max()
full_time_index = pd.date_range(start_time, end_time, freq='H')

# 构建数据张量
data_tensor = np.zeros((len(full_time_index), num_stations, 4))  # [时间, 站点, 特征(inflow, outflow, hour, day)]
missing_stations = set()

for t_idx, ts in enumerate(full_time_index):
    if t_idx % 500 == 0:
        print(f"处理时间步 {t_idx+1}/{len(full_time_index)}: {ts}")
    hour = ts.hour / 24.0
    day = ts.weekday() / 7.0
    for sid in stations:
        idx = station_idx[sid]
        data_tensor[t_idx, idx, 2] = hour
        data_tensor[t_idx, idx, 3] = day
        try:
            row = df_flow.loc[(ts, sid)]
            data_tensor[t_idx, idx, 0] = row['inflow']
            data_tensor[t_idx, idx, 1] = row['outflow']
        except KeyError:
            missing_stations.add(sid)
            # 均值插值
            prev_idx = max(0, t_idx - 1)
            next_idx = min(len(full_time_index) - 1, t_idx + 1)
            prev_in = df_flow.loc[(full_time_index[prev_idx], sid), 'inflow'] if (full_time_index[prev_idx], sid) in df_flow.index else 0
            next_in = df_flow.loc[(full_time_index[next_idx], sid), 'inflow'] if (full_time_index[next_idx], sid) in df_flow.index else 0
            prev_out = df_flow.loc[(full_time_index[prev_idx], sid), 'outflow'] if (full_time_index[prev_idx], sid) in df_flow.index else 0
            next_out = df_flow.loc[(full_time_index[next_idx], sid), 'outflow'] if (full_time_index[next_idx], sid) in df_flow.index else 0
            data_tensor[t_idx, idx, 0] = (prev_in + next_in) / 2 if (prev_in + next_in) > 0 else 0
            data_tensor[t_idx, idx, 1] = (prev_out + next_out) / 2 if (prev_out + next_out) > 0 else 0

print(f"缺失数据的站点数: {len(missing_stations)}")

# 全局归一化
inflow_values = data_tensor[:, :, 0].flatten()
outflow_values = data_tensor[:, :, 1].flatten()
global_min = np.array([np.percentile(inflow_values[inflow_values > 0], 5), np.percentile(outflow_values[outflow_values > 0], 5)])
global_max = np.array([np.percentile(inflow_values, 95), np.percentile(outflow_values, 95)])
norm_range = global_max - global_min
for i in range(num_stations):
    data_tensor[:, i, 0] = (data_tensor[:, i, 0] - global_min[0]) / norm_range[0]
    data_tensor[:, i, 1] = (data_tensor[:, i, 1] - global_min[1]) / norm_range[1]

# 构建邻接矩阵（地理距离 + 流量相关性）
coords = df_stations[['lat', 'lng']].values
distances = cdist(coords, coords, metric='euclidean')
sigma2 = np.var(distances)
adj_matrix = np.exp(-distances**2 / sigma2)
flow_matrix = np.corrcoef(data_tensor[:, :, 0].T)  # 基于流入相关性
flow_matrix[np.isnan(flow_matrix)] = 0
adj_matrix = 0.5 * adj_matrix + 0.5 * flow_matrix
adj_matrix = np.where(adj_matrix > 0.1, adj_matrix, 0)  # 稀疏化

# 自适应邻接矩阵初始嵌入
node_embeddings = np.random.randn(num_stations, 10)

# 数据切分（输入 24 小时，预测 1 小时）
seq_length = 24
X, Y = [], []
for i in range(len(full_time_index) - seq_length):
    X.append(data_tensor[i:i+seq_length])
    Y.append(data_tensor[i+seq_length, :, :2])  # 只预测流入和流出
X = np.array(X)
Y = np.array(Y)

# 按 7:2:1 划分
train_ratio, val_ratio = 0.7, 0.2
n_samples = len(X)
n_train = int(n_samples * train_ratio)
n_val = int(n_samples * val_ratio)
X_train, Y_train = X[:n_train], Y[:n_train]
X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

# 保存数据
np.savez("./dataset.npz", X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
np.save("./adj_matrix.npy", adj_matrix)
np.save("./node_embeddings.npy", node_embeddings)
with open("./normalization.json", 'w') as f:
    json.dump({'inflow': {'min': global_min[0], 'max': global_max[0]}, 'outflow': {'min': global_min[1], 'max': global_max[1]}}, f)
with open("./station_ids.json", 'w') as f:
    json.dump(stations, f)
with open("./times.json", 'w') as f:
    json.dump([str(t) for t in full_time_index], f)

print("数据预处理完成！")