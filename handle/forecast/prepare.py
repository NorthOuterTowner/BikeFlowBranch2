'''
构建张量
'''
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import pymysql

# 参数设置
timesteps = 12     # 输入过去12小时
predict_steps = 1  # 预测未来1小时
features = ['inflow', 'outflow']

# 连接数据库
print("正在连接数据库...")
conn = pymysql.connect(host='localhost', user='zq', password='123456', database='traffic', charset='utf8mb4')
print("数据库连接成功！")

# 读取数据
print("正在读取 station_hourly_flow 数据表...")
sql = "SELECT station_id, timestamp, inflow, outflow FROM station_hourly_flow ORDER BY timestamp"
df = pd.read_sql(sql, conn)
conn.close()
print(f"共读取 {len(df)} 条记录")

# 处理时间
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"时间范围：{df['timestamp'].min()} ~ {df['timestamp'].max()}")

# 获取所有站点
stations = df['station_id'].unique()
stations.sort()
station_idx = {sid: i for i, sid in enumerate(stations)}
print(f"总共站点数：{len(stations)}，示例站点ID：{stations[:5]}")

# 重建三维张量结构
print("开始构建 (时间 × 站点 × 特征) 张量...")
df.set_index(['timestamp', 'station_id'], inplace=True)
df = df.sort_index()

full_time_index = pd.date_range(start=df.index.get_level_values(0).min(),
                                end=df.index.get_level_values(0).max(),
                                freq='h')
print(f"时间步数共 {len(full_time_index)} 个（每小时一条）")

# 初始化张量：时间步 × 站点数 × 特征数
data_tensor = np.zeros((len(full_time_index), len(stations), len(features)))

# 填充张量
for t_idx, ts in enumerate(full_time_index):
    if t_idx % 500 == 0:  # 每处理500步输出一次
        print(f"正在处理时间步 {t_idx+1}/{len(full_time_index)}: {ts}")
    for sid in stations:
        try:
            row = df.loc[(ts, sid)]
            data_tensor[t_idx, station_idx[sid], 0] = row['inflow']
            data_tensor[t_idx, station_idx[sid], 1] = row['outflow']
        except KeyError:
            continue  # 缺失默认填0

print("数据张量构建完成！")
print(f"最终张量的 shape 为：{data_tensor.shape}")

'''
数据切分
'''
print("\n开始构造训练、验证、测试集...")
X = []
Y = []

total_steps = len(data_tensor)
for i in range(total_steps - timesteps - predict_steps + 1):
    x = data_tensor[i : i + timesteps]  # 取12小时
    y = data_tensor[i + timesteps : i + timesteps + predict_steps]  # 取1小时
    X.append(x)
    Y.append(y)

X = np.array(X)  # shape: (样本数, 12, 站点数, 2)
Y = np.array(Y)  # shape: (样本数, 1, 站点数, 2)

print("切分完成！")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# === 数据集划分：训练 70%，验证 20%，测试 10% ===
num_samples = len(X)
train_size = int(num_samples * 0.7)
val_size = int(num_samples * 0.2)

X_train = X[:train_size]
Y_train = Y[:train_size]
X_val = X[train_size:train_size+val_size]
Y_val = Y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
Y_test = Y[train_size+val_size:]

print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# === 保存为 npz 文件 ===
output_path = "./handle/forecast/stgcn_dataset.npz"
np.savez_compressed(output_path,
    X_train=X_train, Y_train=Y_train,
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test
)

print(f"\n数据集保存完成：{output_path}")

'''
邻接矩阵
'''
import pandas as pd
import numpy as np
import pymysql
import re
from math import radians, sin, cos, sqrt, atan2

# Haversine 距离函数（单位：km）
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# 合法站点ID正则规则（如 HB101）
pattern = re.compile(r'^[A-Z]{2}\d+$')
def is_valid_station_id(sid):
    if pd.isna(sid):
        return False
    return bool(pattern.match(str(sid)))

# 读取站点信息
conn = pymysql.connect(host='localhost', user='zq', password='123456', database='traffic', charset='utf8mb4')
df = pd.read_sql("SELECT station_id, lat, lng FROM station_info", conn)
conn.close()

# 清洗不合规站点ID
df = df[df['station_id'].apply(is_valid_station_id)]

# 排序一致性（和张量保持顺序一致）
df = df.sort_values('station_id').reset_index(drop=True)
stations = df['station_id'].tolist()
num_stations = len(stations)

# 构建邻接矩阵
A = np.zeros((num_stations, num_stations))
threshold_km = 1.0  # 最大连接距离（公里）

print("正在构建邻接矩阵...")
for i in range(num_stations):
    for j in range(num_stations):
        if i == j:
            A[i, j] = 1
        else:
            d = haversine(df.loc[i, 'lng'], df.loc[i, 'lat'], df.loc[j, 'lng'], df.loc[j, 'lat'])
            if d <= threshold_km:
                A[i, j] = 1  # 也可替换为 1/d

print("邻接矩阵构建完成，形状：", A.shape)
np.save("./handle/forecast/adj_matrix.npy", A)
print("已保存为 ./handle/forecast/adj_matrix.npy")
