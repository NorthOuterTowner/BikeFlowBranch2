'''
构建张量，划分数据集，邻接矩阵
'''
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import re
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体
try:
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 中文字体设置失败，将使用默认字体")

# 参数设置
timesteps = 24     # 输入过去24小时
predict_steps = 1  # 预测未来1小时
features = ['inflow', 'outflow']
train_ratio, val_ratio = 0.7, 0.2

# 连接数据库
print("正在连接数据库...")
engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')
print("数据库连接成功！")

# 读取数据
print("正在读取 station_hourly_flow 数据表...")
sql = "SELECT station_id, timestamp, inflow, outflow FROM station_hourly_flow ORDER BY timestamp"
df = pd.read_sql(sql, engine)
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

# 初始化张量
data_tensor = np.zeros((len(full_time_index), len(stations), len(features)))

# 填充张量
for t_idx, ts in enumerate(full_time_index):
    if t_idx % 500 == 0:
        print(f"正在处理时间步 {t_idx+1}/{len(full_time_index)}: {ts}")
    for sid in stations:
        try:
            row = df.loc[(ts, sid)]
            data_tensor[t_idx, station_idx[sid], 0] = row['inflow']
            data_tensor[t_idx, station_idx[sid], 1] = row['outflow']
        except KeyError:
            continue

# ========== 修改点1：按站点归一化 ==========
print("\n正在进行按站点归一化...")
# 修改归一化函数
# 替换原有归一化函数
def normalize_by_station(data):
    """改进的归一化方法，处理常数值站点"""
    # 使用百分位数避免极端值
    min_vals = np.percentile(data, 5, axis=0, keepdims=True)  # 5%分位数作为min
    max_vals = np.percentile(data, 95, axis=0, keepdims=True) # 95%分位数作为max
    
    # 处理常数值站点
    constant_mask = (max_vals - min_vals) < 1e-8
    max_vals[constant_mask] = min_vals[constant_mask] + 1.0  # 给常数值站点设置1的范围
    
    # 添加安全范围
    range_vals = np.clip(max_vals - min_vals, 1e-4, None)
    return (data - min_vals) / range_vals


# 对流入和流出分别按站点归一化
data_tensor[:, :, 0] = normalize_by_station(data_tensor[:, :, 0])  # 流入
data_tensor[:, :, 1] = normalize_by_station(data_tensor[:, :, 1])  # 流出

# 保存归一化参数（改为保存每个站点的参数）
norm_params = {
    'inflow_min': data_tensor[:, :, 0].min(axis=0).tolist(),
    'inflow_max': data_tensor[:, :, 0].max(axis=0).tolist(),
    'outflow_min': data_tensor[:, :, 1].min(axis=0).tolist(),
    'outflow_max': data_tensor[:, :, 1].max(axis=0).tolist()
}
print("\n关键站点归一化参数验证:")
sample_indices = [0, 10, -1]  # 检查首、中、尾站点
for i in sample_indices:
    print(f"站点{i}: 流入({norm_params['inflow_min'][i]:.2f}-{norm_params['inflow_max'][i]:.2f}) "
          f"流出({norm_params['outflow_min'][i]:.2f}-{norm_params['outflow_max'][i]:.2f})")
# ========================================

print("数据张量构建完成！")
print(f"最终张量的 shape 为：{data_tensor.shape}")

# 可视化检查
print("\n检查归一化后的数据分布:")
plt.figure(figsize=(12, 6))
sns.heatmap(data_tensor[:24, :, 0].T, cmap='viridis')  # 显示第一天各站点的流入情况
plt.title("归一化后各站点流入量分布", fontproperties=font_prop)
plt.xlabel("时间步", fontproperties=font_prop)
plt.ylabel("站点索引", fontproperties=font_prop)
plt.show()

# 数据切分
print("\n开始构造训练、验证、测试集...")
X, Y, sample_times = [], [], []
for i in range(len(data_tensor) - timesteps - predict_steps + 1):
    X.append(data_tensor[i:i+timesteps])
    Y.append(data_tensor[i+timesteps:i+timesteps+predict_steps])
    sample_times.append(str(full_time_index[i+timesteps]))

X = np.array(X)
Y = np.array(Y)
print("切分完成！")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# 数据集划分
num_samples = len(X)
train_size = int(num_samples * train_ratio)
val_size = int(num_samples * val_ratio)
X_train = X[:train_size]
Y_train = Y[:train_size]
X_val = X[train_size:train_size+val_size]
Y_val = Y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
Y_test = Y[train_size+val_size:]
all_times = sample_times

print(f"\n数据集划分结果:")
print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 保存数据集
output_path = "./handle/forecast/stgcn_dataset.npz"
np.savez_compressed(output_path,
    X_train=X_train, Y_train=Y_train,
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test
)
print(f"\n数据集保存完成：{output_path}")

# 保存所有时间戳
with open('./handle/forecast/all_times.json', 'w', encoding='utf-8') as f:
    json.dump(all_times, f, ensure_ascii=False)
print(f"所有时间戳已保存为 all_times.json，共 {len(all_times)} 条")

# 保存归一化参数
with open('./handle/forecast/normalization.json', 'w', encoding='utf-8') as f:
    json.dump(norm_params, f, ensure_ascii=False)
print("归一化参数已保存为 normalization.json")

# 保存站点ID
with open('./handle/forecast/station_ids.json', 'w', encoding='utf-8') as f:
    json.dump(stations.tolist(), f, ensure_ascii=False)
print("站点ID列表已保存为 station_ids.json")

# ========== 修改点2：改进邻接矩阵构建 ==========
print("\n正在构建改进版邻接矩阵...")

# 定义haversine距离计算函数
def haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度坐标之间的球面距离(km)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a)) 
    r = 6371  # 地球平均半径，单位km
    return c * r

def build_adjacency_matrix(stations_df, flow_data):
    """
    构建考虑以下因素的邻接矩阵：
    1. 地理距离
    2. 流量模式相似性
    3. 物理连接关系
    """
    n_stations = len(stations_df)
    A = np.zeros((n_stations, n_stations))
    
    # 1. 地理距离矩阵
    print("计算地理距离...")
    dist_matrix = np.zeros((n_stations, n_stations))
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                lon1, lat1 = stations_df.loc[i, 'lng'], stations_df.loc[i, 'lat']
                lon2, lat2 = stations_df.loc[j, 'lng'], stations_df.loc[j, 'lat']
                dist_matrix[i, j] = haversine(lon1, lat1, lon2, lat2)
    
    # 2. 流量相关性矩阵
    print("计算流量相关性...")
    inflow_corr = np.corrcoef(flow_data[:, :, 0].T)
    outflow_corr = np.corrcoef(flow_data[:, :, 1].T)
    flow_corr = (inflow_corr + outflow_corr) / 2
    flow_corr = np.nan_to_num(flow_corr, 0)  # 处理NaN
    
    # 3. 构建最终邻接矩阵
    print("组合矩阵...")
    sigma = 0.5  # 高斯核参数
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                A[i, j] = 1  # 自连接
            else:
                # 地理相似性 (高斯核)
                geo_sim = np.exp(-dist_matrix[i, j]**2 / (2 * sigma**2))
                
                # 流量模式相似性
                flow_sim = (flow_corr[i, j] + 1) / 2  # 转换到[0,1]
                
                # 组合权重 (几何平均)
                A[i, j] = np.sqrt(geo_sim * flow_sim)
    
    # 可视化检查
    plt.figure(figsize=(10, 8))
    sns.heatmap(A[:20, :20], cmap='YlOrRd')  # 显示前20个站点的关系
    plt.title("邻接矩阵示例 (前20个站点)", fontproperties=font_prop)
    plt.show()
    
    return A

# 加载站点坐标信息
df_stations = pd.read_sql("SELECT station_id, lat, lng FROM station_info", engine)
df_stations = df_stations[df_stations['station_id'].isin(stations)]
df_stations = df_stations.sort_values('station_id').reset_index(drop=True)

# 构建改进版邻接矩阵
A = build_adjacency_matrix(df_stations, data_tensor)
np.save("./handle/forecast/adj_matrix.npy", A)
print("\n改进版邻接矩阵构建完成，形状：", A.shape)
print("已保存为 ./handle/forecast/adj_matrix.npy")
# ===========================================

# 新增：检查数据多样性
print("\n数据多样性检查:")
print(f"各站点流入量均值差异: {data_tensor[:,:,0].mean(axis=0).std():.4f}")
print(f"各站点流出量均值差异: {data_tensor[:,:,1].mean(axis=0).std():.4f}")

# 在prepare.py中添加数据检查
print("\n原始数据统计:")
print(f"流入 - 均值: {data_tensor[:,:,0].mean():.4f}, 非零比例: {(data_tensor[:,:,0] > 0).mean():.2%}")
print(f"流出 - 均值: {data_tensor[:,:,1].mean():.4f}, 非零比例: {(data_tensor[:,:,1] > 0).mean():.2%}")

# 检查归一化后的数据
norm_data = normalize_by_station(data_tensor[:,:,0])
print("\n归一化后统计:")
print(f"最小值: {norm_data.min()}, 最大值: {norm_data.max()}")
print(f"零值比例: {(norm_data == 0).mean():.2%}")