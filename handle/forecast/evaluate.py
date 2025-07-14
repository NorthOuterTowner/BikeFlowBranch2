import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import json
from stgcn_model import STGCN
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime

# 参数
timesteps = 24
num_features = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据集
print("加载验证/测试集...")
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_val = data['X_val']
Y_val = data['Y_val']
X_test = data['X_test']
Y_test = data['Y_test']
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 3, 2, 1).to(device)

# 加载小时和星期索引
sample_times = json.load(open('./handle/forecast/all_times.json', 'r', encoding='utf-8'))
val_start = data['X_train'].shape[0]
val_end = val_start + X_val.shape[0]
test_end = val_end + X_test.shape[0]
val_hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in sample_times[val_start:val_end]]
val_days = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() for t in sample_times[val_start:val_end]]
test_hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in sample_times[val_end:test_end]]
test_days = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() for t in sample_times[val_end:test_end]]
val_hours = torch.tensor(val_hours, dtype=torch.long).to(device)
val_days = torch.tensor(val_days, dtype=torch.long).to(device)
test_hours = torch.tensor(test_hours, dtype=torch.long).to(device)
test_days = torch.tensor(test_days, dtype=torch.long).to(device)

# 加载归一化参数
with open('./handle/forecast/normalization.json', 'r', encoding='utf-8') as f:
    norm_params = json.load(f)
inflow_min, inflow_max = norm_params['inflow_min'], norm_params['inflow_max']
outflow_min, outflow_max = norm_params['outflow_min'], norm_params['outflow_max']

# 加载邻接矩阵
A = np.load('./handle/forecast/adj_matrix.npy')
edge_index, edge_weight = dense_to_sparse(torch.tensor(A, dtype=torch.float32).to(device))

# 加载模型
print("加载模型...")
num_nodes = A.shape[0]
model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features).to(device)
model.load_state_dict(torch.load('./handle/forecast/stgcn_model_best.pth'))
model.eval()

# 评估函数
# 修改后的评估函数
def evaluate(model, X, Y, hours, days, name='验证集', batch_size=8):
    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        dataset = torch.utils.data.TensorDataset(X, hours, days, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_x, batch_hours, batch_days, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_hours = batch_hours.to(device)
            batch_days = batch_days.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x, edge_index, edge_weight, batch_hours, batch_days)
            pred_list.append(output.cpu().numpy())
            true_list.append(batch_y.cpu().numpy())
    
    # 合并结果
    pred = np.concatenate(pred_list, axis=0)
    true = np.concatenate(true_list, axis=0)
    pred = pred.transpose(0, 3, 2, 1)  # [batch, time, node, feature]
    true = true.transpose(0, 3, 2, 1)
    
    # 转换为numpy数组
    inflow_min = np.array(norm_params['inflow_min'])
    inflow_max = np.array(norm_params['inflow_max'])
    outflow_min = np.array(norm_params['outflow_min'])
    outflow_max = np.array(norm_params['outflow_max'])
    
    # 按站点反归一化
    for s in range(pred.shape[2]):
        pred[:, :, s, 0] = pred[:, :, s, 0] * (inflow_max[s] - inflow_min[s]) + inflow_min[s]
        pred[:, :, s, 1] = pred[:, :, s, 1] * (outflow_max[s] - outflow_min[s]) + outflow_min[s]
        true[:, :, s, 0] = true[:, :, s, 0] * (inflow_max[s] - inflow_min[s]) + inflow_min[s]
        true[:, :, s, 1] = true[:, :, s, 1] * (outflow_max[s] - outflow_min[s]) + outflow_min[s]
    
    # 计算指标
    pred_flat = pred.reshape(-1, num_features)
    true_flat = true.reshape(-1, num_features)
    mse = mean_squared_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_flat, pred_flat)
    
    print(f"\n{name}评估结果（反归一化后）:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MSE  : {mse:.4f}")
    
    # 计算各站点预测差异
    station_diff = np.std(pred, axis=(0,1))  # [node, feature]
    print("\n各站点预测差异统计:")
    print(f"平均标准差 - 流入: {np.mean(station_diff[:,0]):.4f}")
    print(f"平均标准差 - 流出: {np.mean(station_diff[:,1]):.4f}")
    
    # 可视化
    for station_idx in range(min(5, num_nodes)):
        plt.figure(figsize=(10, 5))
        plt.plot(pred[0, :, station_idx, 0], label='Pred Inflow')
        plt.plot(true[0, :, station_idx, 0], label='True Inflow')
        #plt.title(f'Station {station_ids[station_idx]} Inflow')
        plt.xlabel('Time Steps')
        plt.ylabel('Flow')
        plt.legend()
        plt.show()
# 执行评估
evaluate(model, X_val, Y_val, val_hours, val_days, name="验证集", batch_size=8)
evaluate(model, X_test, Y_test, test_hours, test_days, name="测试集", batch_size=8)
