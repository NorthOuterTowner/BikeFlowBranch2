# ./handle/forecast/evaluate.py

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
from stgcn_model import STGCN
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 参数 ===
timesteps = 12
num_features = 2  # inflow + outflow

# === 加载数据集 ===
print("加载验证/测试集...")
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_val = data['X_val']
Y_val = data['Y_val']
X_test = data['X_test']
Y_test = data['Y_test']

# 转为 PyTorch 格式
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 2, 1)  # [B, C, N, T]
Y_val = torch.tensor(Y_val, dtype=torch.float32).permute(0, 3, 2, 1)  # [B, C, N, T]
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 2, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 3, 2, 1)

# === 加载邻接矩阵 ===
A = np.load('./handle/forecast/adj_matrix.npy')
adj = torch.tensor(A, dtype=torch.float32)

# === 加载模型 ===
print("加载模型...")
num_nodes = A.shape[0]
model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features)
model.load_state_dict(torch.load('./handle/forecast/stgcn_model.pth'))
model.eval()

# === 评估函数 ===
def evaluate(model, X, Y, name='验证集'):
    with torch.no_grad():
        pred = model(X, adj)  # 输出: [B, C, N, T]
        pred = pred.permute(0, 3, 2, 1).numpy()  # -> [B, T, N, C]
        true = Y.permute(0, 3, 2, 1).numpy()     # -> [B, T, N, C]

    # 取出 inflow + outflow，展平成二维
    pred = pred.reshape(-1, num_features)
    true = true.reshape(-1, num_features)

    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)

    print(f"\n{name}评估结果:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MSE  : {mse:.4f}")

# === 执行评估 ===
evaluate(model, X_val, Y_val, name="验证集")
evaluate(model, X_test, Y_test, name="测试集")
