'''
模型训练部分
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stgcn_model import STGCN

# === 参数设置 ===
timesteps = 12
predict_steps = 1
num_features = 2  # inflow + outflow
num_epochs = 20
batch_size = 64
learning_rate = 0.001

# === 读取数据 ===
print("加载数据集...")
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_train = data['X_train']   # shape: [样本数, 12, 节点数, 2]
Y_train = data['Y_train']   # shape: [样本数, 1, 节点数, 2]

X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 2, 1)  # -> [B, C, N, T]
Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 3, 2, 1)  # -> [B, C, N, T]

print(f"训练样本数: {X_train.shape[0]}, 输入 shape: {X_train.shape}, 标签 shape: {Y_train.shape}")

# === 加载邻接矩阵 ===
A = np.load('./handle/forecast/adj_matrix.npy')
adj = torch.tensor(A, dtype=torch.float32)

# === 初始化模型 ===
num_nodes = A.shape[0]
model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === 开始训练 ===
print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i in range(0, X_train.shape[0], batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = Y_train[i:i+batch_size]

        optimizer.zero_grad()
        output = model(x_batch, adj)

        # 输出和 y_batch 都是 [B, C, N, T]，T=1
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (X_train.shape[0] // batch_size)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# === 保存模型 ===
torch.save(model.state_dict(), './handle/forecast/stgcn_model.pth')
print("\n模型已保存为 stgcn_model.pth")
