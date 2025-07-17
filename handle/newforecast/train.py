import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from graph_wavenet import GraphWaveNet
from torch_geometric.utils import dense_to_sparse

# 加载数据
data = np.load("./dataset.npz")
X_train, Y_train = data['X_train'], data['Y_train']
X_val, Y_val = data['X_val'], data['Y_val']
adj_matrix = np.load("./adj_matrix.npy")
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float32))

# 转换为张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

# 维度检查
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

# 检查零值比例
print(f"Train zero ratio: inflow={np.mean(Y_train.cpu().numpy()[:, :, 0] == 0):.2%}, outflow={np.mean(Y_train.cpu().numpy()[:, :, 1] == 0):.2%}")
print(f"Val zero ratio: inflow={np.mean(Y_val.cpu().numpy()[:, :, 0] == 0):.2%}, outflow={np.mean(Y_val.cpu().numpy()[:, :, 1] == 0):.2%}")

# 模型和优化器
model = GraphWaveNet(num_nodes=86, in_channels=4, hidden_channels=128, num_layers=6, dropout=0.3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# 自定义损失
class ImprovedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = nn.HuberLoss()
    
    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"预测张量形状 {pred.shape} 与目标张量形状 {target.shape} 不匹配")
        main_loss = self.base_loss(pred, target)
        non_zero_reward = torch.mean((pred > 0).float())
        print(f"Loss components: main={main_loss.item():.4f}, non_zero={non_zero_reward.item():.4f}")
        return main_loss - 1.0 * non_zero_reward  # 移除 diversity_reward

criterion = ImprovedLoss()

# 训练循环
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience, early_stop_counter = 20, 0
for epoch in range(200):
    model.train()
    train_loss = 0
    for i in range(0, len(X_train), 8):
        x = X_train[i:i+8]
        y = Y_train[i:i+8]
        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(x)
    train_loss /= len(X_train)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_val), 8):
            x = X_val[i:i+8]
            y = Y_val[i:i+8]
            out = model(x, edge_index, edge_weight)
            loss = criterion(out, y)
            val_loss += loss.item() * len(x)
    val_loss /= len(X_val)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/200, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "graph_wavenet_best.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

print("训练完成！")