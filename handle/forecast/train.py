import sys
import time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import dense_to_sparse
from stgcn_model import STGCN
from datetime import datetime
import json
import matplotlib.pyplot as plt

# 清空显存
torch.cuda.empty_cache()

# 参数配置
timesteps = 24
predict_steps = 1
num_features = 2
num_epochs = 20  # 增加epoch数量
batch_size = 8  # 增大batch size
learning_rate = 0.001

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
print("加载数据集...")
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_train = data['X_train']
Y_train = data['Y_train']
X_val = data['X_val']
Y_val = data['Y_val']
X_test = data['X_test']
Y_test = data['Y_test']

# 转换为Tensor并转移到设备
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.float32).permute(0, 3, 2, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 2, 1).to(device)

print(f"\n数据统计:")
print(f"训练集 - 流入均值: {X_train[:,0].mean().item():.4f}, 流出均值: {X_train[:,1].mean().item():.4f}")
print(f"验证集 - 流入均值: {X_val[:,0].mean().item():.4f}, 流出均值: {X_val[:,1].mean().item():.4f}")

# 生成小时和星期索引
sample_times = json.load(open('./handle/forecast/all_times.json', 'r', encoding='utf-8'))
train_hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in sample_times[:X_train.shape[0]]]
train_days = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() for t in sample_times[:X_train.shape[0]]]
val_hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in sample_times[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]]]
val_days = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() for t in sample_times[X_train.shape[0]:X_train.shape[0]+X_val.shape[0]]]
train_hours = torch.tensor(train_hours, dtype=torch.long).to(device)
train_days = torch.tensor(train_days, dtype=torch.long).to(device)
val_hours = torch.tensor(val_hours, dtype=torch.long).to(device)
val_days = torch.tensor(val_days, dtype=torch.long).to(device)

# 加载邻接矩阵
A = np.load('./handle/forecast/adj_matrix.npy')
edge_index, edge_weight = dense_to_sparse(torch.tensor(A, dtype=torch.float32).to(device))

# 初始化模型
num_nodes = A.shape[0]
model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features).to(device)

# 优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
class WeightedMSELossWithDiversity(nn.Module):
    def __init__(self, inflow_weight=2.0, outflow_weight=1.0, diversity_weight=0.2):
        super().__init__()
        self.inflow_weight = inflow_weight
        self.outflow_weight = outflow_weight
        self.diversity_weight = diversity_weight

    def forward(self, pred, target):
        # [B, C, N, 1] -> [B, N, 1, C] -> squeeze: [B, N, C]
        pred = pred.permute(0, 2, 3, 1).squeeze(-2)
        target = target.permute(0, 2, 3, 1).squeeze(-2)

        inflow_loss = ((pred[..., 0] - target[..., 0]) ** 2).mean()
        outflow_loss = ((pred[..., 1] - target[..., 1]) ** 2).mean()
        base_loss = self.inflow_weight * inflow_loss + self.outflow_weight * outflow_loss

        # 多样性惩罚：鼓励标准差大
        station_std = torch.std(pred, dim=[0, 1]).mean()
        diversity_penalty = 1.0 / (station_std + 1e-6)

        return base_loss + self.diversity_weight * diversity_penalty

criterion = WeightedMSELossWithDiversity(inflow_weight=2.0)
# 学习率调度器
steps_per_epoch = len(X_train) // batch_size
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.02,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    pct_start=0.3
)

# 训练循环
print("\n开始训练...")
best_val_loss = float('inf')
patience = 10
counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    total_loss = 0
    batch_count = 0
    
    # 训练批次
    for i in range(0, X_train.shape[0], batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = Y_train[i:i+batch_size]
        batch_hours = train_hours[i:i+batch_size]
        batch_days = train_days[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_x, edge_index, edge_weight, batch_hours, batch_days)
        loss = criterion(outputs, batch_y)
        
        # 检查异常损失
        if torch.isnan(loss):
            print(f"警告: 第 {epoch+1} 轮第 {i//batch_size+1} 批出现NaN损失")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 打印批次信息
        #if (i//batch_size) % 50 == 0:
        #    print(f"Epoch {epoch+1} | Batch {i//batch_size+1} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 计算平均训练损失
    avg_train_loss = total_loss / batch_count
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, val_hours, val_days, Y_val),
            batch_size=batch_size*2,
            shuffle=False
        )
        for batch_x, batch_h, batch_d, batch_y in val_loader:
            outputs = model(batch_x, edge_index, edge_weight, batch_h, batch_d)
            val_loss += criterion(outputs, batch_y).item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 打印epoch信息
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 检查预测输出范围
    sample_output = outputs[0].cpu().detach().numpy()
    print(f"预测输出范围: [{sample_output.min():.4f}, {sample_output.max():.4f}]")
    print(f"零值比例: {(sample_output == 0).mean():.2%}")
    
    # 早停机制和模型保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), './handle/forecast/stgcn_model_best.pth')
        counter = 0
        print("模型已保存！")
    else:
        counter += 1
        if counter >= patience:
            print(f"\n早停触发！最佳验证损失: {best_val_loss:.4f}")
            break

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./handle/forecast/loss_curve.png')
plt.close()

print("\n训练完成！最佳模型已保存为 stgcn_model_best.pth")