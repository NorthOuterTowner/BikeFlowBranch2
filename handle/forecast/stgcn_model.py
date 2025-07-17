import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# stgcn_model.py 重要修改
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        # 扩大感受野
        self.temporal1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gcn = GCNConv(out_channels, out_channels)
        self.temporal2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 9), padding=(0, 4))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
        # 添加残差连接
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x, edge_index, edge_weight):
        residual = x
        if self.res_conv is not None:
            residual = self.res_conv(residual)
            
        x = F.relu(self.bn1(self.temporal1(x)))
        b, c, n, t = x.shape
        
        # GCN处理
        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        x = F.relu(self.gcn(x, edge_index, edge_weight))
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        
        x = F.relu(self.bn2(self.temporal2(x)))
        x = self.dropout(x)
        
        return x + residual  # 残差连接
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, time_dim=24, day_dim=7):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes  # 添加这一行
        self.time_embedding = nn.Embedding(time_dim, in_channels)
        self.day_embedding = nn.Embedding(day_dim, in_channels)
        self.block1 = STGCNBlock(in_channels, 64, num_nodes)
        self.block2 = STGCNBlock(64, 64, num_nodes)
        self.block3 = STGCNBlock(64, 64, num_nodes)
        self.block4 = STGCNBlock(64, 64, num_nodes)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(1, 1)) # 去掉Sigmoid，输出范围不限

    

    def forward(self, x, edge_index, edge_weight, hour_idx, day_idx):
        # 修改时间嵌入和日期嵌入的维度扩展方式
        time_emb = self.time_embedding(hour_idx).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        day_emb = self.day_embedding(day_idx).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        x = x + time_emb + day_emb
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)
        x = self.block3(x, edge_index, edge_weight)
        x = self.block4(x, edge_index, edge_weight)
        x = self.final_conv(x)
        x = x[:, :, :, -1:]  # 只保留最后一个时间步
        return x