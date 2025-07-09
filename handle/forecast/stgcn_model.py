# ./handle/forecast/stgcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()
        self.spatial = nn.Linear(num_nodes, num_nodes)  # 可替换为 GCN
        self.temporal2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x, adj):
        # x: [batch, channels, nodes, time]
        x = self.temporal1(x)
        x = self.relu(x)
        x = torch.einsum('bcnl,nm->bcml', x, adj)  # GCN 图卷积
        x = self.temporal2(x)
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels, 64, num_nodes)
        self.block2 = STGCNBlock(64, 64, num_nodes)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(1, 1))

    def forward(self, x, adj):
        # x: [batch, channels, nodes, time]
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = self.final_conv(x)  # [batch, out_channels, nodes, time]
        x = x[:, :, :, -1:]     # 取最后一个时间步，形状变为 [batch, out_channels, nodes, 1]
        return x

