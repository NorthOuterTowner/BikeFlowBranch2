import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import json
from stgcn_model import STGCN
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from collections import Counter
import matplotlib.dates as mdates

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class STGCN_Evaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载数据
        self.load_data()
        # 加载模型
        self.load_model()
        
    def load_data(self):
        """加载数据集和相关信息"""
        print("加载数据集...")
        data = np.load('./handle/forecast/stgcn_dataset.npz')
        
        # 数据预处理
        self.X_val = torch.tensor(data['X_val'], dtype=torch.float32).permute(0, 3, 2, 1).to(self.device)
        self.Y_val = torch.tensor(data['Y_val'], dtype=torch.float32).permute(0, 3, 2, 1).to(self.device)
        self.X_test = torch.tensor(data['X_test'], dtype=torch.float32).permute(0, 3, 2, 1).to(self.device)
        self.Y_test = torch.tensor(data['Y_test'], dtype=torch.float32).permute(0, 3, 2, 1).to(self.device)
        
        # 加载时间信息
        self.sample_times = json.load(open('./handle/forecast/all_times.json', 'r', encoding='utf-8'))
        self.analyze_time_distribution(data['X_train'].shape[0])
        
        # 加载归一化参数
        with open('./handle/forecast/normalization.json', 'r', encoding='utf-8') as f:
            self.norm_params = json.load(f)
        
        # 加载邻接矩阵
        A = np.load('./handle/forecast/adj_matrix.npy')
        self.edge_index, self.edge_weight = dense_to_sparse(
            torch.tensor(A, dtype=torch.float32).to(self.device))
        self.num_nodes = A.shape[0]
    
    def analyze_time_distribution(self, train_size):
        """分析时间分布特征"""
        # 训练集时间分布
        train_hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour 
                      for t in self.sample_times[:train_size]]
        hour_counts = Counter(train_hours)
        
        plt.figure(figsize=(10, 5))
        plt.bar(hour_counts.keys(), hour_counts.values(), color='skyblue')
        plt.xlabel("小时 (0-23)")
        plt.ylabel("样本数量")
        plt.title("训练集小时分布")
        plt.grid(True)
        plt.savefig("./handle/forecast/train_hour_distribution.png")
        plt.close()
        
        # 验证/测试集时间范围
        val_start = train_size
        val_end = val_start + self.X_val.shape[0]
        test_end = val_end + self.X_test.shape[0]
        
        self.val_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') 
                         for t in self.sample_times[val_start:val_end]]
        self.test_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') 
                          for t in self.sample_times[val_end:test_end]]
        
        print("\n可用时间范围:")
        print(f"验证集: {self.val_times[0]} 至 {self.val_times[-1]}")
        print(f"测试集: {self.test_times[0]} 至 {self.test_times[-1]}")
    
    def load_model(self):
        """加载预训练模型"""
        print("加载STGCN模型...")
        self.model = STGCN(num_nodes=self.num_nodes, 
                          in_channels=2, 
                          out_channels=2).to(self.device)
        self.model.load_state_dict(torch.load('./handle/forecast/stgcn_model_best.pth'))
        self.model.eval()
    
    def denormalize(self, data, feature_idx, station_idx):
        """反归一化单个站点某个特征"""
        if feature_idx == 0:  # inflow
            min_val = self.norm_params['inflow_min'][station_idx]
            max_val = self.norm_params['inflow_max'][station_idx]
        else:  # outflow
            min_val = self.norm_params['outflow_min'][station_idx]
            max_val = self.norm_params['outflow_max'][station_idx]
        
        return data * (max_val - min_val) + min_val

    
    def evaluate(self, X, Y, times, name='验证集', batch_size=8):
        """完整评估流程"""
        self.model.eval()
        pred_list, true_list = [], []
        
        # 准备时间特征
        hours = torch.tensor([t.hour for t in times], dtype=torch.long).to(self.device)
        days = torch.tensor([t.weekday() for t in times], dtype=torch.long).to(self.device)
        
        # 批量预测
        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(X, hours, days, Y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            for batch_x, batch_h, batch_d, batch_y in loader:
                output = self.model(batch_x, self.edge_index, self.edge_weight, batch_h, batch_d)
                pred_list.append(output.cpu().numpy())
                true_list.append(batch_y.cpu().numpy())
        
        # 合并结果
        pred = np.concatenate(pred_list, axis=0).transpose(0, 3, 2, 1)  # [batch, time, node, feature]
        true = np.concatenate(true_list, axis=0).transpose(0, 3, 2, 1)
        
        # 反归一化
        for s in range(self.num_nodes):
            pred[:, :, s, 0] = self.denormalize(pred[:, :, s, 0], 0, s)
            pred[:, :, s, 1] = self.denormalize(pred[:, :, s, 1], 1, s)
            true[:, :, s, 0] = self.denormalize(true[:, :, s, 0], 0, s)
            true[:, :, s, 1] = self.denormalize(true[:, :, s, 1], 1, s)

        # >>> 在这里统一把负值截断为 0（评估前） <<<
        pred = np.maximum(pred, 0)

        # 计算全局指标
        pred_flat = pred.reshape(-1, 2)
        true_flat = true.reshape(-1, 2)
        
        metrics = {
            'MAE': mean_absolute_error(true_flat, pred_flat),
            'RMSE': np.sqrt(mean_squared_error(true_flat, pred_flat)),
            'MSE': mean_squared_error(true_flat, pred_flat)
        }
        
        # 打印评估结果
        print(f"\n{name}评估结果（反归一化后）:")
        print(f"  MAE  : {metrics['MAE']:.4f}")
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  MSE  : {metrics['MSE']:.4f}")
        
        return pred, true, metrics
    
    def plot_station_timeslice(self, pred, true, station_idx=0, start_idx=0, end_idx=200, name='验证集'):
        """
        绘制某个站点在指定时间范围内的预测与真实流量趋势图（实线+平滑，无圆点）
        """
        # 获取该站点的预测和真实值并 flatten
        station_pred_in = pred[:, :, station_idx, 0].flatten()
        station_true_in = true[:, :, station_idx, 0].flatten()
        station_pred_out = pred[:, :, station_idx, 1].flatten()
        station_true_out = true[:, :, station_idx, 1].flatten()

        # 截断负值为0
        station_pred_in = np.maximum(station_pred_in, 0)
        station_pred_out = np.maximum(station_pred_out, 0)

        # 取时间范围
        if name == '验证集':
            time_list = self.val_times
        elif name == '测试集':
            time_list = self.test_times
        else:
            raise ValueError(f"未知数据集名称: {name}，必须为 '验证集' 或 '测试集'")

        time_points = time_list[start_idx:end_idx]

        # 截取y值
        y_true_in = station_true_in[start_idx:end_idx]
        y_pred_in = station_pred_in[start_idx:end_idx]
        y_true_out = station_true_out[start_idx:end_idx]
        y_pred_out = station_pred_out[start_idx:end_idx]

        # 简单平滑处理（window=3）
        def smooth(arr, window=3):
            return np.convolve(arr, np.ones(window)/window, mode='same')

        y_true_in_smooth = smooth(y_true_in)
        y_pred_in_smooth = smooth(y_pred_in)
        y_true_out_smooth = smooth(y_true_out)
        y_pred_out_smooth = smooth(y_pred_out)

        plt.figure(figsize=(16, 6))

        # 设置时间格式
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%m-%d %H:%M')

        # 流入趋势图
        plt.subplot(1, 2, 1)
        plt.plot(time_points, y_true_in_smooth, label='真实流入', color='red', linestyle='-', linewidth=2)
        plt.plot(time_points, y_pred_in_smooth, label='预测流入', color='blue', linestyle='-', linewidth=2)
        plt.title(f'站点{station_idx} - 流入趋势图 ({name})')
        plt.xlabel('时间')
        plt.ylabel('流入量')
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # 流出趋势图
        plt.subplot(1, 2, 2)
        plt.plot(time_points, y_true_out_smooth, label='真实流出', color='green', linestyle='-', linewidth=2)
        plt.plot(time_points, y_pred_out_smooth, label='预测流出', color='orange', linestyle='-', linewidth=2)
        plt.title(f'站点{station_idx} - 流出趋势图 ({name})')
        plt.xlabel('时间')
        plt.ylabel('流出量')
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'./handle/forecast/station_{station_idx}_slice_{name}_time.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    evaluator = STGCN_Evaluator()
    
    # 验证集评估
    print("\n" + "="*50)
    print("验证集评估")
    print("="*50)
    val_pred, val_true, val_metrics = evaluator.evaluate(
        evaluator.X_val, evaluator.Y_val, evaluator.val_times, "验证集")
    
    # 测试集评估
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)
    test_pred, test_true, test_metrics = evaluator.evaluate(
        evaluator.X_test, evaluator.Y_test, evaluator.test_times, "测试集")
    
    # 可视化示例
    evaluator.plot_station_timeslice(test_pred, test_true, station_idx=2, start_idx=0, end_idx=200, name='测试集')