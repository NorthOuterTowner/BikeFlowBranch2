import sys
import numpy as np
import torch
from sqlalchemy import create_engine, text
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse

# 确保能导入STGCN模型
sys.path.append('./handle/forecast')  # 添加模型所在路径
from stgcn_model import STGCN  # 现在应该能正确导入了


# 配置
sys.stdout.reconfigure(encoding='utf-8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_predict():
    """加载数据并执行预测"""
    print("加载数据...")
    try:
        data = np.load('./handle/forecast/stgcn_dataset.npz')
        X_test = data['X_test'][:100]
        
        # 加载邻接矩阵
        A = np.load('./handle/forecast/adj_matrix.npy')
        edge_index, _ = dense_to_sparse(torch.tensor(A, dtype=torch.float32).to(device))
        
        # 初始化模型
        model = STGCN(num_nodes=A.shape[0], in_channels=2, out_channels=2).to(device)
        model.load_state_dict(torch.load('./handle/forecast/stgcn_model_best.pth'))
        model.eval()
        
        # 时间处理
        with open('./handle/forecast/all_times.json', 'r', encoding='utf-8') as f:
            all_times = json.load(f)
        test_start = data['X_train'].shape[0] + data['X_val'].shape[0]
        predict_times = all_times[test_start:test_start+100]
        
        hours = torch.tensor([
            datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in predict_times
        ], dtype=torch.long).to(device)
        days = torch.tensor([
            datetime.strptime(t, '%Y-%m-%d %H:%M:%S').weekday() for t in predict_times
        ], dtype=torch.long).to(device)
        
        # 执行预测
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).permute(0,3,2,1).to(device)
            outputs = model(inputs, edge_index, None, hours, days)
            
            # 检查点1: 模型原始输出
            print("\n[检查点1] 模型原始输出统计:")
            print(f"形状: {outputs.shape}")
            print(f"均值: {outputs.mean().item():.4f}")
            print(f"标准差: {outputs.std().item():.4f}")
            print(f"数值范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"零值比例: {(outputs == 0).float().mean().item():.2%}")
            
            preds = outputs.permute(0,3,2,1).cpu().numpy().squeeze(1)
        
        # 反归一化
        # 按站点反归一化
         # 检查点2: 转置后的输出
        print("\n[检查点2] 转置后输出统计:")
        print(f"形状: {preds.shape}")
        print(f"均值: {np.mean(preds):.4f}")
        print(f"数值范围: [{np.min(preds):.4f}, {np.max(preds):.4f}]")
        # 修改反归一化代码
        with open('./handle/forecast/normalization.json', 'r') as f:
            norm = json.load(f)
            
        inflow_min = np.array(norm['inflow_min'])
        inflow_max = np.array(norm['inflow_max'])
        outflow_min = np.array(norm['outflow_min'])
        outflow_max = np.array(norm['outflow_max'])
        print("\n[检查点3] 归一化参数示例:")
        print(f"流入范围: min={inflow_min[:5]}, max={inflow_max[:5]}...")
        print(f"流出范围: min={outflow_min[:5]}, max={outflow_max[:5]}...")

        # 确保范围有效
        inflow_range = np.clip(inflow_max - inflow_min, 1e-4, None)
        outflow_range = np.clip(outflow_max - outflow_min, 1e-4, None)

        for s in range(preds.shape[1]):
            preds[:, s, 0] = preds[:, s, 0] * (inflow_max[s] - inflow_min[s]) + inflow_min[s]
            preds[:, s, 1] = preds[:, s, 1] * (outflow_max[s] - outflow_min[s]) + outflow_min[s]
        
        # 检查点4: 反归一化后
        print("\n[检查点4] 反归一化后统计:")
        print(f"流入 - 均值: {np.mean(preds[:,:,0]):.4f}, 范围: [{np.min(preds[:,:,0]):.4f}, {np.max(preds[:,:,0]):.4f}]")
        print(f"流出 - 均值: {np.mean(preds[:,:,1]):.4f}, 范围: [{np.min(preds[:,:,1]):.4f}, {np.max(preds[:,:,1]):.4f}]")
        
        # 后处理
        preds = np.round(np.clip(preds, 0, None), 2)
        
        # 检查点5: 后处理后
        print("\n[检查点5] 后处理后统计:")
        print(f"流入 - 均值: {np.mean(preds[:,:,0]):.4f}, 零值比例: {np.mean(preds[:,:,0] == 0):.2%}")
        print(f"流出 - 均值: {np.mean(preds[:,:,1]):.4f}, 零值比例: {np.mean(preds[:,:,1] == 0):.2%}")
        
        return preds, predict_times
    
    except Exception as e:
        print(f"加载或预测过程中出错: {str(e)}")
        raise

def analyze_predictions(preds, station_ids):
    """增强版数据分析"""
    print("\n=== 预测深度分析 ===")
    
    # 计算每个站点的流量波动率 (标准差/均值)
    flow_ratios = []
    for s in range(preds.shape[1]):
        inflow = preds[:,s,0]
        outflow = preds[:,s,1]
        if np.mean(inflow) > 0:
            ratio = np.std(inflow) / np.mean(inflow)
            flow_ratios.append((ratio, s, 'inflow'))
        if np.mean(outflow) > 0:
            ratio = np.std(outflow) / np.mean(outflow)
            flow_ratios.append((ratio, s, 'outflow'))
    
    # 按波动率排序
    flow_ratios.sort(reverse=True, key=lambda x: x[0])
    
    print("\n波动最大的5个站点流量:")
    for ratio, s, flow_type in flow_ratios[:5]:
        station_id = station_ids[s]
        values = preds[:,s,0] if flow_type == 'inflow' else preds[:,s,1]
        print(f"  站点 {station_id} {flow_type}: "
              f"波动率={ratio:.2f}, "
              f"范围[{np.min(values):.1f}-{np.max(values):.1f}]")
    
    # 检查数据多样性
    unique_inflows = len(np.unique(preds[:,:,0].round(1)))
    unique_outflows = len(np.unique(preds[:,:,1].round(1)))
    print(f"\n数据多样性检查: 唯一流入值={unique_inflows}, 唯一流出值={unique_outflows}")
    
    if unique_inflows < 10 or unique_outflows < 10:
        print("警告: 预测结果多样性不足，可能存在模型退化问题！")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(preds, Y_test, predict_times, station_ids):
    """
    评估STGCN预测结果，分析零值分布、潮汐效应和预测质量

    参数:
    - preds: 预测结果，形状为 (时间点, 站点数, 2)，2表示[inflow, outflow]
    - Y_test: 测试集真实值，形状为 (时间点, 站点数, 2)
    - predict_times: 预测时间戳列表
    - station_ids: 站点ID列表

    输出:
    - 零值分析结果
    - 潮汐效应分析图
    - 预测质量指标（MAE, RMSE）
    - 可视化图表
    """
    print("\n=== 预测结果评估 ===")

    # 确保输入数据形状匹配
    if preds.shape != Y_test.shape:
        print(f"警告: 预测数据形状 {preds.shape} 与真实数据形状 {Y_test.shape} 不匹配")
        return

    # 转换为numpy数组并确保非负
    preds = np.clip(preds, 0, None)
    Y_test = np.clip(Y_test, 0, None)
    num_times, num_stations, num_features = preds.shape

    # 提取小时信息
    hours = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').hour for t in predict_times]
    hours = np.array(hours)

    # === 零值分析 ===
    print("\n[零值分析]")
    # 整体零值比例
    inflow_zero_ratio = np.mean(preds[:, :, 0] == 0)
    outflow_zero_ratio = np.mean(preds[:, :, 1] == 0)
    print(f"整体零值比例 - 流入: {inflow_zero_ratio:.2%}, 流出: {outflow_zero_ratio:.2%}")

    # 按站点统计零值比例
    station_zero_ratios = []
    for s in range(num_stations):
        inflow_zero = np.mean(preds[:, s, 0] == 0)
        outflow_zero = np.mean(preds[:, s, 0] == 0)
        station_zero_ratios.append((inflow_zero, outflow_zero, station_ids[s]))
    station_zero_ratios.sort(reverse=True, key=lambda x: x[0])  # 按流入零值比例排序
    print("\n零值比例最高的5个站点:")
    for inflow_z, outflow_z, sid in station_zero_ratios[:5]:
        print(f"站点 {sid}: 流入零值比例={inflow_z:.2%}, 流出零值比例={outflow_z:.2%}")

    # 按小时统计零值比例
    hourly_zero_ratios = []
    for h in range(24):
        mask = hours == h
        if np.sum(mask) > 0:
            inflow_zero = np.mean(preds[mask, :, 0] == 0)
            outflow_zero = np.mean(preds[mask, :, 1] == 0)
            hourly_zero_ratios.append((h, inflow_zero, outflow_zero))
    print("\n零值比例最高的5个小时:")
    for h, inflow_z, outflow_z in sorted(hourly_zero_ratios, key=lambda x: x[1], reverse=True)[:5]:
        print(f"小时 {h}: 流入零值比例={inflow_z:.2%}, 流出零值比例={outflow_z:.2%}")

    # 零值热力图
    zero_heatmap = np.zeros((24, num_stations))
    for h in range(24):
        mask = hours == h
        if np.sum(mask) > 0:
            zero_heatmap[h, :] = np.mean(preds[mask, :, 0] == 0, axis=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(zero_heatmap, cmap='Reds', vmin=0, vmax=1)
    plt.title("按小时和站点的流入零值比例热力图")
    plt.xlabel("站点索引")
    plt.ylabel("小时")
    plt.show()

    # === 潮汐效应分析 ===
    print("\n[潮汐效应分析]")
    # 按小时聚合平均流量
    hourly_inflow_pred = np.zeros(24)
    hourly_outflow_pred = np.zeros(24)
    hourly_inflow_true = np.zeros(24)
    hourly_outflow_true = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if np.sum(mask) > 0:
            hourly_inflow_pred[h] = np.mean(preds[mask, :, 0])
            hourly_outflow_pred[h] = np.mean(preds[mask, :, 1])
            hourly_inflow_true[h] = np.mean(Y_test[mask, :, 0])
            hourly_outflow_true[h] = np.mean(Y_test[mask, :, 1])

    # 绘制潮汐效应曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(24), hourly_inflow_true, 'b-', label='真实流入')
    plt.plot(range(24), hourly_inflow_pred, 'b--', label='预测流入')
    plt.plot(range(24), hourly_outflow_true, 'r-', label='真实流出')
    plt.plot(range(24), hourly_outflow_pred, 'r--', label='预测流出')
    plt.title("按小时的平均流量（潮汐效应）")
    plt.xlabel("小时")
    plt.ylabel("平均流量")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 按站点分析潮汐效应（选择波动最大的站点）
    station_stds = np.std(preds[:, :, 0], axis=0)
    top_stations = np.argsort(station_stds)[-3:]  # 选择波动最大的3个站点
    plt.figure(figsize=(15, 5))
    for i, s in enumerate(top_stations):
        plt.subplot(1, 3, i+1)
        inflow_pred = [np.mean(preds[hours == h, s, 0]) for h in range(24)]
        outflow_pred = [np.mean(preds[hours == h, s, 1]) for h in range(24)]
        inflow_true = [np.mean(Y_test[hours == h, s, 0]) for h in range(24)]
        outflow_true = [np.mean(Y_test[hours == h, s, 1]) for h in range(24)]
        plt.plot(range(24), inflow_true, 'b-', label='真实流入')
        plt.plot(range(24), inflow_pred, 'b--', label='预测流入')
        plt.plot(range(24), outflow_true, 'r-', label='真实流出')
        plt.plot(range(24), outflow_pred, 'r--', label='预测流出')
        plt.title(f"站点 {station_ids[s]} 潮汐效应")
        plt.xlabel("小时")
        plt.ylabel("流量")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 预测质量评估 ===
    print("\n[预测质量评估]")
    mae_inflow = mean_absolute_error(preds[:, :, 0].flatten(), Y_test[:, :, 0].flatten())
    mae_outflow = mean_absolute_error(preds[:, :, 1].flatten(), Y_test[:, :, 1].flatten())
    rmse_inflow = np.sqrt(mean_squared_error(preds[:, :, 0].flatten(), Y_test[:, :, 0].flatten()))
    rmse_outflow = np.sqrt(mean_squared_error(preds[:, :, 1].flatten(), Y_test[:, :, 1].flatten()))
    print(f"MAE - 流入: {mae_inflow:.4f}, 流出: {mae_outflow:.4f}")
    print(f"RMSE - 流入: {rmse_inflow:.4f}, 流出: {rmse_outflow:.4f}")

    # 按小时分析误差
    hourly_mae_inflow = np.zeros(24)
    hourly_mae_outflow = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if np.sum(mask) > 0:
            hourly_mae_inflow[h] = mean_absolute_error(preds[mask, :, 0], Y_test[mask, :, 0])
            hourly_mae_outflow[h] = mean_absolute_error(preds[mask, :, 1], Y_test[mask, :, 1])
    plt.figure(figsize=(12, 6))
    plt.plot(range(24), hourly_mae_inflow, 'b-', label='流入MAE')
    plt.plot(range(24), hourly_mae_outflow, 'r-', label='流出MAE')
    plt.title("按小时的预测误差（MAE）")
    plt.xlabel("小时")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 按站点分析误差
    station_mae_inflow = np.mean(np.abs(preds[:, :, 0] - Y_test[:, :, 0]), axis=0)
    station_mae_outflow = np.mean(np.abs(preds[:, :, 1] - Y_test[:, :, 1]), axis=0)
    print("\n预测误差最大的5个站点:")
    top_error_stations = np.argsort(station_mae_inflow)[-5:]
    for s in top_error_stations:
        print(f"站点 {station_ids[s]}: 流入MAE={station_mae_inflow[s]:.4f}, 流出MAE={station_mae_outflow[s]:.4f}")

    print("\n=== 评估完成 ===")
def write_to_database(preds, predict_times):
    """更健壮的数据库写入"""
    print("\n准备写入数据库...")
    try:
        # 加载站点ID
        with open('./handle/forecast/station_ids.json', 'r') as f:
            station_ids = json.load(f)
        
        # 获取站点容量
        engine = create_engine('mysql+pymysql://zq:123456@localhost/traffic?charset=utf8mb4')
        with engine.connect() as conn:
            capacities = pd.read_sql("SELECT station_id, capacity FROM station_info", conn)
        capacity_map = capacities.set_index('station_id')['capacity'].fillna(20).astype(int).to_dict()
        
        # 准备数据
        records = []
        for t in range(preds.shape[0]):
            dt = datetime.strptime(predict_times[t], '%Y-%m-%d %H:%M:%S')
            for s in range(preds.shape[1]):
                records.append({
                    'station_id': station_ids[s],
                    'date': dt.date().isoformat(),
                    'hour': int(dt.hour),
                    'inflow': round(float(preds[t,s,0]), 2),
                    'outflow': round(float(preds[t,s,1]), 2),
                    'stock': int(capacity_map.get(station_ids[s], 20) * 0.5),
                    'updated_at': dt.strftime('%Y-%m-%d %H:%M:%S')
                })
        print("\n[检查点6] 即将写入的示例记录:")
        for i in range(min(3, len(records))):  # 打印前3条记录
            print(f"记录{i+1}: {records[i]}")
        
        # 分批写入
        with engine.begin() as conn:
            # 使用更高效的日期范围删除
            start_time = predict_times[0]
            end_time = predict_times[-1]
            conn.execute(text("""
                DELETE FROM station_hourly_status 
                WHERE CONCAT(date, ' ', LPAD(hour, 2, '0')) BETWEEN :start AND :end
            """), {'start': start_time[:13], 'end': end_time[:13]})
            
            # 分批插入
            insert_sql = text("""
                INSERT INTO station_hourly_status 
                (station_id, date, hour, inflow, outflow, stock, updated_at)
                VALUES (:station_id, :date, :hour, :inflow, :outflow, :stock, :updated_at)
                ON DUPLICATE KEY UPDATE
                inflow=VALUES(inflow), outflow=VALUES(outflow), stock=VALUES(stock), updated_at=VALUES(updated_at)
            """)
            
            batch_size = 100
            for i in range(0, len(records), batch_size):
                conn.execute(insert_sql, records[i:i+batch_size])
        
        print(f"成功写入 {len(records)} 条记录 (共 {preds.shape[0]} 个时间点 × {preds.shape[1]} 个站点)")
    
    except Exception as e:
        print(f"数据库写入失败: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 执行预测
        preds, predict_times = load_and_predict()
        
        # 加载站点ID
        with open('./handle/forecast/station_ids.json', 'r') as f:
            station_ids = json.load(f)
        
        # 加载测试集真实数据并调整形状
        data = np.load('./handle/forecast/stgcn_dataset.npz')
        Y_test = data['Y_test'][:100].squeeze(1)  # 移除时间步长维度，从 (100, 1, 86, 2) 变为 (100, 86, 2)
        
        # 分析结果
        analyze_predictions(preds, station_ids)
        
        # 调用评估函数
        evaluate_predictions(preds, Y_test, predict_times, station_ids)
        
        # 可视化前3个波动大的站点
        plt.figure(figsize=(15, 6))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.plot(preds[:,i,0], 'r-', label='流入')
            plt.plot(preds[:,i,1], 'b--', label='流出')
            plt.title(f'站点 {station_ids[i]} 流量预测')
            plt.xlabel('时间点')
            plt.ylabel('流量')
            plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 写入数据库
        write_to_database(preds, predict_times)
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        sys.exit(1)