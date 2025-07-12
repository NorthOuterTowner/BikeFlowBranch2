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
        
        # 分析结果
        analyze_predictions(preds, station_ids)
        
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