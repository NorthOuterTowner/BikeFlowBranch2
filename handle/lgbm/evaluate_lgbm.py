import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# 1. 加载数据和模型
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])
encoder = load('./handle/lgbm/station_id_encoder.pkl')
model_in = load('./handle/lgbm/inflow_model.pkl')
model_out = load('./handle/lgbm/outflow_model.pkl')

# 2. 确保station_id是数值类型（训练时编码后的格式）
df['station_id'] = df['station_id'].astype(int)  # 强制转换为int

# 3. 选择特定时间点
target_time = df['timestamp'].iloc[0]  # 示例：选择第一个时间点
time_slice = df[df['timestamp'] == target_time].copy()

# 4. 安全解码station_id（处理可能的无效值）
try:
    time_slice['station_id_decoded'] = encoder.inverse_transform(time_slice['station_id'])
except ValueError as e:
    print(f"解码错误: {e}")
    # 回退方案：直接使用原始ID
    time_slice['station_id_decoded'] = time_slice['station_id'].astype(str)

# 5. 预测
features = ['station_id', 'hour', 'dayofweek', 'is_weekend',
            'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
            'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3']
time_slice['pred_inflow'] = model_in.predict(time_slice[features])
time_slice['pred_outflow'] = model_out.predict(time_slice[features])

# 6. 可视化（按站点ID排序后显示）
time_slice = time_slice.sort_values('station_id_decoded')

plt.figure(figsize=(15, 6))

# 流入对比
plt.subplot(1, 2, 1)
plt.bar(range(len(time_slice)), time_slice['inflow_next'], 
        width=0.4, label='True Inflow', alpha=0.7, color='blue')
plt.bar([x + 0.4 for x in range(len(time_slice))], time_slice['pred_inflow'], 
        width=0.4, label='Predicted Inflow', alpha=0.7, color='red')
plt.title(f'Inflow Comparison\nTime: {target_time}')
plt.xlabel('Station Index')
plt.ylabel('Flow Count')
plt.xticks(range(len(time_slice)), time_slice['station_id_decoded'], rotation=90)
plt.legend()
plt.grid(True)

# 流出对比
plt.subplot(1, 2, 2)
plt.bar(range(len(time_slice)), time_slice['outflow_next'], 
        width=0.4, label='True Outflow', alpha=0.7, color='green')
plt.bar([x + 0.4 for x in range(len(time_slice))], time_slice['pred_outflow'], 
        width=0.4, label='Predicted Outflow', alpha=0.7, color='orange')
plt.title(f'Outflow Comparison\nTime: {target_time}')
plt.xlabel('Station Index')
plt.ylabel('Flow Count')
plt.xticks(range(len(time_slice)), time_slice['station_id_decoded'], rotation=90)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'./handle/lgbm/station_comparison_{target_time.strftime("%Y%m%d_%H")}.png')
plt.show()