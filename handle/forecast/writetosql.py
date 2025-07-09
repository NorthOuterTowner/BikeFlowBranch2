import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
import pymysql
from datetime import datetime
import json
from stgcn_model import STGCN

# MySQL 连接
conn = pymysql.connect(
    host='localhost',
    user='zq',
    password='123456',
    db='traffic',
    charset='utf8mb4'
)
cursor = conn.cursor()

# 加载邻接矩阵与模型
A = np.load('./handle/forecast/adj_matrix.npy')
adj = torch.tensor(A, dtype=torch.float32)

num_nodes = A.shape[0]
num_features = 2

model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features)
model.load_state_dict(torch.load('./handle/forecast/stgcn_model.pth'))
model.eval()

# 加载预测样本
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_test = data['X_test']
X_pred = X_test[:100]  # 扩展为 100 个样本

with open('./handle/forecast/station_ids.json', 'r', encoding='utf-8') as f:
    station_ids = json.load(f)

assert len(station_ids) == num_nodes, "站点数不匹配！"

# 加载预测时间点
with open('./handle/forecast/predict_times.json', 'r', encoding='utf-8') as f:
    predict_times = json.load(f)

predict_times = predict_times[:len(X_pred)]  # 匹配 X_pred 长度
print("\n=== 选取的预测时间点（来自 predict_times.json） ===")
for i, ts in enumerate(predict_times):
    print(f"样本 {i+1}: 预测时间 = {ts}")

# 模型预测
X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).permute(0, 3, 2, 1)  # [B, C, N, T]
with torch.no_grad():
    output = model(X_pred_tensor, adj)  # [B, C, N, T=1]
output_np = output.permute(0, 3, 2, 1).numpy().squeeze(1)  # [B, N, C]

# 获取站点容量
cursor.execute("SELECT station_id, capacity FROM station_info")
station_capacities = {row[0]: row[1] if row[1] is not None else 20 for row in cursor.fetchall()}
default_capacity = 20

# 初始化库存（假设每个站点初始库存为 capacity // 2）
initial_inventory = {sid: station_capacities.get(sid, default_capacity) // 2 for sid in station_ids}
current_date = None
inventory = initial_inventory.copy()

# 删除原有记录
predict_dates_hours = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').date(), 
                       datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').hour) 
                      for ts in predict_times]
delete_sql = f"""
DELETE FROM station_hourly_status
WHERE (date, hour) IN ({','.join(['(%s, %s)'] * len(predict_dates_hours))})
"""
delete_params = [item for pair in predict_dates_hours for item in pair]
cursor.execute(delete_sql, delete_params)
print(f"已删除原有记录 {len(predict_dates_hours)} 个小时")

# 构造新记录
insert_values = []
now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

for sample_idx in range(output_np.shape[0]):
    dt = datetime.strptime(predict_times[sample_idx], '%Y-%m-%d %H:%M:%S')
    predict_date = dt.date()
    predict_hour = dt.hour

    # 每天 0 点重置库存（可选，取决于业务需求）
    if current_date != predict_date:
        inventory = initial_inventory.copy()
        current_date = predict_date

    for node_idx, station_id in enumerate(station_ids):
        inflow = int(round(output_np[sample_idx, node_idx, 0]))
        outflow = int(round(output_np[sample_idx, node_idx, 1]))
        capacity = station_capacities.get(station_id, default_capacity)
        # 更新库存
        inventory[station_id] += inflow - outflow
        inventory[station_id] = max(0, min(inventory[station_id], capacity))
        insert_values.append((
            station_id,
            predict_date,
            predict_hour,
            inflow,
            outflow,
            int(inventory[station_id]),
            now_str
        ))

print("\n=== 写入数据库的时间点与样本对应关系 ===")
for sample_idx in range(output_np.shape[0]):
    dt = datetime.strptime(predict_times[sample_idx], '%Y-%m-%d %H:%M:%S')
    predict_date = dt.date()
    predict_hour = dt.hour
    print(f"样本 {sample_idx+1}: 写入时间 = {predict_date} {predict_hour:02d}:00")

# 插入数据库
insert_sql = """
INSERT INTO station_hourly_status
(station_id, date, hour, inflow, outflow, stock, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
inflow=VALUES(inflow), outflow=VALUES(outflow), stock=VALUES(stock), updated_at=VALUES(updated_at)
"""
cursor.executemany(insert_sql, insert_values)
conn.commit()

cursor.close()
conn.close()

print(f"预测完成，写入记录数: {len(insert_values)}, 预测样本数: {output_np.shape[0]}, 站点数: {num_nodes}")