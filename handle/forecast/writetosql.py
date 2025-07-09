import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from stgcn_model import STGCN
import pymysql
from datetime import datetime, timedelta
import json

# -----------------------------
# 1. MySQL 连接配置
conn = pymysql.connect(
    host='localhost',
    user='zq',
    password='123456',
    db='traffic',
    charset='utf8mb4'
)
cursor = conn.cursor()

# -----------------------------
# 2. 加载邻接矩阵和模型
A = np.load('./handle/forecast/adj_matrix.npy')
adj = torch.tensor(A, dtype=torch.float32)

num_nodes = A.shape[0]
num_features = 2

model = STGCN(num_nodes=num_nodes, in_channels=num_features, out_channels=num_features)
model.load_state_dict(torch.load('./handle/forecast/stgcn_model.pth'))
model.eval()

# -----------------------------
# 3. 加载数据集，取测试集前3条作为预测输入
data = np.load('./handle/forecast/stgcn_dataset.npz')
X_test = data['X_test']  # (样本数, 12, 节点数, 2)
X_pred = X_test[:3]      # 取3条样本做预测
with open('./handle/forecast/station_ids.json', 'r', encoding='utf-8') as f:
    station_ids = json.load(f)
print("加载到的站点ID示例：", station_ids[:5])

assert len(station_ids) == num_nodes, "站点ID数量与邻接矩阵节点数不符！"
print("\n=== 预测样本信息 ===")
print(f"X_pred shape: {X_pred.shape}")  # [3, 12, 节点数, 2]
print("展示第一个样本的前2小时数据（每小时每个站点的 inflow 和 outflow）:")

# 只打印第一个样本的前两小时的数据（避免输出过长）
for t in range(2):
    print(f"\n第 {t+1} 小时:")
    for node_idx, station_id in enumerate(station_ids):
        inflow = X_pred[0, t, node_idx, 0]
        outflow = X_pred[0, t, node_idx, 1]
        print(f"  站点 {station_id}: inflow={inflow:.1f}, outflow={outflow:.1f}")

# 转换为模型输入格式 [B, C, N, T]
X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).permute(0, 3, 2, 1)

# -----------------------------
# 4. 执行预测
with torch.no_grad():
    output = model(X_pred_tensor, adj)  # [B, C, N, T], T=1

# 转回 numpy [B, T, N, C]
output_np = output.permute(0, 3, 2, 1).numpy()
output_np = output_np.squeeze(axis=1)  # [B, N, C]

# -----------------------------
# 5. 站点ID列表，长度必须和 num_nodes 一致


# -----------------------------
# 6. 预测时间信息准备
# 这里举例基准日期和小时（请替换为你实际预测的日期和小时）
base_date = datetime.strptime('2025-07-09', '%Y-%m-%d')
base_hour = 10

# -----------------------------
# 7. 准备批量插入数据
insert_values = []
# 删除预测时间范围内旧记录（避免重复写入）
predict_hours = [(base_date + timedelta(hours=base_hour + i)) for i in range(output_np.shape[0])]
predict_dates_hours = [(dt.date(), dt.hour) for dt in predict_hours]

delete_sql = """
DELETE FROM station_hourly_status
WHERE (date, hour) IN (%s)
""" % ','.join(["(%s, %s)"] * len(predict_dates_hours))

delete_params = [item for pair in predict_dates_hours for item in pair]
cursor.execute(delete_sql, delete_params)
print(f"已删除原有的 {len(predict_dates_hours)} 个时间点的旧记录")

now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

for sample_idx in range(output_np.shape[0]):
    predict_date = (base_date + timedelta(days=(base_hour + sample_idx) // 24)).date()
    predict_hour = (base_hour + sample_idx) % 24
    for node_idx, station_id in enumerate(station_ids):
        inflow = int(round(output_np[sample_idx, node_idx, 0]))
        outflow = int(round(output_np[sample_idx, node_idx, 1]))
        stock = 0  # 目前无预测库存，暂填0
        insert_values.append((
            station_id,
            predict_date,
            predict_hour,
            inflow,
            outflow,
            stock,
            now_str
        ))

# -----------------------------
# 8. 批量插入数据库
sql = """
INSERT INTO station_hourly_status
(station_id, date, hour, inflow, outflow, stock, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
inflow=VALUES(inflow), outflow=VALUES(outflow), stock=VALUES(stock), updated_at=VALUES(updated_at)
"""

cursor.executemany(sql, insert_values)
conn.commit()

cursor.close()
conn.close()

print(f"已预测并写入 {len(insert_values)} 条记录，预测样本数：{output_np.shape[0]}, 站点数：{num_nodes}")
