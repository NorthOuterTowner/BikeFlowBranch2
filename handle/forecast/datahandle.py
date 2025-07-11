'''
初始数据整合
'''
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pymysql
import pandas as pd
import re
from tqdm import tqdm

# 1. 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='zq',
    password='123456',
    database='traffic',
    charset='utf8mb4'
)
cursor = conn.cursor()

print("正在清空旧数据...")
cursor.execute("DELETE FROM station_hourly_flow")
conn.commit()
print("旧数据已清空")

# 1. 读取bike_trip数据
print("正在从数据库读取骑行记录...")
sql = "SELECT ride_id, started_at, ended_at, start_station_id, end_station_id FROM bike_trip"
df = pd.read_sql(sql, conn)

# 2.清洗数据
pattern = re.compile(r'^[A-Z]{2}\d+$')

# 定义判断函数，判断站点ID是否合规
def is_valid_station_id(s):
    if pd.isna(s):
        return False
    return bool(pattern.match(str(s)))

# 过滤起点站点ID不合规的行
df = df[df['start_station_id'].apply(is_valid_station_id)]

# 过滤终点站点ID不合规的行
df = df[df['end_station_id'].apply(is_valid_station_id)]

# 3. 时间转换及下整小时
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
df = df.dropna(subset=['started_at', 'ended_at'])

df['start_hour'] = df['started_at'].dt.floor('h')
df['end_hour'] = df['ended_at'].dt.floor('h')

# 4. 检查缺失和示例值
print("缺失的起点站点数量:", df['start_station_id'].isnull().sum())
print("缺失的终点站点数量:", df['end_station_id'].isnull().sum())
print("起点站点示例值：", df['start_station_id'].dropna().unique()[:5])
print("终点站点示例值：", df['end_station_id'].dropna().unique()[:5])

# 5. 分别统计流出和流入
outflow = df.groupby(['start_station_id', 'start_hour']).size().reset_index(name='outflow')
inflow = df.groupby(['end_station_id', 'end_hour']).size().reset_index(name='inflow')

outflow.rename(columns={'start_station_id': 'station_id', 'start_hour': 'timestamp'}, inplace=True)
inflow.rename(columns={'end_station_id': 'station_id', 'end_hour': 'timestamp'}, inplace=True)

print(f"outflow shape: {outflow.shape}, inflow shape: {inflow.shape}")

print("outflow 示例（按数量排序，前10条）：")
print(outflow.sort_values('outflow', ascending=False).head(10))

print("inflow 示例（按数量排序，前10条）：")
print(inflow.sort_values('inflow', ascending=False).head(10))

outflow_zero = outflow[outflow['outflow'] == 0]
print(f"outflow中数量为0的条目数: {len(outflow_zero)}")

if not outflow.empty:
    sample_station = outflow.iloc[0]['station_id']
    print(f"示例站点 {sample_station} 的所有流出记录：")
    print(outflow[outflow['station_id'] == sample_station])

# 6. 合并统计结果
merged = pd.merge(outflow, inflow, on=['station_id', 'timestamp'], how='outer').fillna(0)
merged['inflow'] = merged['inflow'].astype(int)
merged['outflow'] = merged['outflow'].astype(int)

print("准备写入 station_hourly_flow 表，共", len(merged), "条记录")

# 7. 写入数据库
insert_sql = """
INSERT INTO station_hourly_flow (station_id, timestamp, inflow, outflow)
VALUES (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE inflow=VALUES(inflow), outflow=VALUES(outflow)
"""

for row in tqdm(merged.itertuples(index=False)):
    cursor.execute(insert_sql, (row.station_id, row.timestamp, row.inflow, row.outflow))

conn.commit()
cursor.close()
conn.close()

print("数据处理完成并写入成功！")
