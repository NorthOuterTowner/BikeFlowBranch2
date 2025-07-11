import sys
sys.stdout.reconfigure(encoding='utf-8')
import pymysql
import pandas as pd
import re

# 连接数据库
conn = pymysql.connect(host='localhost', user='zq', password='123456', database='traffic', charset='utf8mb4')
cursor = conn.cursor()

# 创建 station_info 表（如果尚未存在）
cursor.execute("""
CREATE TABLE IF NOT EXISTS station_info (
    station_id VARCHAR(20) PRIMARY KEY,
    lat FLOAT,
    lng FLOAT,
    station_name VARCHAR(100)
)
""")

# 站点ID合法性正则表达式，只允许两个大写字母开头，后跟数字，比如 HB315
pattern = re.compile(r'^[A-Z]{2}\d+$')

def is_valid_station_id(s):
    if pd.isna(s):
        return False
    return bool(pattern.match(str(s)))

# 从 bike_trip 表提取起点站点信息并过滤合法站点
df = pd.read_sql("""
    SELECT DISTINCT start_station_id AS station_id, start_lat AS lat, start_lng AS lng,start_station_name as station_name
    FROM bike_trip
    WHERE start_station_id IS NOT NULL AND start_lat IS NOT NULL AND start_lng IS NOT NULL AND start_station_name IS NOT NULL
""", conn)
df = df[df['station_id'].apply(is_valid_station_id)]

# 也加入终点站点信息，过滤合法站点
df_end = pd.read_sql("""
    SELECT DISTINCT end_station_id AS station_id, end_lat AS lat, end_lng AS lng,end_station_name AS station_name
    FROM bike_trip
    WHERE end_station_id IS NOT NULL AND end_lat IS NOT NULL AND end_lng IS NOT NULL AND end_station_name IS NOT NULL
""", conn)
df_end = df_end[df_end['station_id'].apply(is_valid_station_id)]

# 合并并去重
df_all = pd.concat([df, df_end]).drop_duplicates(subset='station_id').reset_index(drop=True)

# 写入 station_info 表
cursor.execute("DELETE FROM station_info")  # 清空旧数据（可选）
for row in df_all.itertuples(index=False):
    cursor.execute("""
        INSERT INTO station_info (station_id, lat, lng,station_name)
        VALUES (%s, %s, %s,%s)
        ON DUPLICATE KEY UPDATE lat=VALUES(lat), lng=VALUES(lng)
    """, (row.station_id, row.lat, row.lng,row.station_name))

conn.commit()
cursor.close()
conn.close()

print("已成功提取并写入 station_info 表，共", len(df_all), "个合法站点")
