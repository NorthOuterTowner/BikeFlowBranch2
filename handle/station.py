import sys
sys.stdout.reconfigure(encoding='utf-8')
import pymysql
import pandas as pd

# 连接数据库
conn = pymysql.connect(host='localhost', user='zq', password='123456', database='traffic', charset='utf8mb4')
cursor = conn.cursor()

# 创建 station_info 表（如果尚未存在）
cursor.execute("""
CREATE TABLE IF NOT EXISTS station_info (
    station_id VARCHAR(20) PRIMARY KEY,
    lat FLOAT,
    lng FLOAT
)
""")

# 从 bike_trip 表提取起点站点信息
df = pd.read_sql("""
    SELECT DISTINCT start_station_id AS station_id, start_lat AS lat, start_lng AS lng
    FROM bike_trip
    WHERE start_station_id IS NOT NULL AND start_lat IS NOT NULL AND start_lng IS NOT NULL
""", conn)

# 也加入终点站点信息
df_end = pd.read_sql("""
    SELECT DISTINCT end_station_id AS station_id, end_lat AS lat, end_lng AS lng
    FROM bike_trip
    WHERE end_station_id IS NOT NULL AND end_lat IS NOT NULL AND end_lng IS NOT NULL
""", conn)

# 合并并去重
df_all = pd.concat([df, df_end]).drop_duplicates(subset='station_id')

# 写入 station_info 表
cursor.execute("DELETE FROM station_info")  # 清空旧数据（可选）
for row in df_all.itertuples(index=False):
    cursor.execute("""
        INSERT INTO station_info (station_id, lat, lng)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE lat=VALUES(lat), lng=VALUES(lng)
    """, (row.station_id, row.lat, row.lng))

conn.commit()
cursor.close()
conn.close()

print("✅ 已成功提取并写入 station_info 表，共", len(df_all), "个站点")
