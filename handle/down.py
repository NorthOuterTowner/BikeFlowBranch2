import pandas as pd
import pymysql
import math

# 数据库连接配置
conn = pymysql.connect(
    host='localhost',
    user='zq',          # 修改为你的用户名
    password='123456',  # 修改为你的密码
    database='traffic', # 修改为你的数据库名
    charset='utf8mb4'
)
cursor = conn.cursor()

# 清空表，防止重复导入
cursor.execute("TRUNCATE TABLE bike_trip;")
conn.commit()
print("已清空 bike_trip 表。")

# 定义要导入的文件列表（2025年1-6月）
months = [f"{i:02d}" for i in range(1, 7)]
base_path = "D:/trafiic/"
file_template = "JC-2025{}-citibike-tripdata.csv"

total_rows = 0

for m in months:
    file_path = base_path + file_template.format(m)
    print(f"正在读取文件：{file_path}")

    df = pd.read_csv(file_path, encoding='utf-8', quotechar='"')

    # 转换时间和经纬度字段格式
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    df['start_lat'] = pd.to_numeric(df['start_lat'], errors='coerce')
    df['start_lng'] = pd.to_numeric(df['start_lng'], errors='coerce')
    df['end_lat'] = pd.to_numeric(df['end_lat'], errors='coerce')
    df['end_lng'] = pd.to_numeric(df['end_lng'], errors='coerce')

    # 删除 ride_id、started_at、ended_at 缺失的行
    df = df.dropna(subset=['ride_id', 'started_at', 'ended_at'])

    print(f"准备插入 {len(df)} 条数据...")

    sql = """
    INSERT INTO bike_trip (
        ride_id, rideable_type, started_at, ended_at,
        start_station_name, start_station_id,
        end_station_name, end_station_id,
        start_lat, start_lng, end_lat, end_lng, member_casual
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    inserted_count = 0
    for row in df.itertuples(index=False, name=None):
        row_clean = [None if (isinstance(x, float) and math.isnan(x)) else x for x in row]
        try:
            cursor.execute(sql, tuple(row_clean))
            inserted_count += 1
        except Exception as e:
            print("插入失败行：", row_clean)
            print(e)

    conn.commit()
    print(f"文件 {file_path} 导入完成，成功插入 {inserted_count} 条记录。")
    total_rows += inserted_count

cursor.close()
conn.close()

print(f"所有文件导入完毕，共导入 {total_rows} 条数据。")
