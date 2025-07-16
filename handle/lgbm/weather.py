import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime

# 定义纽约坐标
nyc = Point(40.7128, -74.0060)

# 定义时间范围
start = datetime(2025, 1, 1)
end = datetime(2025, 6, 30, 23)

# 获取每小时天气
data = Hourly(nyc, start, end)
df = data.fetch()

# 重命名索引和列
df.reset_index(inplace=True)
df.rename(columns={'time': 'timestamp'}, inplace=True)

# 只保留常用字段
df = df[['timestamp', 'temp', 'prcp', 'wspd']]  # 气温、降水、风速
df.to_csv('./handle/lgbm/nyc_weather_hourly.csv', index=False)
print("天气数据已保存：./handle/lgbm/nyc_weather_hourly.csv")
