import sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = pymysql.connect(
    host='localhost',
    user='zq',
    password='123456',
    database='traffic',
    charset='utf8mb4'
)

sql = """
SELECT station_id, SUM(inflow) AS total_inflow, SUM(outflow) AS total_outflow
FROM station_hourly_flow
GROUP BY station_id
ORDER BY total_inflow + total_outflow ASC
LIMIT 20
"""

df = pd.read_sql(sql, conn)
conn.close()

plt.figure(figsize=(14,7))
bar_width = 0.4
x = range(len(df))

plt.bar(x, df['total_inflow'], width=bar_width, label='总流入', color='skyblue')
plt.bar([i + bar_width for i in x], df['total_outflow'], width=bar_width, label='总流出', color='salmon')

plt.xticks([i + bar_width / 2 for i in x], df['station_id'], rotation=45)
plt.xlabel('站点ID')
plt.ylabel('数量')
plt.title('流量前20站点的总流入与总流出')
plt.legend()
plt.tight_layout()
plt.show()
