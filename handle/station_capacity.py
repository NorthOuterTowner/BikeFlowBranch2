import pymysql

# 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='zq',
    password='123456',
    database='traffic',
    charset='utf8mb4'
)
cursor = conn.cursor()

# 确保 station_info 表中有 capacity 字段（如果没有就添加）
cursor.execute("SHOW COLUMNS FROM station_info LIKE 'capacity'")
result = cursor.fetchone()
if not result:
    cursor.execute("ALTER TABLE station_info ADD COLUMN capacity INT")

# 站点容量字典
capacities = {
    "HB101": 16, "HB102": 16, "HB103": 14, "HB105": 18, "HB106": 15, "HB201": 14,
    "HB202": 15, "HB203": 13, "HB301": 13, "HB302": 13, "HB303": 12, "HB304": 12,
    "HB305": 14, "HB401": 12, "HB402": 12, "HB404": 11, "HB407": 11, "HB408": 11,
    "HB409": 12, "HB501": 13, "HB502": 13, "HB503": 12, "HB505": 11, "HB506": 13,
    "HB508": 12, "HB601": 15, "HB602": 16, "HB603": 14, "HB608": 12, "HB609": 15,
    "HB610": 12, "HB611": 12, "HB612": 13, "JC002": 18, "JC003": 17, "JC006": 16,
    "JC008": 17, "JC009": 15, "JC013": 14, "JC014": 13, "JC018": 13, "JC019": 12,
    "JC020": 12, "JC022": 12, "JC023": 11, "JC024": 11, "JC027": 11, "JC032": 13,
    "JC034": 14, "JC035": 13, "JC038": 12, "JC051": 12, "JC052": 13, "JC053": 12,
    "JC055": 12, "JC057": 11, "JC059": 11, "JC063": 11, "JC065": 13, "JC066": 17,
    "JC072": 11, "JC074": 11, "JC075": 11, "JC076": 11, "JC077": 11, "JC078": 11,
    "JC080": 11, "JC081": 11, "JC082": 11, "JC084": 11, "JC093": 11, "JC094": 11,
    "JC095": 11, "JC097": 12, "JC098": 13, "JC099": 14, "JC102": 13, "JC103": 16,
    "JC104": 17, "JC105": 15, "JC107": 12, "JC108": 12, "JC109": 12, "JC110": 15,
    "JC115": 17, "JC116": 18
}

# 更新每个站点的 capacity
for station_id, cap in capacities.items():
    cursor.execute("""
        UPDATE station_info
        SET capacity = %s
        WHERE station_id = %s
    """, (cap, station_id))

conn.commit()
cursor.close()
conn.close()

print("已成功写入所有站点的 capacity")
