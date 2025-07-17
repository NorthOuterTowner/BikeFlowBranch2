import requests
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta

# --- 配置 ---
# 从环境变量中安全地获取数据库凭据
DB_CONFIG = {
    'host': 'localhost',  # 数据库主机地址
    'user': 'root',
    'password': 'root',
    'database': 'bike'
}
API_URL = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"


def fetch_station_status():
    """从API获取站点状态数据"""
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()  # 如果请求失败 (如 404, 500), 会抛出异常
        print("成功从API获取数据。")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"错误：无法从API获取数据: {e}")
        return None


def get_station_id_map(cursor):
    """
    从station_info表获取 station_web_id -> station_id 的映射.
    这比在循环中为每个站点查询数据库要高效得多。
    """
    try:
        cursor.execute("SELECT station_web_id, station_id FROM station_info WHERE station_web_id IS NOT NULL")
        id_map = {row['station_web_id']: row['station_id'] for row in cursor.fetchall()}
        print(f"成功加载 {len(id_map)} 个站点ID映射。")
        return id_map
    except Error as e:
        print(f"错误: 无法获取站点ID映射: {e}")
        return {}


def main():
    """主执行函数"""
    print(f"脚本开始于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 获取API数据
    api_data = fetch_station_status()
    if not api_data or 'data' not in api_data or 'stations' not in api_data['data']:
        print("API数据格式不正确或为空，脚本终止。")
        return

    stations_from_api = api_data['data']['stations']

    conn = None
    try:
        # 2. 连接数据库
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # 使用字典游标，方便按列名访问
        print("成功连接到MySQL数据库。")

        # 3. 获取ID映射
        web_id_to_internal_id_map = get_station_id_map(cursor)
        if not web_id_to_internal_id_map:
            print("无法获取ID映射，脚本终止。")
            return

        # 4. 准备数据
        # now = datetime.now()
        now = datetime(2025, 7, 17, 12, 0, 0)
        # 时间向下取整到整点
        target_timestamp = now.replace(minute=0, second=0, microsecond=0)
        target_date = target_timestamp.date()
        target_hour = target_timestamp.hour

        # 上一小时的时间戳
        previous_timestamp = target_timestamp - timedelta(hours=1)
        previous_date = previous_timestamp.date()
        previous_hour = previous_timestamp.hour

        real_data_to_update = []
        flow_data_to_update = []

        for station in stations_from_api:
            web_id = station.get('station_id')

            # 使用映射找到内部ID
            internal_id = web_id_to_internal_id_map.get(web_id)
            if not internal_id:
                # print(f"警告: 在station_info中找不到 station_web_id '{web_id}' 的记录，已跳过。")
                continue

            current_stock = station.get('num_bikes_available', 0)

            # --- 准备 station_real_data 的数据 ---
            real_data_to_update.append((
                internal_id,
                target_date,
                target_hour,
                current_stock,
                now  # update_time
            ))

            # --- 准备 station_hourly_flow 的数据 ---
            # 查询上一小时的库存
            cursor.execute(
                "SELECT stock FROM station_real_data WHERE station_id = %s AND date = %s AND hour = %s",
                (internal_id, previous_date, previous_hour)
            )
            previous_data = cursor.fetchone()

            if previous_data:
                previous_stock = previous_data['stock']
                flow = current_stock - previous_stock

                inflow = max(0, flow)
                outflow = max(0, -flow)

                flow_data_to_update.append((
                    internal_id,
                    target_timestamp,  # 流量表的时间戳是完整的datetime
                    inflow,
                    outflow
                ))

        # 5. 批量执行数据库操作
        if real_data_to_update:
            sql_real_data = """
                INSERT INTO station_real_data (station_id, date, hour, stock, update_time)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE stock = VALUES(stock), update_time = VALUES(update_time)
            """
            cursor.executemany(sql_real_data, real_data_to_update)
            print(f"成功插入/更新 {cursor.rowcount} 条记录到 station_real_data。")

        if flow_data_to_update:
            sql_hourly_flow = """
                INSERT INTO station_hourly_flow (station_id, timestamp, inflow, outflow)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE inflow = VALUES(inflow), outflow = VALUES(outflow)
            """
            cursor.executemany(sql_hourly_flow, flow_data_to_update)
            print(f"成功插入/更新 {cursor.rowcount} 条记录到 station_hourly_flow。")

        # 提交事务
        conn.commit()
        print("数据库事务已提交。")

    except Error as e:
        print(f"数据库操作失败: {e}")
        if conn:
            conn.rollback()
            print("数据库事务已回滚。")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("数据库连接已关闭。")

    print(f"脚本结束于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()