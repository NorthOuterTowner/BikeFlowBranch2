import requests
import mysql.connector
import os
import sys

# --- 配置 ---
STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"

# 从环境变量中安全地获取数据库凭据
DB_CONFIG = {
    'host': 'localhost',  # 数据库主机地址
    'user': 'root',
    'password': 'root',
    'database': 'bike'
}


def validate_db_config():
    """检查是否所有必要的数据库环境变量都已设置。"""
    if not all([DB_CONFIG['user'], DB_CONFIG['password'], DB_CONFIG['database']]):
        print("错误：数据库环境变量 DB_USER, DB_PASSWORD, 和 DB_NAME 必须被设置。")
        print("请在运行脚本前设置它们。")
        sys.exit(1)


def main():
    """主函数，执行整个同步流程。"""
    validate_db_config()

    # 1. 获取 GBFS 站点信息数据
    try:
        print(f"正在从 URL 下载数据: {STATION_INFO_URL}")
        response = requests.get(STATION_INFO_URL, timeout=15)
        response.raise_for_status()  # 如果 HTTP 请求返回错误码，则抛出异常
        gbfs_stations = response.json()['data']['stations']
        print(f"成功获取到 {len(gbfs_stations)} 个站点的信息。")
    except requests.RequestException as e:
        print(f"网络错误：无法获取数据。 {e}")
        sys.exit(1)
    except (KeyError, TypeError) as e:
        print(f"JSON 格式错误：无法解析数据，检查 'data' 或 'stations' 键。 {e}")
        sys.exit(1)

    # 2. 连接数据库并执行更新
    conn = None
    try:
        print(f"正在连接到数据库 '{DB_CONFIG['database']}'...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("数据库连接成功。")

        updated_count = 0
        not_found_count = 0

        # 准备 SQL 更新语句
        # 使用参数化查询来防止 SQL 注入
        update_query = """
            UPDATE station_info 
            SET station_web_id = %s 
            WHERE station_name = %s
        """

        print("开始遍历并更新数据库记录...")
        for station in gbfs_stations:
            api_name = station.get('name')
            api_web_id = station.get('station_id')

            if not api_name or not api_web_id:
                print(f"警告：跳过一条不完整的记录: {station}")
                continue

            # 执行更新
            cursor.execute(update_query, (api_web_id, api_name))

            # cursor.rowcount 会返回受影响的行数 (0 或 1)
            if cursor.rowcount > 0:
                updated_count += 1
                # 为了简洁，可以注释掉下面这行，除非你需要详细的日志
                # print(f"  -> 已更新: '{api_name}' -> station_web_id = {api_web_id}")
            else:
                not_found_count += 1

        # 提交所有更改到数据库
        conn.commit()

        print("\n--- 更新完成 ---")
        print(f"成功更新了 {updated_count} 条记录。")
        if not_found_count > 0:
            print(f"有 {not_found_count} 个来自API的站点在数据库中未找到匹配的 station_name。")

    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        if conn:
            conn.rollback()  # 如果发生错误，回滚所有未提交的更改
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("数据库连接已关闭。")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()