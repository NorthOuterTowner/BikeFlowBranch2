# :bookmark_tabs:数据库说明
## :sparkler: 使用方式
法1：使用 mysql 命令将 SQL 文件导入到本地 MySQL 服务器，即可在本地使用该数据库。
```bash
mysql -u [username] -p [database_name] < [database_name.sql]
```
法2：直接启动mysql命令行，将 SQL 文件导入到本地 MySQL 服务器
```bash
use [database_name]
source /database/trafficData.sql
```
## 表主要属性介绍
### admin
### bike_trip
id:唯一标识自增id

start_station_id 开始站点id，结束类似

start_lat,start_lng 开始经纬度，结束类似
### station_hourly_flow 中间处理表
station_id：站点

timestamp：时间（datetime类型）

inflow\outflow：流入流出量

created_at：创建时间（待定）
### station_hourly_status 预测结果表（即某天某一小时某站点状态）

id：唯一标识

date：日期

hour:哪一小时

inflow、outflow：流入流出量

stock：该站点该时刻库存

updataed_at：更新时间（待定）
