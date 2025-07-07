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