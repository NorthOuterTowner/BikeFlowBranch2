# Backend使用说明

## 简要说明
backend部分 是一个基于 Node.js 的后端应用程序，旨在根据需求为前端提供相关 API。该项目使用 Express 框架构建，并集成了 MySQL 数据库。

## 安装依赖
1. 安装 Node.js 和 npm
   - 请参考 [Node.js 官网](https://nodejs.org/) 下载并安装最新版本的 Node.js。
2. 安装依赖（这一步最好在系统cmd中以管理员身份运行）
    切换到项目目录下的 `BikeFlow/backend` 文件夹，并使用 npm 安装所需的依赖包。
   ```bash
    cd BikeFlow/backend
    npm install
    ```
3. 启动app.js（在VScode的集成终端运行即可）
    ```bash
     node app.js
     ```
    若终端显示🚀 Server is running at http://localhost:${PORT}，则说明后端启动成功。
    由于当前搭建了与数据库进行连接的框架，因此我假定连接的数据库账号为root，密码为root，数据库名为schedule，具体配置在db/dbUtils.js。若出现“数据库连接失败的问题”，可以将其中的内容改为自己本地存在的某个数据库即可。
4. 结束进程
   - 在终端中按 `Ctrl + C` 结束当前进程。

## 测试说明
1.如果在未登录状态测试非登录接口，请在header中加入如下数据：
```bash
--header 'account: admin' \
--header 'token: 301b70c6-ec86-465e-a1da-d16f36e32c08' \
--header 'Content-Type: application/json'
```

## 接口说明
0.登录状态验证  
非登录注册接口需要传入账户与token信息（在登录接口提供）
```bash
--header 'account: admin' \
--header 'token: 301b70c6-ec86-465e-a1da-d16f36e32c08' \
--header 'Content-Type: application/json'
```
1.登陆注册
（1）用户注册（POST）
```bash
/admin/register
```
请求格式
```bash
{
  "account": "admin", //用户名
  "password": "admin", //密码
  "email": "23301xxx@bjtu.edu.cn" //邮箱
}
```
返回格式
```bash
{
  "code": 200, //状态码
  "msg": "注册成功"
}
```
（2）用户登录（POST）
```bash
/admin/login
```
请求格式
```bash
{
  "account": "admin", //用户名
  "password": "admin" //密码
}
```

返回格式
```bash
{
  "code": 200, //状态码
  "msg": "登录成功",
  "data": 
    {
      "account": "admin",
      "password": "",
      "token": "b8aeb113-f7b0-47ab-a373-b8719888bcf0",
      "email": "23301xxx@bjtu.edu.cn"
    }
}

```
（3）重置账号（POST）
```bash
/reset/account
```
输入格式
```bash
{
  "oldName": "admin", //用户名
  "newName": "admin2", //新用户名
}
```
返回格式
```bash
{
  "code": 200, //状态码
  "msg": "账号重置成功"
}
```
2.站点位置  
（1）获取所有的站点位置（GET）
```bash
/stations/locations
```
返回格式
```bash
{
    "station_id": "JC019",//站点编号
    "station_name": "Hilltop",//站点名
    "latitude": 40.7312,//纬度
    "longitude": -74.0576//精度
  },
  {
    "station_id": "JC024",
    "station_name": "Pershing Field",
    "latitude": 40.7427,
    "longitude": -74.0518
  },
  // ... more stations
```
3.站点实际单车数量
（1）获取指定站点在指定时间的单车数量（GET）
```bash
/stations/bikeNum
```
query参数
```bash
{
  "station_id": "JC019" //站点编号
  "date": "2023-10-01" //日期
  "hour": "10" //小时
}
```
返回格式
```bash
{
  "code":200, //状态码
  "bikeNum": 12 //库存量
}
```
（2）获取指定时间所有的站点各自的单车数量（GET）
```bash
/stations/bikeNum/timeAll
```
query参数
```bash
{
  "date": "2023-10-01", //日期
  "hour": "10" //小时
}
```
返回格式
```bash
{
  code:200, //状态码
  "rows": [
      {
          "station_id": "X2019",
          "stock": 12
      },
      {
          "station_id": "X2024",
          "stock": 18
      }
  ]
}
```
（3）获取指定站点的所有单车数量（GET）
```bash
/stations/bikeNum/stationAll
```
query参数
```bash
{
  "station_id": "JC019" //站点编号
}
```
返回格式
```bash
{
  "code":200, //状态码
  "rows": [
      {
          "date": "2023-10-01",
          "hour": 10,
          "stock": 12
      },
      {
          "date": "2023-10-01",
          "hour": 11,
          "stock": 15
      }
  ]
}
```

4.节点预测数量结果
（1）获取指定站点在指定时间的单车数量（GET）
```bash
http://localhost:3000/predict/station?station_id=JC024&predict_time=2025-01-21T07:02:00Z
```
query参数
station_id	String	要查询的站点唯一ID。	JC019  
predict_time	String	查询的时间点，ISO 8601格式 2025-01-21T08:45:00Z  
返回格式
```bash
{
    "station_id": "JC019",
    "lookup_date": "2025-01-21",
    "lookup_hour": 8,
    "status": 
    {
        "inflow": 2,//入车流
        "outflow": 15,//出车流
        "stock": 5 //这个是预测的数量
    }
}
```

（2）获取指定时间所有的站点各自的单车数量（GET）
```bash
http://localhost:3000/predict/stations/all?predict_time=2025-01-21T07:02:00Z
```
query参数
predict_time	String	查询的时间点，ISO 8601格式。	2025-01-22T17:10:00Z
返回格式
```bash
{
    "lookup_date": "2025-01-21",
    "lookup_hour": 7,
    "stations_status": [
        {
            "station_id": "JC019",
            "inflow": 1,//入车流
            "outflow": 12,//出车流
            "stock": 18 //车辆数量
        },
        {
            "station_id": "JC024",
            "inflow": 10,
            "outflow": 2,
            "stock": 15
        },
        {
            "station_id": "HB601",
            "inflow": 2,
            "outflow": 3,
            "stock": 12
        }
    ]
}
```
5. 执行调度过程
（1）执行调度过程（POST）
```bash
/dispatch/change
```
请求格式
```bash
{
    "startStation": "X2019", //起始站点编号
    "endStation": "X2024", //目标站点编号
    "number": 1, //调度数量
    "dispatchDate": "2025-01-16",//调度日期
    "dispatchHour": 17 //调度小时
}
```
返回格式
```bash
{
  "code": 200, //状态码
  "msg": "调度成功"
}
```