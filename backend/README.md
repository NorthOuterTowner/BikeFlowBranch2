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
1.1用户注册（POST）
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
1.2用户登录（POST）
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
1.3重置账号（POST）
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
1.4 重置密码（POST）
```bash
/reset/pwd
```
输入格式
```bash
{
  "email": "admin", 
  "newPassword": "admin2" //新密码
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
执行调度后status设置为1
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
    "dispatchId": 36 //调度编号
}
```
返回格式
```bash
{
  "code": 200, //状态码
  "msg": "调度成功"
}
```
（2）拒绝调度（POST）
拒绝调度将会将调度方案删除
```bash
/dispatch/reject
```
请求格式
```bash
{
  "id":36
}
```
返回格式
```bash
{
    "code": 200,
    "msg": "已拒绝该调度"
}
```
（3）取消调度
status设置为0
```bash
/dispatch/cancelChange
```
请求格式
```bash
{
    "startStation": "X2019", //起始站点编号
    "endStation": "X2024", //目标站点编号
    "number": 1, //调度数量
    "dispatchDate": "2025-01-16",//调度日期
    "dispatchHour": 17 //调度小时
    "dispatchId": 36 //调度编号
}
```
返回格式
```bash
{
  "code": 200, //状态码
  "msg": "已取消调度"
}
```

6. 调度信息操作
   （1）返回某一时间点所有调度信息（get）
```bash
/dispatch
```
请求格式
```bash
query_time	String	查询的时间点，ISO 8601格式。	2025-06-13T08:45:00Z
```
返回格式
```bash
{
    "lookup_date": "2025-06-13",
    "lookup_hour": 6,
    "schedules": 
    [
        {
            "schedule_id": 35,//调度编号
            "bikes_to_move": 2,//移动车
            "status": "待执行",//状态信息
            "start_station": {
                "id": "HB101",
                "name": "Hoboken Terminal - Hudson St & Hudson Pl",
                "lat": 40.7359,
                "lng": -74.0303
            },
            "end_station": {
                "id": "HB304",
                "name": "Marshall St & 2 St",
                "lat": 40.7408,
                "lng": -74.0425
            },
            "updated_at": "2025-07-11T10:32:31.000Z"
        }
        // ... 如果同一调度周期有其他任务，也会在此列出
    ]
}
```
（2）返回某一时间点与某一地点相关调度信息（get）  
role选择end查询所有调出的信息 start为调入信息
```bash
/dispatch/by-station
```
请求格式
```bash
station_id	String	要查询的站点唯一ID。	HB101
query_time	String	查询的时间点，ISO 8601格式。	2025-06-13T08:45:00Z
role	String 可选	筛选站点在调度中的角色。<br> - 'start': 站点作为调出点（起点）。  'end': 站点作为调入点（终点）。如果省略此参数，将返回所有相关任务。	start	2025-06-13T08:45:00Z
```
返回格式
```bash
{
    "lookup_date": "2025-06-13",
    "lookup_hour": 6,
    "station_id": "HB101",
    "role_filter": "all",
    "schedules": 
    [
        {
            "schedule_id": 42,
            "bikes_to_move": 5,
            "status": "pending",
            "start_station": {
                "id": "HB101",
                "name": "Hoboken Terminal - Hudson St & Hudson Pl",
                "lat": 40.7359,
                "lng": -74.0303
            },
            "end_station": 
            {
                "id": "JC053",
                "name": "Lincoln Park",
                "lat": 40.7246,
                "lng": -74.0784
            },
            "updated_at": "2025-07-12T06:00:00.000Z"
        }
        ...
    ]
}
```
（3）返回某一时间点所有有调度出信息的站点（get）
```bash
/search/stationAssign
```
请求格式
```bash
date hour
```
返回格式
```bash
{
    "code": 200,
    "station_result": 
    [
        {
            "station_name": "Hoboken Terminal - Hudson St & Hudson Pl"
        },...
    ]
}
```
（4）将接受的新调度方案加入（post）
```bash
/dispatch/add
```
请求格式
```bash
{
  "schedule_time": "2025-06-13T09:00:00Z",
  "start_station_id": "HB101",
  "end_station_id": "HB304",
  "bikes_to_move": 5
}
```
返回格式
```bash
{
    "message": "调度任务已成功添加。",
    "schedule": {
        "id": 69,
        "date": "2025-06-13",
        "hour": 9,
        "start_id": "HB101",
        "end_id": "HB304",
        "bikes": 5,
        "status": 0,
        "updated_at": "2025-07-15T06:49:09.547Z"
    }
}
```


7、返回导航信息
   （1）根据站点信息返回导航信息（post）
```bash
/guide/route
```
请求格式
```bash
{
  "startCoord": [116.3974, 39.9093],
  "endCoord": [116.4854, 39.9903]
}
startCoord 前端中格式化的地点数据
endCoord 
```
返回格式
```bash
{
   res.json(orsResponse.data);//返回的是这个东西（要以json格式），可以解析一下
}
```

8.修改调度方案
（1）修改调度方案（POST）
```bash
/dispatch/edit
请求格式
```
    "id":[调度方案编号],
    "bikes":[调度数量]

返回格式
```
{
  "code":200,
  "msg":"修改成功
}
```
9.使用deepseek
（1）deepseek根据现有预测和调度方案和用户要求优化并返回增加的调度方案（post）
```bash
/suggestions/dispatch
请求格式
```bash
{
  "target_time": "2025-06-13T09:00:00Z",
  "user_guidance": "优先保证Hoboken总站的车辆充足，可以从附近的站点调车过来。"
}
```
返回格式
```bash
{
    "schedule_time": "2025-06-13T09:35:00Z",
    "optimized_plan": [
        {
            "from_station_id": "HB101",
            "to_station_id": "HB201",
            "bikes_to_move": 2,
            "reason": "HB101 has excess bikes (11) and HB201 is at risk of depletion (2)."
        },
        {
            "from_station_id": "HB101",
            "to_station_id": "HB203",
            "bikes_to_move": 2,
            "reason": "HB101 has excess bikes (11) and HB203 is at risk of depletion (4)."
        }
        ...
    ]//调度方案建议
}
```
（1）deepseek根据现有预测和调度方案和用户要求优化并返回增加的调度方案（post）
```bash
/suggestions/dispatch
请求格式
```bash
{
  "target_time": "2025-06-13T09:00:00Z",
  "user_guidance": "优先保证Hoboken总站的车辆充足，可以从附近的站点调车过来。"
}
```
返回格式
```bash
{
    "schedule_time": "2025-06-13T09:35:00Z",
    "optimized_plan": [
        {
            "from_station_id": "HB101",
            "to_station_id": "HB201",
            "bikes_to_move": 2,
            "reason": "HB101 has excess bikes (11) and HB201 is at risk of depletion (2)."
        },
        {
            "from_station_id": "HB101",
            "to_station_id": "HB203",
            "bikes_to_move": 2,
            "reason": "HB101 has excess bikes (11) and HB203 is at risk of depletion (4)."
        }
        ...
    ]//调度方案建议
}
```
（2）与deepseek直接对话（post）
```bash
/suggestions
请求格式
```bash
{
  "message": "纽约有几种共享单车"
}
```
返回格式
```bash
{
    "original_prompt": "纽约有几种共享单车",
    "suggestion": "截至2023年，纽约市主要有以下共享单车服务：\n\n1. **Citi Bike**（主导系统）\n- 运营方：Lyft（2018年收购Motivate后获得）\n- 规模：全美最大共享单车系统，纽约市覆盖曼哈顿/布鲁克林/皇后区/泽西市等\n- 车辆：约25,000辆（含传统自行车和电动辅助自行车）\n- 特点：标志性「花旗蓝」涂装，30%车辆为电动助力车（2023年数据）\n\n2. **Lyft Bike**（特殊区域）\n- 在罗斯福岛等特定区域运营\n- 实际与Citi Bike属同一系统，使用相同APP\n\n3. **JUMP by Uber**（已退出）\n- 曾于2018-2020年运营（红色电动车）\n- 因收购纠纷和疫情于2020年停止服务\n\n4. **新型微移动服务试验**\n- 2022年起试点共享电动滑板车（Citi Bike未参与）\n- 目前仅限布朗克斯区等外围区域\n\n重要提示：\n- Citi Bike占据95%以上市场份额\n- 纽约市通过特许经营制度严格控制运营商数量\n- 非合作企业的共享单车会被政府直接没收（如2023年中国公司「小靓单车」违规投放案例）\n\n建议用户优先使用Citi Bike，其APP实时显示：\n- 各站点车辆/空桩数量\n- 电动自行车可用情况\n- 骑行优惠信息（含NYCHA居民专项计划）"//给出的建议
}
```

10.统计数据
（1）指定时间段总流量（POST）
```bash
/statistics/flow/time
```
输入格式
```bash
{
  "startDate": "2025-01-01", //开始日期
  "startHour": "10", //开始小时
  "endDate": "2025-02-01" , //结束日期
  "endHour": "12" //结束小时
}
```
输出格式
```bash
{
    "code": 200,
    "data": 
    {
        "inflow": 50304,
        "outflow": 50291,
        "total": 100595
    }
}
```
（2）获取top10站点及其流量（POST）
```bash
/statistics/top10
```
请求格式
```bash
{
  "startDate": "2025-01-01", //开始日期
  "startHour": "10", //开始小时
  "endDate": "2025-02-01", //结束日期
  "endHour": "12" //结束小时
}
```
返回格式
```bash
{
    "code": 200,
    "data": [
        {
            "station_id": "HB102",
            "total_flow": "6136"
        },
        {
            "station_id": "JC115",
            "total_flow": "5658"
        },
        {
            "station_id": "JC109",
            "total_flow": "2654"
        },
        {
            "station_id": "HB105",
            "total_flow": "2640"
        },
        {
            "station_id": "JC066",
            "total_flow": "2404"
        },
        {
            "station_id": "JC009",
            "total_flow": "2363"
        },
        {
            "station_id": "HB101",
            "total_flow": "2335"
        },
        {
            "station_id": "HB609",
            "total_flow": "2240"
        },
        {
            "station_id": "HB603",
            "total_flow": "2233"
        },
        {
            "station_id": "JC116",
            "total_flow": "2099"
        }
    ]
}
```
（3）获取某一时间点总流量（GET）
```bash
/statistics/flow/day
```
请求格式
```bash
query_date	String	要查询的日期，格式为 YYYY-MM-DD。	"2025-01-21"
```
返回格式
```bash
{
  "query_time": "2025-01-21T08:00:00Z",
  "total_inflow": 150,//进入车流量
  "total_outflow": 145,//出车流量
  "total_flow": 295//总流量
  "query_date": "2025-01-21",
  "hourly_flows": 
  [
    {
      "hour": 0,//对应的时间
      "total_inflow": 15,
      "total_outflow": 18,
      "total_flow": 33
    },
    {
      "hour": 1,
      "total_inflow": 8,
      "total_outflow": 12,
      "total_flow": 20
    },
    // ...
  ]
}
```

（4）获取某一时间点总流量（GET）
```bash
/statistics/flow/days
```
请求格式
```bash
query_date	String	要查询的日期，格式为 YYYY-MM-DD。	"2025-01-21"
```
返回格式
```
    "target_date": "2025-01-30",
    "daily_summary": [
        {
            "date": "2025-01-15",//数据的时间
            "total_inflow": 1885,
            "total_outflow": 1886,
            "total_flow": 3771
        },
        {
            "date": "2025-01-16",
            "total_inflow": 1983,
            "total_outflow": 1982,
            "total_flow": 3965
        },
        {
            "date": "2025-01-17",
            "total_inflow": 2085,
            "total_outflow": 2089,
            "total_flow": 4174
        },
        ...
```

11.调度方案请求(get)
```bash
/schedule
```
请求格式
```bash
{
    "date": "2025-06-13",//日期
    "hour": 9//整点时间，生成是该点的调度方案
}
```
返回格式
```bash
{
    "success": true,
    "message": "调度成功",
    "output": "[调度完成] 调度动作数：6"
}
{
    "success": false,
    "message": "调度执行失败",
    "error": "[警告] 时间段内无数据：..."
}
```