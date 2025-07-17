const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const sequelize = require('../orm/sequelize');
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)

const authMiddleware = require("../utils/auth")

const ScheduleModel = require('../orm/models/stationSchedule');
const InfoModel = require('../orm/models/Station');
const StationSchedule = ScheduleModel(sequelize,DataTypes);
const StationInfo = InfoModel(sequelize,DataTypes);

const redis = require("redis");
const redisClient = require("../db/redis");

const dispatchQueue = require("../queue/dispatchQueue");

// 辅助函数
const toYYYYMMDD = (date) => date.toISOString().slice(0, 10);

/**
 * @api {post} /change 执行调度
 * @apiDescription 根据调度内容从起点站取车，计算调度完成时间，并在调度完成后将终点站单车余量增加。
 *
 * @apiHeader {String} Authorization 用户登录令牌，需通过authMiddleware验证。
 *
 * @apiParam {String} startStation 调度起点站ID。
 * @apiParam {String} endStation 调度终点站ID。
 * @apiParam {Number} number 调度车辆数量。
 * @apiParam {String} dispatchDate 调度日期，格式 YYYY-MM-DD。
 * @apiParam {Number} dispatchHour 调度小时，0-23。
 * @apiParam {String} dispatchId 调度记录ID。
 *
 * @apiSuccess {Number} code 状态码，200表示调度已开始。
 * @apiSuccess {String} msg 返回提示信息，"开始进行调度"。
 *
 * @apiError (400) {Number} code 400
 * @apiError (400) {String} error 错误信息，如"无效站点"。
 *
 * @apiError (422) {Number} code 422
 * @apiError (422) {String} error 调度方案不可行，如"调度数量超过本站点车余量"。
 *
 * @apiError (500) {Number} code 500
 * @apiError (500) {String} error 服务器错误提示，如"调度失败"。
 */
router.post('/change', authMiddleware, async (req, res) => {
  let { startStation,endStation,number,dispatchDate, dispatchHour,dispatchId } = req.body
  
  const queryStartsql = "select 1 from `station_real_data` where `station_id` = ? "
  const queryEndSql = "select 1 from `station_real_data` where `station_id` = ? "
  
  let {err:startErr,rows:startRows} = await db.async.all(queryStartsql,[startStation])
  let {err:endErr,rows:endRows} = await db.async.all(queryEndSql,[endStation])

  if(startErr == null && endErr == null && startRows.length > 0 && endRows.length > 0){
    let changeableStock = 0
    const getStockSql = "select `stock` from `station_real_data` where `station_id` = ? and `date` = ? and `hour` = ? "
    let {err:searchErr,rows:searchRows} = await db.async.all(getStockSql,[startStation,dispatchDate,dispatchHour]) 
    console.log(searchRows)
    changeableStock = searchRows[0].stock

    if(changeableStock < number){
        return res.status(422).send({//语义错误
            code:422,
            error:"该调度方案不可行，调度数量超过本站点车余量"
        })
    }else{
      
      let {lat:startLat,lng:startLng} = await getStationCoordinates(startStation)
      let {lat:endLat,lng:endLng} = await getStationCoordinates(endStation)

      const distance = calcLength(startLat,startLng,endLat,endLng)

      let time = (distance/1000/20)*60*60*1000/20
      console.log(time)
      //change status of dispatch
      const statusSql = "update `station_schedule` set `status` = 1 where `id` = ? ;"
      await db.async.run(statusSql,[dispatchId])

      let dispatchHourInt=parseInt(dispatchHour);
      while(dispatchHourInt <= 23){
        await afterTimeSchedule(number,startStation,dispatchDate,dispatchHourInt);
        dispatchHourInt+=1;
      }

      const scheduleKey = `dispatch:${dispatchId}`;
      const startTime = Date.now(); // 当前时间 + 延迟时间

      await redisClient.hSet(scheduleKey, "startTime", startTime);
      await redisClient.hSet(scheduleKey, "calcDelay", time);

      let expireTime = parseInt(time/1000)
      await redisClient.expire(scheduleKey, expireTime);

      await dispatchQueue.add(
          {
            type:"dispatch",
            number,
            startStation,
            endStation,
            dispatchDate,
            dispatchHour,
            dispatchId
          },{
            delay: time,
            attempts:3
          }
        );

      res.status(200).send({
          code:200,
          msg:"开始进行调度",
          time
      })
    }  
  }else{
      if(startRows.length == 0 || endRows.length == 0){
          res.status(400).send({//客户端参数错误
              code:400,
              error:"无效站点"
          })
      }else{
          res.status(500).send({
              code:500,
              error:"调度失败"
          })
      }
  }
});

/**
 * @api {post} /cancelChange Cancel Bike Dispatch
 * 
 * @apiDescription Cancels a bike dispatch between two stations and schedules return trips.
 *
 * @apiBody {String} startStation ID of the starting station
 * @apiBody {String} endStation ID of the destination station
 * @apiBody {Number} number Number of bikes being dispatched
 * @apiBody {String} dispatchDate Date of dispatch (YYYY-MM-DD format)
 * @apiBody {Number} dispatchHour Hour of dispatch (0-23)
 * @apiBody {String} dispatchId Unique ID of the dispatch
 *
 * @apiSuccess {Number} code HTTP status code (200)
 * @apiSuccess {String} msg Success message
 *
 * @apiError {Number} code HTTP status code (400)
 * @apiError {String} msg Error message when stations don't exist
 *
 * @apiError {Number} code HTTP status code (500)
 * @apiError {String} msg Internal server error
 */
router.post('/cancelChange',authMiddleware, async (req,res) => {
  let { startStation,endStation,number,dispatchDate, dispatchHour,dispatchId } = req.body
  
  const queryStartsql = "select 1 from `station_real_data` where `station_id` = ? "
  const queryEndSql = "select 1 from `station_real_data` where `station_id` = ? "
  
  let {err:startErr,rows:startRows} = await db.async.all(queryStartsql,[startStation])
  let {err:endErr,rows:endRows} = await db.async.all(queryEndSql,[endStation])

  if(startErr == null && endErr == null && startRows.length > 0 && endRows.length > 0){
  
    let {lat:startLat,lng:startLng} = await getStationCoordinates(startStation)
    let {lat:endLat,lng:endLng} = await getStationCoordinates(endStation)

    const distance = calcLength(startLat,startLng,endLat,endLng)

    let time = (distance/1000/20)*60*60*1000/20

    dispatchHour=parseInt(dispatchHour);
    
    const scheduleKey = `dispatch:${dispatchId}`;
    let startTime, calcDelay;
    const result = await redisClient.hGetAll(scheduleKey);

    if (Object.keys(result).length === 0) {
      console.log("没有找到对应的调度任务");
    } else {
      startTime = result.startTime
      calcDelay = result.calcDelay

      await redisClient.del(scheduleKey);
    }
    await dispatchQueue.add(
        {
          type:"cancelStart",
          number,
          startStation,
          endStation,
          dispatchDate,
          dispatchHour,
          dispatchId
        },{
          delay: (Date.now() - startTime),
          attempts:3
        }
      );

    await dispatchQueue.add(
        {
          type:"cancelEnd",
          number,
          startStation,
          endStation,
          dispatchDate,
          dispatchHour,
          dispatchId
        },{
          delay: time - (Date.now() - startTime),
          attempts:3
        }
      );
    res.status(200).send({
        code:200,
        msg:"开始返回调度",
        time:(Date.now() - startTime)
    })
  }
});

/**
 * 根据站点ID获取经纬度
 * @param {String} stationId 
 * @returns 
 */
async function getStationCoordinates(stationId) {
  try {
    const station = await Station.findOne({
      attributes: ['lat', 'lng'], 
      where: {
        station_id: stationId
      },
      raw: true
    });

    if (!station) {
      throw new Error(`未找到站点记录`);
    }

    return {
      lat: station.lat,
      lng: station.lng
    };
  } catch (error) {
    throw error;
  }
}

/**
 * 计算两个经纬度坐标之间的距离（单位：米）
 * @param {number} lat1 - 起点纬度
 * @param {number} lng1 - 起点经度
 * @param {number} lat2 - 终点纬度
 * @param {number} lng2 - 终点经度
 * @returns {number} 两点之间的距离（米）
 */
function calcLength(lat1, lng1, lat2, lng2) {
  
  if (isNaN(lat1)) throw new Error('无效的起点纬度');
  if (isNaN(lng1)) throw new Error('无效的起点经度');
  if (isNaN(lat2)) throw new Error('无效的终点纬度');
  if (isNaN(lng2)) throw new Error('无效的终点经度');

  // 将经纬度从度数转换为弧度
  const toRad = (degree) => degree * Math.PI / 180;
  const R = 6371000; // 地球半径（米）

  const φ1 = toRad(lat1);
  const φ2 = toRad(lat2);
  const Δφ = toRad(lat2 - lat1);
  const Δλ = toRad(lng2 - lng1);

  // Haversine公式
  const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ/2) * Math.sin(Δλ/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

  return Math.round(R * c);
}



StationSchedule.belongsTo(StationInfo, { foreignKey: 'start_id', as: 'startStation' });
StationSchedule.belongsTo(StationInfo, { foreignKey: 'end_id', as: 'endStation' });
/**
 * @api {get} /api/v1/schedules 查询调度任务
 * @apiDescription 接收一个具体时间，查询该日期、该（向前取整）小时的所有已规划调度任务。
 * @apiParam {String} query_time ISO 8601 格式的查询时间 (e.g., "2025-06-13T09:45:00Z").
 */
router.get('/',authMiddleware, async (req, res) => {
    const { query_time } = req.query;

    if (!query_time) {
        return res.status(400).json({ error: 'Missing required parameter: query_time.' });
    }

    try {
        const queryDate = new Date(query_time);
        if (isNaN(queryDate.getTime())) {
            return res.status(400).json({ error: 'Invalid query_time format. Please use ISO 8601 format.' });
        }

        // --- 核心逻辑：计算目标日期和3小时周期的起始小时 ---
        const dateForQuery = toYYYYMMDD(queryDate);
        const originalHour = queryDate.getUTCHours();
        // 向下取整到最近的3的倍数
        const hourForQuery = Math.floor(originalHour / 3) * 3;

        // 使用 Sequelize ORM 进行查询
        const schedules = await StationSchedule.findAll({
            where: {
                date: dateForQuery,
                hour: hourForQuery
            },
            include: [
                {
                    model: StationInfo,
                    as: 'startStation',
                    attributes: ['station_id', 'station_name', 'lat', 'lng']
                },
                {
                    model: StationInfo,
                    as: 'endStation',
                    attributes: ['station_id', 'station_name', 'lat', 'lng']
                }
            ],
            attributes: { exclude: ['start_id', 'end_id'] },
            order: [['id', 'ASC']] // 按ID升序排序
        });

        // 将查询结果格式化为 API 需要的最终格式
        const formattedSchedules = schedules.map(schedule => ({
            schedule_id: schedule.id,
            bikes_to_move: schedule.bikes,
            status: mapScheduleStatus(schedule.status),
            start_station: schedule.startStation ? { // 添加空值判断，防止关联数据不存在时报错
                id: schedule.startStation.station_id,
                name: schedule.startStation.station_name,
                lat: schedule.startStation.lat,
                lng: schedule.startStation.lng
            } : null,
            end_station: schedule.endStation ? {
                id: schedule.endStation.station_id,
                name: schedule.endStation.station_name,
                lat: schedule.endStation.lat,
                lng: schedule.endStation.lng
            } : null,
            updated_at: schedule.updated_at
        }));

        res.json({
            lookup_date: dateForQuery,
            lookup_hour: hourForQuery,
            schedules: formattedSchedules
        });
    } catch (err) {
        console.error('Schedule Lookup API Error (Sequelize):', err);
        res.status(500).json({ error: 'An internal server error occurred.' });
    }
});

// 辅助函数，将数据库中的 status 数字映射为可读文本
// 0: pending, 1: executing, 2: completed
const mapScheduleStatus = (statusInt) => {
    switch (statusInt) {
        case 0: return '待执行';
        case 1: return '正在执行';
        case 2: return '已完成';
        default: return '未知';
    }
};

/**
 * @api {post} /reject 管理员拒绝该调度请求
 * @apiDescription 管理员通过调度ID拒绝该调度请求，只有状态为0（未使用）的调度可以被拒绝，拒绝后状态变为-1。
 *
 * @apiHeader {String} Authorization 用户登录令牌，需通过authMiddleware验证。
 *
 * @apiParam {String} id 调度ID，必须。
 *
 * @apiSuccess {Number} code 状态码，200表示成功。
 * @apiSuccess {String} msg 返回提示信息，"已拒绝该调度"。
 *
 * @apiError (400) {Number} code 400
 * @apiError (400) {String} err 错误信息，可能为"参数id不能为空"或"该调度已使用"。
 *
 * @apiError (404) {Number} code 404
 * @apiError (404) {String} err 错误信息，调度不存在。
 *
 * @apiError (500) {Number} code 500
 * @apiError (500) {String} msg "服务器错误"
 * @apiError (500) {String} err 具体错误信息。
 */
router.post('/reject',authMiddleware, async (req,res)=>{
  let {id} = req.body;

  if (!id) {
    return res.status(400).send({
      code: 400,
      err: "参数id不能为空"
    });
  }

  const trySql = " select `status` from `station_schedule` where `id` = ? ;"
  let {err,rows} = await db.async.all(trySql,[id])
  console.log(rows[0].status)
  if(rows[0].status != 0){
    return res.status(400).send({
      code:400,
      err:"该调度已使用"
    });
  }else{
    try{
      const statusSql = " delete from `station_schedule` where `id` = ?;"
      await db.async.run(statusSql,[id])
      res.status(200).send({
        code:200,
        msg:"已拒绝该调度"
      })
    }catch(err){
      res.status(200).send({
        code:500,
        msg:"服务器错误",
        err
      })
    }
  }
})

/**
 * @api {get} /api/v1/schedules/by-station 按站点和时间查询调度任务
 * @apiDescription 查询特定站点在指定时间所属的3小时周期内的所有相关任务。
 * @apiParam {String} station_id 站点的ID。
 * @apiParam {String} query_time ISO 8601 格式的查询时间。
 * @apiParam {String} [role] 可选, 站点的角色 ('start' 或 'end')。
 */
router.get('/by-station', authMiddleware, async (req, res) => {
    const { station_id, query_time, role } = req.query;

    // 1. 参数校验
    if (!station_id || !query_time) {
        return res.status(400).json({ error: 'Missing required parameters: station_id and query_time.' });
    }
    if (role && !['start', 'end'].includes(role)) {
        return res.status(400).json({ error: "Invalid role parameter. Allowed values are 'start' or 'end'." });
    }

    try {
        const queryDate = new Date(query_time);
        if (isNaN(queryDate.getTime())) {
            return res.status(400).json({ error: 'Invalid query_time format. Please use ISO 8601 format.' });
        }

        // (可选但推荐) 校验站点是否存在
        const { rows: stationExistsRows } = await db.async.all('SELECT 1 FROM station_info WHERE station_id = ? LIMIT 1', [station_id]);
        if (stationExistsRows.length === 0) {
            return res.status(404).json({ error: `Station with ID ${station_id} not found.` });
        }

        // 2. 动态构建 SQL 查询
        const dateForQuery = toYYYYMMDD(queryDate);
        const hourForQuery = Math.floor(queryDate.getUTCHours() / 3) * 3;

        // 基础SQL，使用别名 s, start_info, end_info
        let baseSql = `
            SELECT
                s.id AS schedule_id,
                s.bikes,
                s.status,
                s.updated_at,
                start_info.station_id AS start_station_id,
                start_info.station_name AS start_station_name,
                start_info.lat AS start_lat,
                start_info.lng AS start_lng,
                end_info.station_id AS end_station_id,
                end_info.station_name AS end_station_name,
                end_info.lat AS end_lat,
                end_info.lng AS end_lng
            FROM
                station_schedule AS s
            LEFT JOIN
                station_info AS start_info ON s.start_id = start_info.station_id
            LEFT JOIN
                station_info AS end_info ON s.end_id = end_info.station_id
        `;

        let whereConditions = 'WHERE s.date = ? AND s.hour = ?';
        let params = [dateForQuery, hourForQuery];

        // 动态添加站点和角色相关的查询条件
        if (role === 'start') {
            whereConditions += ' AND s.start_id = ?';
            params.push(station_id);
        } else if (role === 'end') {
            whereConditions += ' AND s.end_id = ?';
            params.push(station_id);
        } else {
            whereConditions += ' AND (s.start_id = ? OR s.end_id = ?)';
            params.push(station_id, station_id);
        }

        const finalSql = `${baseSql} ${whereConditions} ORDER BY s.id ASC;`;

        // 3. 执行查询
        const { rows: schedules } = await db.async.all(finalSql, params);

        // 4. 格式化响应
        const formattedSchedules = schedules.map(schedule => ({
            schedule_id: schedule.schedule_id,
            bikes_to_move: schedule.bikes,
            status: mapScheduleStatus(schedule.status),
            start_station: schedule.start_station_id ? {
                id: schedule.start_station_id,
                name: schedule.start_station_name,
                lat: schedule.start_lat,
                lng: schedule.start_lng
            } : null,
            end_station: schedule.end_station_id ? {
                id: schedule.end_station_id,
                name: schedule.end_station_name,
                lat: schedule.end_lat,
                lng: schedule.end_lng
            } : null,
            updated_at: schedule.updated_at
        }));

        res.json({
            lookup_date: dateForQuery,
            lookup_hour: hourForQuery,
            station_id: station_id,
            role_filter: role || 'all',
            schedules: formattedSchedules
        });

    } catch (err) {
        console.error('Schedule by Station & Time Lookup API Error (SQL):', err);
        res.status(500).json({ error: 'An internal server error occurred.' });
    }
});

async function afterTimeSchedule(number,startStation,dispatchDate,dispatchHour){
  const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
}

router.post('/edit',authMiddleware, async (req,res)=>{
  let {id,bikes} = req.body
  try{
    const editSql = 'update `station_schedule` set `bikes` = ? where `id` = ?;'
    await db.async.run(editSql,[bikes,id])
    return res.status(200).send({
      code:200,
      msg:"调度信息修改成功"
    })
  }catch(err){
    return res.status(500).send({
      code:500,
      err
    })
  }  
});

/**
 * @api {post} /api/v1/dispatch/add 新增调度任务
 * @apiDescription 将一条新的调度方案添加到 station_schedule 表中。
 */
router.post('/add', authMiddleware, async (req, res) => {
    const { schedule_time, start_station_id, end_station_id, bikes_to_move } = req.body;

    // 1. 输入验证
    if (!schedule_time || !start_station_id || !end_station_id || !bikes_to_move) {
        return res.status(400).json({ error: "请求体参数不完整或类型错误，需要 schedule_time, start_station_id, end_station_id, bikes_to_move。" });
    }
    if (typeof bikes_to_move !== 'number' || bikes_to_move <= 0) {
        return res.status(400).json({ error: "调度车辆数 (bikes_to_move) 必须为正整数。" });
    }
    if (start_station_id === end_station_id) {
        return res.status(400).json({ error: "起点站和终点站不能为同一站点。" });
    }

    const scheduleDate = new Date(schedule_time);
    if (isNaN(scheduleDate.getTime())) {
        return res.status(400).json({ error: '无效的 schedule_time 格式。' });
    }

    try {
        // 2. 验证站点ID是否存在 (并发查询提高效率)
        const [startStationRows, endStationRows] = await Promise.all([
            db.async.all('SELECT 1 FROM station_info WHERE station_id = ?', [start_station_id]),
            db.async.all('SELECT 1 FROM station_info WHERE station_id = ?', [end_station_id])
        ]);

        if (startStationRows.rows.length === 0) {
            return res.status(404).json({ error: `起点站ID '${start_station_id}' 不存在。` });
        }
        if (endStationRows.rows.length === 0) {
            return res.status(404).json({ error: `终点站ID '${end_station_id}' 不存在。` });
        }

        // 3. 准备插入的数据
        const dateForQuery = toYYYYMMDD(scheduleDate);
        // 注意：这里是按照3小时周期逻辑来存储，如果希望直接存原始小时，可以去掉 Math.floor
        const hourForQuery = Math.floor(scheduleDate.getUTCHours() / 3) * 3;
        const newUpdatedAt = new Date();

        // 4. 执行 INSERT 操作
        const sql = `
            INSERT INTO station_schedule (date, hour, start_id, end_id, bikes, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        `;
        // status 默认为 0 (pending)
        const params = [dateForQuery, hourForQuery, start_station_id, end_station_id, bikes_to_move, 0, newUpdatedAt];

        const { result } = await db.async.run(sql, params);

        // 5. 构造并返回成功响应
        res.status(201).json({
            message: "调度任务已成功添加。",
            schedule: {
                id: result.insertId, // 获取新插入记录的自增ID
                date: dateForQuery,
                hour: hourForQuery,
                start_id: start_station_id,
                end_id: end_station_id,
                bikes: bikes_to_move,
                status: 0,
                updated_at: newUpdatedAt.toISOString()
            }
        });

    } catch (err) {
        console.error('Add Schedule API Error:', err);
        res.status(500).json({ error: '服务器内部错误。' });
    }
});


module.exports = router;