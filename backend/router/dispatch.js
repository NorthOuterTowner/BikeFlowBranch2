const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)

const authMiddleware = require("../utils/auth")

const ScheduleModel = require('../orm/models/stationSchedule');
const InfoModel = require('../orm/models/Station');
const StationSchedule = ScheduleModel(sequelize,DataTypes);
const StationInfo = InfoModel(sequelize,DataTypes);

/**
 * 执行调度：
 * 执行调度内容,根据调度内容从调度起点取车，计算调度完成时间，在调度完成后将调度终点增加单车余量
 */
router.post('/change', authMiddleware, async (req, res) => {
    let { startStation,endStation,number,dispatchDate, dispatchHour } = req.body
    
    const queryStartsql = "select 1 from `station_real_data` where `station_id` = ? "
    const queryEndSql = "select 1 from `station_real_data` where `station_id` = ? "
    
    let {err:startErr,rows:startRows} = await db.async.all(queryStartsql,[startStation])
    let {err:endErr,rows:endRows} = await db.async.all(queryEndSql,[endStation])

    if(startErr == null && endErr == null && startRows.length > 0 && endRows.length > 0){
        let changeableStock = 0
        const getStockSql = "select `stock` from `station_real_data` where `station_id` = ? and `date` = ? and `hour` = ? "
        let {err:searchErr,rows:searchRows} = await db.async.all(getStockSql,[startStation,dispatchDate,dispatchHour]) 

        changeableStock = searchRows[0].stock

        if(changeableStock < number){
            res.status(422).send({//语义错误
                code:422,
                error:"该调度方案不可行，调度数量超过本站点车余量"
            })
        }else{
            const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
            const changeSql2 = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
            
            await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
            
            let {lat:startLat,lng:startLng} = await getStationCoordinates(startStation)
            let {lat:endLat,lng:endLng} = await getStationCoordinates(endStation)

            const distance = calcLength(startLat,startLng,endLat,endLng)

            let time = (distance/1000/20)*60*60*1000

            setTimeout(async()=>{
                await db.async.run(changeSql2,[number,endStation,dispatchDate,dispatchHour])
            },time)
            
            res.status(200).send({
                code:200,
                msg:"开始进行调度"
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
router.get('/', async (req, res) => {
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
        case 0: return 'pending';
        case 1: return 'executing';
        case 2: return 'completed';
        default: return 'unknown';
    }
};

// 辅助函数，将 Date 对象格式化为 'YYYY-MM-DD'
const toYYYYMMDD = (date) => {
    return date.toISOString().slice(0, 10);
};

module.exports = router;