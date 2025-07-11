const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)

const authMiddleware = require("../utils/auth")

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

            //change status of dispatch
            const statusSql = "update `station_schedule` set `status` = 1 where `id` = ? ;"
            await db.async.run(statusSql,[dispatchId])

            setTimeout(async()=>{
                await db.async.run(changeSql2,[number,endStation,dispatchDate,dispatchHour])
            },time)

            dispatchHour+=1;
            while(dispatchHour<=23){
              afterTimeSchedule(number,startStation,endStation,dispatchDate,dispatchHour);
              dispatchHour++;
            }

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

/**
 * 递归进行更改，将调度时间之后的所有时间对应余量均进行更改
 * @param {*} number 
 * @param {*} startStation 
 * @param {*} endStation 
 * @param {*} dispatchDate 
 * @param {*} dispatchHour 
 */
async function afterTimeSchedule(number,startStation,endStation,dispatchDate,dispatchHour){
  const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  const changeSql2 = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
  await db.async.run(changeSql2,[number,endStation,dispatchDate,dispatchHour])
}


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
      const statusSql = " update `station_schedule` set `status` = -1 where `id` = ?;"
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

module.exports = router;