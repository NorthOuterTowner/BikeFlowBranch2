const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)

const authMiddleware = require("../utils/auth")

/**
 * 执行调度：
 * 执行调度内容,根据调度内容从调度起点取车，计算调度完成时间，在调度完成后将调度终点增加单车余量
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
 * TO-DO：管理员拒绝该调度请求
 */
router.post('/reject',authMiddleware, async (req,res)=>{
  let {id} = req.body;
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
})

module.exports = router;