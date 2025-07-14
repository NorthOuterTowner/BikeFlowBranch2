const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const authMiddleware = require("../utils/auth")


/**
 * @api {get} /locations 查询所有站点信息
 * @apiDescription 从 station_info 表获取所有站点的基础信息列表。
 *
 * @apiSuccess {Object[]} rows 站点列表
 * @apiSuccess {String} rows.station_id 站点ID
 * @apiSuccess {String} rows.station_name 站点名称
 * @apiSuccess {Number} rows.lat 纬度
 * @apiSuccess {Number} rows.lng 经度
 * @apiSuccess {Number} rows.capacity 站点容量
 *
 * @apiError (500) {Object} error 数据库查询失败
 */
router.get('/locations', authMiddleware, async (req, res) => {
  try {
    // --- 修改点：从 station_info 表读取 ---
    const sql = `
        SELECT
            station_id,
            station_name,
            lat AS latitude,
            lng AS longitude,
            capacity
        FROM
            station_info
        ORDER BY
            station_id ASC; -- 按ID排序，使返回结果稳定
    `;

    const { err, rows } = await db.async.all(sql);

    // 数据库查询本身出错
    if (err) {
      console.error('Failed to fetch station locations:', err);
      // 将具体的数据库错误信息记录在服务器日志，但返回给客户端一个通用的错误消息
      return res.status(500).json({ error: '数据库查询失败' });
    }

    // 成功返回查询结果
    return res.status(200).json(rows);

  } catch (err) {
    // 捕获 Promise aync/await 过程中的意外错误
    console.error('Unexpected error in /locations route:', err);
    return res.status(500).json({ error: '服务器内部错误' });
  }
});

/**
 * @api {get} /bikeNum 获取指定站点在指定时刻的单车余量
 *
 * @apiQuery {String} station_id 站点ID
 * @apiQuery {String} date 日期（格式：YYYY-MM-DD）
 * @apiQuery {Number} hour 小时（0-23）
 *
 * @apiSuccess {Number} code 200
 * @apiSuccess {Number} bikeNum 单车余量
 *
 * @apiError (500) {Number} code 500
 * @apiError (500) {String} msg 错误信息（"无此站点" | "无此日期数据" | "查询失败"）
 */
router.get("/bikeNum",authMiddleware, async(req,res)=>{
  const {station_id,date,hour} = req.query;
  const querySql = "select `stock` from `station_real_data` where `station_id` = ? and `date` = ? and `hour` = ? "
  let {err,rows} = await db.async.all(querySql,[station_id,date,hour])
  if(err==null && rows.length > 0){
    return res.send({
      code:200,
      bikeNum:rows[0].stock
    })
  }else{
    let errMsg = "查询失败"
    try{
      const trySql1 = "select * from `station_real_data` where `station_id` = ?"
      let {err:err1,rows:info1} = await db.async.all(trySql1,[station_id])
      if(info1.length > 0){
        const trySql2 = "select * from `station_real_data` where `date` = ?"
        let {err:err2,rows:info2} = await db.async.all(trySql2,[date])
        if(info2.length == 0){
          errMsg = "无此日期数据"
        }
      }else{
        errMsg = "无此站点"
      }
    }catch(e){
      errMsg = "查询失败"
    }

    return res.status(500).send({
      code:500,
      msg:errMsg
    })
  }
});

/**
 * @api {get} /bikeNum/timeAll 获取指定时间所有站点的实际单车余量
 * @apiQuery {String} date 日期（格式：YYYY-MM-DD）
 * @apiQuery {Number} hour 小时（0-23）
 *
 * @apiSuccess {Number} code 200
 * @apiSuccess {Object[]} rows 单车余量列表
 * @apiSuccess {String} rows.station_id 站点ID
 * @apiSuccess {Number} rows.stock 单车余量
 *
 * @apiError (400) {Number} code 400
 * @apiError (400) {String} msg "参数缺失"
 */
router.get("/bikeNum/timeAll",authMiddleware,async(req,res)=>{
  const {date,hour} = req.query
  if(!date || !hour){
    return res.status(400).send({
      code:400,
      msg:"参数缺失"
    })
  }
  const querySql = " select `station_id`,`stock` from `station_real_data` where `date` = ? and `hour` = ? "
  let {err,rows} = await db.async.all(querySql,[date,hour])
  if(err == null && rows.length > 0){
    return res.status(200).send({
      code:200,
      rows
    })
  }
})

/**
 * @api {get} /bikeNum/stationAll 获取指定站点在所有时间的实际单车余量
 *
 * @apiQuery {String} station_id 站点ID
 *
 * @apiSuccess {Number} code 200
 * @apiSuccess {Object[]} rows 单车余量列表
 * @apiSuccess {String} rows.date 日期（YYYY-MM-DD）
 * @apiSuccess {Number} rows.hour 小时（0-23）
 * @apiSuccess {Number} rows.stock 单车余量
 */
router.get("/bikeNum/stationAll",authMiddleware,async(req,res)=>{
  const {station_id} = req.query
  const querySql = " select `date`,`hour`,`stock` from `station_real_data` where `station_id` = ? "
  let {err,rows} = await db.async.all(querySql,[station_id])
  if(err == null && rows.length > 0){
    return res.status(200).send({
      code:200,
      rows
    })
  }
})


module.exports = router;