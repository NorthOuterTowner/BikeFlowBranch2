const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

async function authMiddleware(req, res, next) {
    const account = req.header('account');
    const token = req.header('token');
    if (!account || !token) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    try {
        const { err, rows } = await db.async.all(
            'SELECT * FROM admin WHERE account = ? AND token = ?',
            [account, token]
        );
        if (rows.length === 0) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        req.user = rows[0];
        next();
    } catch (err) {
        res.status(500).json({ error: 'Auth check failed' });
    }
}

router.get('/locations', authMiddleware, async (req, res) => {
  try {
      const { err, rows } = await db.async.all(
          `SELECT DISTINCT start_station_id AS station_id,
                           start_station_name AS station_name,
                           start_lat AS latitude,
                           start_lng AS longitude
           FROM bike_trip`
      );
    res.status(200).json(rows);
  } catch (err) {
    res.status(500).json({ error: '数据库查询失败' });
  }
});

router.get("/bikeNum",authMiddleware, async(req,res)=>{
  const {station_id,date,hour} = req.query;
  const querySql = "select `stock` from `station_real_data` where `station_id` = ? and `date` = ? and `hour` = ? "
  let {err,rows} = await db.async.all(querySql,[station_id,date,hour])
  if(err==null && rows.length > 0){
    res.send({
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

    res.send({
      code:500,
      msg:errMsg
    })
  }
});

router.get("/bikeNum/timeAll",authMiddleware,async(req,res)=>{
  const {date,hour} = req.query
  const querySql = " select `station_id`,`stock` from `station_real_data` where `date` = ? and `hour` = ? "
  let {err,rows} = await db.async.all(querySql,[date,hour])
  if(err == null && rows.length > 0){
    res.send({
      code:200,
      rows
    })
  }
})

router.get("/bikeNum/stationAll",authMiddleware,async(req,res)=>{
  const {station_id} = req.query
  const querySql = " select `date`,`hour`,`stock` from `station_real_data` where `station_id` = ? "
  let {err,rows} = await db.async.all(querySql,[station_id])
  if(err == null && rows.length > 0){
    res.send({
      code:200,
      rows
    })
  }
})


module.exports = router;