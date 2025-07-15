const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils");
const authMiddleware = require('../utils/auth');

/**
 * 指定时间段总流量
 */
router.post("/flow/time",authMiddleware,async (req,res)=>{
    const {startDate,startHour,endDate,endHour} = req.body
    const [year1, month1, day1] = startDate.split('-').map(Number);
    const [year2, month2, day2] = endDate.split('-').map(Number);
    const startDt = new Date(year1, month1-1, day1, startHour, 0, 0);
    const endDt = new Date(year2, month2-1, day2, endHour, 0, 0);
    const inflowData = "select sum(inflow) as cnt from `station_hourly_flow` where `timestamp` >= ? and `timestamp` <= ? ;"
    const outflowData = "select sum(outflow) as cnt from `station_hourly_flow` where `timestamp` >= ? and `timestamp` <= ? ;"
    
    let {err:errIn,rows:rowIn} = await db.async.all(inflowData,[startDt,endDt])
    let {err:errOut,rows:rowOut} = await db.async.all(outflowData,[startDt,endDt])
    console.log(rowIn[0].cnt)
    console.log(rowOut[0].cnt)
    let flow = 0
    if(errIn == null && errOut == null){
        flow = parseInt(rowIn[0].cnt) + parseInt(rowOut[0].cnt)
        return res.status(200).send({
            code:200,
            flow
        })
    }else{
        return res.status(500).send({
            code:500,
            err:"服务器错误"
        })
    }

});

/**
 * 指定站点和时间段流入流出量
 */
router.post("/io/timeStation",authMiddleware,async (req,res)=>{

});

/**
 * 指定站点和日期各时间段流量
 */
router.post("/flow/station",authMiddleware,async (req,res)=>{

});

module.exports = router