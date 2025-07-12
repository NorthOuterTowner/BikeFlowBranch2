const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils");
const authMiddleware = require('../utils/auth');

const sequelize = require('../orm/sequelize');
const { DataTypes } = require('sequelize')
const ScheduleModel = require('../orm/models/stationSchedule');
const InfoModel = require('../orm/models/Station');
const StationSchedule = ScheduleModel(sequelize,DataTypes);
const StationInfo = InfoModel(sequelize,DataTypes);
const { Op } = require('sequelize');
/**
 * @api {get} /search/assign 查询某时段的调度任务
 * 
 * @apiDescription 根据指定的日期与小时，查询当前是否有需要调度的站点流程。
 * 
 * @apiQuery {String} date 指定的调度日期（格式：YYYY-MM-DD）
 * @apiQuery {Number} hour 指定的小时（0~23）
 * 
 * @apiSuccess {Number} code 状态码，成功为200
 * @apiSuccess {Object[]} [rows] 调度任务列表（有数据时返回）
 * @apiSuccess {String} rows.id 调度任务ID
 * @apiSuccess {String} rows.start_id 起始站点ID
 * @apiSuccess {String} rows.end_id 目标站点ID
 * @apiSuccess {Number} rows.bikes 调度车辆数

 * @apiError {Number} code 错误码，服务器内部错误为500
 * @apiError {String} msg 错误描述信息
 */

router.get("/assign",authMiddleware,async (req,res)=>{
  const {date,hour} = req.query
  const result = await StationSchedule.findAll({
    attributes:['id','start_id','end_id','bikes'],
    where:{
      date,
      hour
    },
    raw:true
  })
  if(result != null ){
    return res.status(200).send({
      code:200,
      result
    })
  }else if(result == null){
    return res.status(200).send({
      code:200,
      msg:"当前无需要调度的流程"
    })
  }else{
    return res.status(500).send({
      code:500,
      msg:"服务器内部错误"
    })
  }
})

/**
 * @api {get} /stationAssign Get Station Assignments
 * 
 * @apiQuery {String} date The date to query (format: YYYY-MM-DD)
 * @apiQuery {Number} hour The hour to query (0-23)
 *
 * @apiSuccess {Number} code HTTP status code (200)
 * @apiSuccess {Object[]} station_result Array of station information
 * @apiSuccess {String} station_result.station_name Name of the station
 * 
 * @apiError {Number} code HTTP status code (404)
 * @apiError {String} msg Error message
 */
router.get("/stationAssign",authMiddleware, async (req,res) =>{
  const {date,hour} = req.query

  const id_result = await StationSchedule.findAll({
    attributes: ['start_id'],
    where: {
      date,
      hour
    },
    raw: true
  });
  if(id_result==null){
    return res.status(404).send({
      code:404,
      msg:"无对应站点"
    })
  }
  const stationIds = id_result.map(item => item.start_id);
  console.log(stationIds)
  const station_result = await StationInfo.findAll({
    attributes: ['station_name'],
    where:{
      station_id:{
        [Op.in]:stationIds
      }
    },
    raw:true
  });

    return res.status(200).send({
        code:200,
        station_result
    })
})

module.exports = router