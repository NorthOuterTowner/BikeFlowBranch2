const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils");
const authMiddleware = require('../utils/auth');

/**
 * 指定时间段总流量
 */
router.post("/flow/time", authMiddleware, async (req, res) => {
    try {
        const { startDate, startHour, endDate, endHour } = req.body;
        
        // 解析日期和时间
        const [year1, month1, day1] = startDate.split('-').map(Number);
        const [year2, month2, day2] = endDate.split('-').map(Number);
        
        // 创建Date对象并格式化为MySQL兼容格式
        const formatDateForMySQL = (date) => {
            return date.toISOString().slice(0, 19).replace('T', ' ');
        };
        
        const startDt = formatDateForMySQL(new Date(year1, month1-1, day1, startHour, 0, 0));
        const endDt = formatDateForMySQL(new Date(year2, month2-1, day2, endHour, 0, 0));
        
        console.log("查询时间范围:", startDt, "至", endDt);

        // 优化：使用一个查询同时获取流入和流出总量
        const flowDataSql = `
            SELECT 
                SUM(inflow) AS total_inflow,
                SUM(outflow) AS total_outflow
            FROM station_hourly_flow 
            WHERE timestamp BETWEEN ? AND ?;
        `;
        
        // 执行查询
        const { err, rows } = await db.async.all(flowDataSql, [startDt, endDt]);
        
        if (err) {
            console.error("数据库查询错误:", err);
            return res.status(500).send({
                code: 500,
                message: "数据库查询错误",
                error: err.message
            });
        }

        // 处理结果
        const totalInflow = parseInt(rows[0]?.total_inflow || 0);
        const totalOutflow = parseInt(rows[0]?.total_outflow || 0);
        const totalFlow = totalInflow + totalOutflow;
        
        console.log(`查询结果: 流入=${totalInflow}, 流出=${totalOutflow}, 总流量=${totalFlow}`);
        
        return res.status(200).send({
            code: 200,
            data: {
                inflow: totalInflow,
                outflow: totalOutflow,
                total: totalFlow
            }
        });
        
    } catch (error) {
        console.error("服务器错误:", error);
        return res.status(500).send({
            code: 500,
            message: "服务器内部错误",
            error: error.message
        });
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

/**
 * 指定开始时间和结束时间记录top10流量的站点
 */
router.post("/top", authMiddleware, async (req, res) => {
    const { startDate, startHour, endDate, endHour } = req.body;
    
    // 解析日期和时间
    const [year1, month1, day1] = startDate.split('-').map(Number);
    const [year2, month2, day2] = endDate.split('-').map(Number);
    
    // 创建Date对象
    const startDt = new Date(year1, month1-1, day1, startHour, 0, 0);
    const endDt = new Date(year2, month2-1, day2, endHour, 0, 0);
    
    // 格式化为MySQL兼容的DATETIME字符串
    function formatDateForMySQL(date) {
        return date.toISOString().slice(0, 19).replace('T', ' ');
    }
    
    const mysqlStartDt = formatDateForMySQL(startDt);
    const mysqlEndDt = formatDateForMySQL(endDt);
    
    console.log("Formatted dates:", mysqlStartDt, mysqlEndDt);
    
    try {
        const inTopSql = `
            SELECT station_id, SUM(inflow + outflow) AS total_flow 
            FROM station_hourly_flow 
            WHERE timestamp >= ? AND timestamp <= ? 
            GROUP BY station_id
            ORDER BY total_flow DESC 
            LIMIT 10;
        `;
        
        const { err, rows } = await db.async.all(inTopSql, [mysqlStartDt, mysqlEndDt]);
        
        if (err) {
            console.error("Database error:", err);
            return res.status(500).send({
                code: 500,
                message: "Database error",
                error: err.message
            });
        }
        
        return res.status(200).send({
            code: 200,
            data: rows
        });
        
    } catch (error) {
        console.error("Unexpected error:", error);
        return res.status(500).send({
            code: 500,
            message: "Internal server error"
        });
    }
});
module.exports = router