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

/**
 * @api {get} /flow/hour 查询小时总流量
 * @apiDescription 传入一个整点时间，返回该小时内系统中所有站点的总流入、总流出和总流量。
 * @access Private
 */
router.get('/flow/hour', authMiddleware, async (req, res) => {
    // 1. 从请求查询参数中获取时间点
    const { query_time } = req.query;

    // 2. 参数校验
    if (!query_time) {
        return res.status(400).json({ error: '请求失败，缺少 "query_time" 参数。' });
    }

    const queryDate = new Date(query_time);
    if (isNaN(queryDate.getTime())) {
        return res.status(400).json({ error: '无效的时间格式。请使用 ISO 8601 格式。' });
    }
    // 校验是否为整点时间
    if (queryDate.getUTCMinutes() !== 0 || queryDate.getUTCSeconds() !== 0 || queryDate.getUTCMilliseconds() !== 0) {
        return res.status(400).json({ error: '请求的时间点必须为整点时间。' });
    }

    try {
        // 3. 准备SQL查询
        // 使用 SUM() 聚合函数来计算总和
        const sql = `
            SELECT
                SUM(inflow) AS total_inflow,
                SUM(outflow) AS total_outflow
            FROM
                station_hourly_flow
            WHERE
                timestamp = ?;
        `;

        // 格式化时间以匹配数据库的 DATETIME 类型
        // 'YYYY-MM-DD HH:MM:SS'
        const timestampForQuery = queryDate.toISOString().slice(0, 19).replace('T', ' ');
        const params = [timestampForQuery];

        // 4. 执行查询
        const { rows } = await db.async.all(sql, params);

        // 5. 处理并返回结果
        if (!rows || rows.length === 0 || rows[0].total_inflow === null) {
            // 如果查询结果为空 (SUM返回null)，说明该小时没有流量数据
            return res.json({
                query_time: query_time,
                total_inflow: 0,
                total_outflow: 0,
                total_flow: 0
            });
        }

        const result = rows[0];
        const totalInflow = parseInt(result.total_inflow, 10);
        const totalOutflow = parseInt(result.total_outflow, 10);

        res.json({
            query_time: query_time,
            total_inflow: totalInflow,
            total_outflow: totalOutflow,
            total_flow: totalInflow + totalOutflow // 总流量 = 总流入 + 总流出
        });

    } catch (err) {
        console.error('Hourly Flow API Error:', err);
        res.status(500).json({ error: '服务器内部错误。' });
    }
});

module.exports = router