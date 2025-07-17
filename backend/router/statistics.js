const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils");
const authMiddleware = require('../utils/auth');

/**
 * @api {post} /flow/time 获取时间范围内总流量
 * @apiDescription 统计指定时间段内的总流入量、总流出量与总流量（inflow + outflow）。日期与小时精确到每小时，数据来源于 station_hourly_flow 表。
 * 
 * @apiBody {String} startDate 起始日期（格式：yyyy-mm-dd）
 * @apiBody {Number} startHour 起始小时（0~23）
 * @apiBody {String} endDate 结束日期（格式：yyyy-mm-dd）
 * @apiBody {Number} endHour 结束小时（0~23）

 * @apiSuccess {Number} code 状态码（200表示成功）
 * @apiSuccess {Object} data 数据对象
 * @apiSuccess {Number} data.inflow 总流入数量
 * @apiSuccess {Number} data.outflow 总流出数量
 * @apiSuccess {Number} data.total 总流量（流入 + 流出）
 * 
 * @apiError {Number} code 错误码
 * @apiError {String} message 错误信息
 * @apiError {String} [error] 错误详情（调试用）
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
 * @api {post} /top 获取流量前十的站点
 * @apiDescription 获取在指定时间段内，总流量（inflow + outflow）排名前10的站点。
 * 
 * @apiHeader {String} token 用户认证Token，格式为Bearer Token。
 * 
 * @apiBody {String} startDate 起始日期（格式为 "YYYY-MM-DD"）
 * @apiBody {Number} startHour 起始小时（0-23）
 * @apiBody {String} endDate 截止日期（格式为 "YYYY-MM-DD"）
 * @apiBody {Number} endHour 截止小时（0-23）
 * 
 * @apiSuccess {Number} code 返回码（200表示成功）
 * @apiSuccess {Object[]} data 前10个站点的流量数据数组
 * @apiSuccess {String} data.station_id 站点ID
 * @apiSuccess {Number} data.total_flow 流量总和（inflow + outflow）
 * 
 * @apiError {Number} code 错误码（500表示服务器错误）
 * @apiError {String} message 错误信息
 * @apiError {String} [error] 错误详情（可选）
 * 
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
 * @api {get} /flow/day 查询某天每小时的总流量
 * @apiDescription 传入一个日期，返回该日期内0-23点每小时的系统总流量数据。
 * @access Private
 */
router.get('/flow/day', authMiddleware, async (req, res) => {
    // 1. 从请求查询参数中获取日期
    const { query_date } = req.query;

    // 2. 参数校验
    if (!query_date) {
        return res.status(400).json({ error: '请求失败，缺少 "query_date" 参数。' });
    }

    // 验证日期格式是否有效 (例如 'YYYY-MM-DD')
    const queryDate = new Date(query_date);
    if (isNaN(queryDate.getTime()) || !/^\d{4}-\d{2}-\d{2}$/.test(query_date)) {
        return res.status(400).json({ error: '无效的日期格式。请使用 "YYYY-MM-DD" 格式。' });
    }

    try {
        // 3. 准备SQL查询
        // 使用 HOUR(timestamp) 来按小时分组
        // 使用 SUM() 来计算每小时的总和
        const sql = `
            SELECT
                HOUR(timestamp) AS hour,
                SUM(inflow) AS total_inflow,
                SUM(outflow) AS total_outflow
            FROM
                station_hourly_flow
            WHERE
                DATE(timestamp) = ?
            GROUP BY
                HOUR(timestamp)
            ORDER BY
                hour ASC;
        `;

        const params = [query_date];

        // 4. 执行查询
        const { rows } = await db.async.all(sql, params);

        // 5. 格式化并补全数据
        // 创建一个包含0-23点默认数据的数组
        const hourlyData = Array.from({ length: 24 }, (_, i) => ({
            hour: i,
            total_inflow: 0,
            total_outflow: 0,
            total_flow: 0
        }));

        // 用数据库返回的数据填充默认数组
        rows.forEach(row => {
            const hourIndex = row.hour;
            if (hourIndex >= 0 && hourIndex < 24) {
                const totalInflow = parseInt(row.total_inflow, 10);
                const totalOutflow = parseInt(row.total_outflow, 10);
                hourlyData[hourIndex] = {
                    hour: hourIndex,
                    total_inflow: totalInflow,
                    total_outflow: totalOutflow,
                    total_flow: totalInflow + totalOutflow
                };
            }
        });

        // 6. 返回最终结果
        res.json({
            query_date: query_date,
            hourly_flows: hourlyData
        });

    } catch (err) {
        console.error('Daily Flow API Error:', err);
        res.status(500).json({ error: '服务器内部错误。' });
    }
});

function formatDateToLocalYYYYMMDD(date) {
    if (!(date instanceof Date) || isNaN(date)) {
        return null; // 或者返回一个默认值，如 'N/A'
    }

    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0'); // 月份从0开始，需要+1，并补零
    const day = String(date.getDate()).padStart(2, '0'); // 补零

    return `${year}-${month}-${day}`;
}

/**
 * @api {get} /flow/days 查询每日总流量
 * @apiDescription 传入一个日期，返回该日期之前15天内，每一天的总流入、总流出和总流量。
 * @access Private
 */
router.get('/flow/days', authMiddleware, async (req, res) => {
    // 1. 从请求查询参数中获取目标日期
    const { target_date } = req.query;

    // 2. 参数校验
    if (!target_date) {
        return res.status(400).json({ error: '请求失败，缺少 "target_date" 参数。' });
    }

    const targetDateObj = new Date(target_date);
    if (isNaN(targetDateObj.getTime())) {
        return res.status(400).json({ error: '无效的日期格式。请使用 YYYY-MM-DD 格式。' });
    }

    // 格式化为 'YYYY-MM-DD'，以避免时区问题
    const formattedTargetDate = targetDateObj.toISOString().slice(0, 10);

    try {
        // 3. 准备SQL查询
        // - WHERE timestamp >= DATE_SUB(?, INTERVAL 15 DAY): 查询范围是目标日期前15天
        // - WHERE timestamp < ?: 查询范围不包括目标日期当天
        // - GROUP BY DATE(timestamp): 按天对数据进行分组
        // - ORDER BY flow_date ASC: 按日期升序排列结果
        const sql = `
            SELECT
                DATE(timestamp) AS flow_date,
                SUM(inflow) AS total_inflow,
                SUM(outflow) AS total_outflow
            FROM
                station_hourly_flow
            WHERE
                timestamp >= DATE_SUB(?, INTERVAL 15 DAY) AND timestamp < ?
            GROUP BY
                flow_date
            ORDER BY
                flow_date ASC;
        `;

        const params = [formattedTargetDate, formattedTargetDate];

        // 4. 执行查询
        const { rows } = await db.async.all(sql, params);

        // 5. 格式化并返回结果
        const formattedResult = rows.map(row => {
            // 使用新的辅助函数来格式化日期
            const localDate = formatDateToLocalYYYYMMDD(row.flow_date);

            // 对数值进行一次解析，然后复用，代码更清晰
            const inflow = parseInt(row.total_inflow, 10) || 0;
            const outflow = parseInt(row.total_outflow, 10) || 0;

            return {
                date: localDate,
                total_inflow: inflow,
                total_outflow: outflow,
                total_flow: inflow + outflow // 直接使用变量相加
            };
        });

        res.json({
            target_date: formattedTargetDate,
            daily_summary: formattedResult
        });

    } catch (err) {
        console.error('Daily Flow Summary API Error:', err);
        res.status(500).json({ error: '服务器内部错误。' });
    }
});

module.exports = router