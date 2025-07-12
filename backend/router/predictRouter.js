const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');

const authMiddleware = require("../utils/auth")

// 将 Date 对象格式化为 'YYYY-MM-DD'
const toYYYYMMDD = (date) => {
    return date.toISOString().slice(0, 10);
};

/**
 * @api {get} /api/v1/predict/station 查询单个站点的状态
 * @apiDescription 接收一个具体时间，查询 `station_hourly_status` 表中
 *                 该日期、该（向前取整）小时的已存储状态数据。
 * @apiParam {String} station_id 站点的ID (e.g., "JC019").
 * @apiParam {String} predict_time ISO 8601 格式的查询时间 (e.g., "2025-01-21T08:45:00Z").
 */
router.get('/station', authMiddleware,async (req, res) => {
    const { station_id, predict_time } = req.query;

    if (!station_id || !predict_time) {
        return res.status(400).json({ error: 'Missing required parameters: station_id and predict_time.' });
    }

    try {
        const queryDate = new Date(predict_time);
        if (isNaN(queryDate.getTime())) {
            return res.status(400).json({ error: 'Invalid predict_time format. Please use ISO 8601 format.' });
        }

        // --- 提取日期和小时（向前取整）---
        const dateForQuery = toYYYYMMDD(queryDate); // '2025-01-21'
        const hourForQuery = queryDate.getUTCHours();    // 8

        const sql = `
            SELECT inflow, outflow, stock
            FROM station_hourly_status
            WHERE
                station_id = ? AND
                date = ? AND
                hour = ?
            LIMIT 1;
        `;

        const params = [station_id, dateForQuery, hourForQuery];
        const { rows } = await db.async.all(sql, params);

        if (!rows || rows.length === 0) {
            return res.status(404).json({
                error: `No status data found for station ${station_id} on ${dateForQuery} at hour ${hourForQuery}.`
            });
        }

        // 直接返回查询到的数据
        res.json({
            station_id: station_id,
            lookup_date: dateForQuery,
            lookup_hour: hourForQuery,
            status: {
                inflow: rows[0].inflow,
                outflow: rows[0].outflow,
                stock: rows[0].stock
            }
        });

    } catch (err) {
        console.error('Station Status Lookup API Error:', err);
        return res.status(500).json({ error: 'An internal server error occurred.' });
    }
});

/**
 * @api {get} /api/v1/predict/stations/all 查询所有站点的状态
 * @apiDescription 接收一个具体时间，查询 `station_hourly_status` 表中
 *                 该日期、该（向前取整）小时，所有站点的已存储状态数据。
 * @apiParam {String} predict_time ISO 8601 格式的查询时间 (e.g., "2025-01-21T17:10:00Z").
 */
router.get('/stations/all', authMiddleware,async (req, res) => {
    const { predict_time } = req.query;

    if (!predict_time) {
        return res.status(400).json({ error: 'Missing required parameter: predict_time.' });
    }

    try {
        const queryDate = new Date(predict_time);
        if (isNaN(queryDate.getTime())) {
            return res.status(400).json({ error: 'Invalid predict_time format. Please use ISO 8601 format.' });
        }

        // --- 提取日期和小时（向前取整）---
        const dateForQuery = toYYYYMMDD(queryDate);
        const hourForQuery = queryDate.getUTCHours();

        const sql = `
            SELECT station_id, inflow, outflow, stock
            FROM station_hourly_status
            WHERE
                date = ? AND
                hour = ?;
        `;

        const params = [dateForQuery, hourForQuery];
        const { rows } = await db.async.all(sql, params);

        console.log('数据库查询参数:', params);
        console.log('数据库查询结果:', rows);


        if (!rows || rows.length === 0) {
            return res.status(404).json({
                error: `No status data found for any station on ${dateForQuery} at hour ${hourForQuery}.`
            });
        }

        const formattedStatuses = rows.map(row => ({
            station_id: row.station_id,
            inflow: row.inflow,
            outflow: row.outflow,
            stock: row.stock
        }));

        res.json({
            lookup_date: dateForQuery,
            lookup_hour: hourForQuery,
            stations_status: formattedStatuses
        });

    } catch (err) {
        console.error('All Stations Status Lookup API Error:', err);
        res.status(500).json({ error: 'An internal server error occurred.' });
    }
});

module.exports = router;
