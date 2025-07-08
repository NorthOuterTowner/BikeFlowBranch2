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
        const [rows] = await db.query(
            'SELECT * FROM user WHERE account = ? AND token = ?',
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
    const [rows] = await db.query(
      `SELECT DISTINCT start_station_id AS station_id, start_station_name AS station_name, start_lat AS latitude, start_lng AS longitude FROM bike_trip`
    );
    res.status(200).json(rows);
  } catch (err) {
    res.status(500).json({ error: '数据库查询失败' });
  }
});

module.exports = router;