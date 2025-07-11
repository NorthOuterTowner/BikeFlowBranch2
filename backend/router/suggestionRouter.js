const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');
const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)

const authMiddleware = require("../utils/auth")

/**
 * 根据DeepSeek获取调度建议
 */
router.get('/', authMiddleware, async (req, res) => {

    try {

        /**
         * TO-DO LIST:
         * 1.messages
         * 2.API key
         */
        let messages = ''

        const response = await axios.post(
        'https://api.deepseek.com/v1/chat/completions',
        {
            model: 'deepseek-chat',
            messages: messages,
            temperature: 0.7
        },
        {
            headers: {
            'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`,
            'Content-Type': 'application/json'
            }
        }
        );

        res.json(response.data);
    } catch (err) {
        console.error('调用 DeepSeek API 出错:', err.response?.data || err.message);
        res.status(500).json({ error: '调用 DeepSeek 失败' });
    }

});

module.exports = router;