const express = require('express');
const router = express.Router();
const { db } = require('../db/dbUtils');
const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const StationModel = require('../orm/models/Station');
const Station = StationModel(sequelize,DataTypes)
const axios = require('axios'); // 需要引入axios来调用外部API

const authMiddleware = require("../utils/auth")


// /**
//  * 根据站点数据生成给AI的提示词 (Prompt)
//  * @param {Array} stationData - 从数据库获取的站点数据
//  * @param {string} targetTimeISO - ISO格式的目标时间字符串
//  * @returns {Array} - 符合DeepSeek API格式的messages数组
//  */
// function generateSchedulingPrompt(stationData, targetTimeISO) {
//     // 将站点数据格式化为易于AI阅读的字符串
//     const formattedStationData = JSON.stringify(stationData, null, 2);
//
//     // 这是提示词工程的核心部分
//     const prompt = `
// 你是一个智能城市共享单车调度系统的AI专家。
// 你的任务是根据当前各站点的车辆数和对未来的预测，生成一个高效的车辆调度计划，以解决“潮汐现象”导致的部分站点车辆堆积和部分站点无车可用的问题。
//
// 【调度目标时间】: ${targetTimeISO}
//
// 【各站点数据】:
// 以下是各站点的实时数据和预测数据。
// "capacity": 站点总容量
// "current_bikes": 当前车辆数
// "predicted_bikes_at_target": 在目标时间点的预测车辆数
//
// ${formattedStationData}
//
// 【你的任务】:
// 1. 分析以上数据，找出哪些站点在目标时间点可能会出现车辆“溢出”（预测车辆数 > 容量），哪些站点可能会“枯竭”（预测车辆数 < 5 或远低于需求）。
// 2. 生成一个或多个具体的调度任务来平衡车辆分布。
// 3. 你的建议必须是可执行的，即从一个站点调出的车辆数不能超过该站点的可用车辆数，调入一个站点的车辆数不能使该站点超过其容量。
//
// 【输出格式】:
// 请严格按照以下JSON格式返回你的调度建议列表。不要包含任何JSON格式之外的解释性文字。
//
// [
//   {
//     "from_station_id": number, // 车辆调出站点的ID
//     "from_station_name": "string", // 车辆调出站点的名称
//     "to_station_id": number, // 车辆调入站点的ID
//     "to_station_name": "string", // 车辆调入站点的名称
//     "bikes_to_move": number, // 建议调度的单车数量
//     "reason": "string" // 调度的理由，例如：'缓解JVL517站点的车辆堆积风险，并补充JXL101站点的车辆需求。'
//   }
// ]
// `;
//
//     return [{ role: 'user', content: prompt }];
// }

/**
 * @route   GET /suggestions/dispatch
 * @desc    根据预测时间和站点数据，从AI获取单车调度建议
 * @access  Private (需要认证)
 */
router.get('/dispatch', authMiddleware, async (req, res) => {
    // 1. 获取并验证请求参数
    const { target_time,request } = req.query;
    if (!target_time) {
        return res.status(400).json({ error: '请求失败，缺少 "target_time" 参数。' });
    }

    try {
        // // 2. 从数据库获取所需数据
        // const stationData = await getStationDataForAI(new Date(target_time));
        // if (!stationData || stationData.length === 0) {
        //     return res.status(404).json({ message: '未找到足够的站点数据用于分析。' });
        // }
        //
        // // 3. 生成提示词 (Prompt)
        // const messages = generateSchedulingPrompt(stationData, target_time);
        const messages = [{ role: 'user', content: request }];
        // 4. 调用DeepSeek API
        // 从环境变量中获取API密钥，这是最佳实践
        const apiKey = 'sk-fa73fa4c4eaf402b9e770ee92cbc0dbf';
        if (!apiKey) {
            console.error('错误: DEEPSEEK_API_KEY 环境变量未设置。');
            return res.status(500).json({ error: '服务器配置错误。' });
        }

        const response = await axios.post(
            'https://api.deepseek.com/v1/chat/completions',
            {
                model: 'deepseek-chat',
                messages: messages,
                temperature: 0.5, // 对于需要精确结果的任务，温度可以设低一些
                response_format: { type: "json_object" } // 请求JSON格式输出，提高稳定性
            },
            {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        // 5. 将AI的响应返回给前端
        // DeepSeek的JSON模式会把结果包在 "choices"[0]."message"."content" 里，它是一个字符串
        // 我们需要先解析这个字符串再返回给前端
        const aiContent = JSON.parse(response.data.choices[0].message.content);
        res.json(aiContent);

    } catch (err) {
        // 统一的错误处理
        if (err.isAxiosError) {
            // AI API调用相关的错误
            console.error('调用 DeepSeek API 出错:', err.response?.data || err.message);
            res.status(502).json({ error: '调用AI服务失败。' }); // 502 Bad Gateway 更合适
        } else {
            // 数据库或其他内部错误
            console.error('处理调度建议请求时发生错误:', err);
            res.status(500).json({ error: '服务器内部错误。' });
        }
    }
});


module.exports = router;