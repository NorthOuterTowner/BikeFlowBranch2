const express = require('express');
const router = express.Router();
// 引入你的原生 SQL 查询工具，这行是正确的
const { db } = require('../db/dbUtils');
const axios = require('axios');
const authMiddleware = require("../utils/auth");

/**
 * [已重构] 使用原生 SQL 聚合为 AI 提供决策所需的所有数据
 * @param {Date} targetTime - 目标时间
 * @returns {Promise<Object>} 包含站点数据和现有调度计划的对象
 */
async function getComprehensiveDataForAI(targetTime) {
    const dateForQuery = targetTime.toISOString().slice(0, 10);
    const hourForQuery = Math.floor(targetTime.getUTCHours() / 3) * 3;

    // 1. 构建聚合查询 SQL
    const stationDataSQL = `
        SELECT
            s.station_id,
            s.station_name,
            s.capacity,
            r.stock AS real_stock,
            h.inflow AS predicted_inflow,
            h.outflow AS predicted_outflow
        FROM
            station_info s
        LEFT JOIN
            station_real_data r ON s.station_id = r.station_id
                               AND r.date = ?
                               AND r.hour = ?
        LEFT JOIN
            station_hourly_status h ON s.station_id = h.station_id
                                    AND h.date = ?
                                    AND h.hour = ?
    `;

    // 2. 查询当前调度计划的 SQL
    const scheduleSQL = `
        SELECT
            start_id,
            end_id,
            bikes
        FROM
            station_schedule
        WHERE
            date = ? AND hour = ?
    `;

    // 3. 【核心修改】使用 db.async.all 并行执行查询
    const [stationDataResponse, scheduleDataResponse] = await Promise.all([
        db.async.all(stationDataSQL, [dateForQuery, hourForQuery, dateForQuery, hourForQuery]),
        db.async.all(scheduleSQL, [dateForQuery, hourForQuery])
    ]);

    // 从返回的对象中提取 rows 数组
    const stationDataResult = stationDataResponse.rows;
    const scheduleData = scheduleDataResponse.rows;

    // 4. 在 Node.js 中处理数据（这部分逻辑不变）
    const scheduleChangeMap = new Map();
    scheduleData.forEach(schedule => {
        scheduleChangeMap.set(schedule.start_id, (scheduleChangeMap.get(schedule.start_id) || 0) - schedule.bikes);
        scheduleChangeMap.set(schedule.end_id, (scheduleChangeMap.get(schedule.end_id) || 0) + schedule.bikes);
    });

    const comprehensiveStationData = stationDataResult.map(station => {
        const scheduleChange = scheduleChangeMap.get(station.station_id) || 0;

        const real_stock = station.real_stock;
        const predicted_inflow = station.predicted_inflow || 0;
        const predicted_outflow = station.predicted_outflow || 0;

        let stock_after_schedule = null;
        if (real_stock !== null && real_stock !== undefined) {
            stock_after_schedule = real_stock + predicted_inflow - predicted_outflow + scheduleChange;
        }

        return {
            ...station,
            stock_after_schedule: stock_after_schedule
        };
    });

    return {
        stationData: comprehensiveStationData,
        existingSchedule: scheduleData
    };
}


// generateOptimizationPrompt 函数保持不变
function generateOptimizationPrompt(stationData, existingSchedule, userGuidance, targetTimeISO) {
    // ... (此函数无需修改)
    const formattedStationData = JSON.stringify(stationData, null, 2);
    const formattedExistingSchedule = JSON.stringify(existingSchedule, null, 2);

    const userGuidanceSection = userGuidance
        ? `
【用户特别指令】
请在优化时重点考虑以下用户提出的要求：
"${userGuidance}"
`
        : '';

    const prompt = `
你是一位顶级的共享单车调度优化专家。

【分析背景】
- 目标调度时间: ${targetTimeISO}
- 我已经为你预先计算好了在执行“现有调度计划”后，每个站点的预期车辆数。
${userGuidanceSection}
【预计算后的站点状态】
这是关键决策数据。
- "capacity": 站点总容量
- "stock_after_schedule": 在目标时间点，执行完现有调度计划后的最终预测车辆数。

${formattedStationData}

【供参考的现有调度计划】
这是我们预计算时使用的原始计划。
${formattedExistingSchedule}

【你的核心任务】
1. 首先，理解并尊重【用户特别指令】（如果提供的话）。
2. 然后，分析【预计算后的站点状态】，找出哪些站点的 "stock_after_schedule" 存在问题（例如，远超容量导致溢出，或低于5辆导致枯竭）。
3. 结合以上所有信息，给出一个你认为**最终、最优**的调度计划来解决这些问题。

【输出格式】
请严格按照以下JSON格式返回你的优化建议。最终的JSON对象必须只包含一个名为 "optimized_plan" 的键，其值为一个调度任务数组。如果认为无需调度，可以返回空数组。不要添加任何额外的解释性文字。

{
  "optimized_plan": [
    {
      "from_station_id": "string",
      "to_station_id": "string",
      "bikes_to_move": number,
      "reason": "string"
    }
  ]
}
`;
    return [{ role: 'user', content: prompt }];
}


router.post('/dispatch', authMiddleware, async (req, res) => {
    const { target_time, user_guidance } = req.body;
    if (!target_time) {
        return res.status(400).json({ error: '请求失败，缺少 "target_time" 参数。' });
    }

    try {
        const { stationData, existingSchedule } = await getComprehensiveDataForAI(new Date(target_time));
        if (!stationData || stationData.length === 0) {
            return res.status(404).json({ message: '未找到足够的站点数据用于分析。' });
        }

        const messages = generateOptimizationPrompt(stationData, existingSchedule, user_guidance, target_time);

        const apiKey = 'sk-fa73fa4c4eaf402b9e770ee92cbc0dbf'; // 建议从环境变量读取 process.env.DEEPSEEK_API_KEY
        if (!apiKey) {
            console.error('错误: DEEPSEEK_API_KEY 环境变量未设置。');
            return res.status(500).json({ error: '服务器配置错误。' });
        }

        const response = await axios.post(
            'https://api.deepseek.com/v1/chat/completions',
            {
                model: 'deepseek-chat',
                messages: messages,
                temperature: 0.5,
                response_format: { type: "json_object" }
            },
            {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        const aiContent = JSON.parse(response.data.choices[0].message.content);

        const finalResponse = {
            schedule_time: target_time,
            optimized_plan: aiContent.optimized_plan || []
        };

        res.json(finalResponse);

    } catch (err) {
        if (err.isAxiosError) {
            console.error('调用 DeepSeek API 出错:', err.response?.data || err.message);
            res.status(502).json({ error: '调用AI服务失败。' });
        } else {
            console.error('处理调度建议请求时发生错误:', err);
            res.status(500).json({ error: '服务器内部错误。' });
        }
    }
});

/**
 * @api {post} /api/v1/suggestions 与AI对话获取建议
 * @apiDescription 提供一个通用的对话接口，前端可以发送任何与系统相关的问题，
 *                 由AI提供分析和建议。
 * @access Private
 */
router.post('/', authMiddleware, async (req, res) => {
    // 1. 从请求体中获取用户的提问
    const { message } = req.body;

    // 2. 参数校验
    if (!message || typeof message !== 'string' || message.trim() === '') {
        return res.status(400).json({ error: '请求失败，"message" 参数不能为空。' });
    }

    try {
        // 3. 为AI设定角色，以获取更相关的回答 (System Prompt)
        const messages = [
            {
                role: 'system',
                content: '你是一个专业的共享单车运营分析师和调度系统助手。请根据用户的问题，提供专业、具体、可行的分析和建议。'
            },
            {
                role: 'user',
                content: message
            }
        ];

        // 4. 从环境变量中获取API密钥
        const apiKey = process.env.DEEPSEEK_API_KEY || 'sk-fa73fa4c4eaf402b9e770ee92cbc0dbf'; // 请替换或使用环境变量
        if (!apiKey) {
            console.error('错误: DEEPSEEK_API_KEY 环境变量未设置。');
            return res.status(500).json({ error: '服务器配置错误。' });
        }

        // 5. 发起对 DeepSeek API 的请求
        const response = await axios.post(
            'https://api.deepseek.com/v1/chat/completions',
            {
                model: 'deepseek-chat',
                messages: messages,
                temperature: 0.7, // 中等温度，平衡准确性与创造性
            },
            {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        // 6. 提取并返回 AI 的回复
        const aiReply = response.data.choices[0].message.content;

        res.json({
            original_prompt: message,
            suggestion: aiReply
        });

    } catch (err) {
        // 统一的错误处理
        if (err.isAxiosError) {
            console.error('调用 DeepSeek API 出错:', err.response?.data || err.message);
            res.status(502).json({ error: '调用AI服务失败。' });
        } else {
            console.error('处理建议请求时发生错误:', err);
            res.status(500).json({ error: '服务器内部错误。' });
        }
    }
});

module.exports = router;