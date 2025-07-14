const express = require('express');
const axios = require('axios');
const router = express.Router();

const ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImE0ZjM4NDNiZmE3NDQ0YTM4MmNhNmEyMWM4NWUxYjU0IiwiaCI6Im11cm11cjY0In0=';
const ORS_BASE_URL = 'https://api.openrouteservice.org/v2';

router.post('/route', async (req, res) => {
    const { startCoord, endCoord } = req.body;

    // 基本的输入验证
    if (!startCoord || !endCoord) {
        return res.status(400).json({ error: 'Missing start or end coordinates' });
    }

    try {
        const orsResponse = await axios.post(
            `${ORS_BASE_URL}/directions/driving-car/geojson`,
            {
                coordinates: [startCoord, endCoord],
                format: 'geojson',
                instructions: true,
                language: 'zh-cn',
            },
            {
                headers: {
                    'Accept': 'application/json, application/geo+json',
                    'Authorization': ORS_API_KEY,
                    'Content-Type': 'application/json; charset=utf-8',
                },
            }
        );

        // 将从ORS获取的数据直接返回给前端
        res.json(orsResponse.data);

    } catch (error) {
        console.error('ORS获取路线失败:', error.response ? error.response.data : error.message);
        res.status(500).json({ error: '获取路线失败' });
    }
});

module.exports = router;