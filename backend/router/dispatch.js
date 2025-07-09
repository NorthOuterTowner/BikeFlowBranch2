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
        const { err, rows } = await db.async.all(
            'SELECT * FROM admin WHERE account = ? AND token = ?',
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

router.post('/change', authMiddleware, async (req, res) => {
    let { startStation,endStation,number,dispatchDate, dispatchHour } = req.body
    
    const queryStartsql = "select 1 from `station_real_data` where `station_id` = ? "
    const queryEndSql = "select 1 from `station_real_data` where `station_id` = ? "
    
    let {err:startErr,rows:startRows} = await db.async.all(queryStartsql,[startStation])
    let {err:endErr,rows:endRows} = await db.async.all(queryEndSql,[endStation])

    if(startErr == null && endErr == null && startRows.length > 0 && endRows.length > 0){
        let changeableStock = 0
        const getStockSql = "select `stock` from `station_real_data` where `station_id` = ? and `date` = ? and `hour` = ? "
        let {err:searchErr,rows:searchRows} = await db.async.all(getStockSql,[startStation,dispatchDate,dispatchHour]) 
        changeableStock = searchRows[0].stock

        if(changeableStock < number){
            res.send({
                code:500,
                msg:"该调度方案不可行，调度数量超过本站点车余量"
            })
        }else{
            const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
            const changeSql2 = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
            
            await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
            await db.async.run(changeSql2,[number,endStation,dispatchDate,dispatchHour])
            
            res.send({
                code:200,
                msg:"开始进行调度"
            })
        }  
    }else{
        if(startRows.length == 0 || endRows.length == 0){
            res.send({
                code:500,
                msg:"无效站点"
            })
        }else{
            res.send({
                code:500,
                msg:"调度失败"
            })
        }
    }
});

module.exports = router;