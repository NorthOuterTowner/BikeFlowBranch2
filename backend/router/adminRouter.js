const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")

router.post('/login', async (req, res) => {
  let {account,password} = req.body;
  console.log(account)
  console.log(password)
  const sql = "select * from `admin` where `account` = ? AND `password` = ?"
  let {err,rows} = await db.async.all(sql,[account,password])

  if (err == null && rows.length > 0){
    let login_account = rows[0].account
    let login_token = uuidv4();
    const set_token_sql = "update `admin` set `token` = ? where `account` = ?"
    await db.async.run(set_token_sql,[login_token,account])

    let admin_info = rows[0]
    admin_info.token = login_token

    res.send({
      code:200,
      msg:"登陆成功",
      data:admin_info
    })
  }else{
      res.send({
        res:500,
        msg:"登录失败"
      })
    }
  }
);

router.get('/register', async (req, res) => {
  res.json({ msg: 'Hello from /register' });
});

module.exports = router;
