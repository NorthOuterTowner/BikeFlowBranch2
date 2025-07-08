const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")
const crypto = require('crypto');
const fs = require('fs');

router.post('/login', async (req, res) => {
  let {account,password} = req.body;

  const hash = crypto.createHash('sha256');
  hash.update(password);
  const hashpwd = hash.digest('hex');

  const sql = "select * from `admin` where `account` = ? AND `password` = ?"
  let {err,rows} = await db.async.all(sql,[account,hashpwd])

  if (err == null && rows.length > 0){
    let login_account = rows[0].account
    let login_token = uuidv4();
    const set_token_sql = "update `admin` set `token` = ? where `account` = ?"
    await db.async.run(set_token_sql,[login_token,account])

    let admin_info = rows[0]
    admin_info.password = ""
    admin_info.token = login_token

    res.send({
      code:200,
      msg:"登陆成功",
      data:admin_info
    })
  }else{
      res.send({
        code:500,
        msg:"登录失败"
      })
    }
  }
);

router.post('/register', async (req, res) => {
  let {account,password} = req.body

  const hash = crypto.createHash('sha256');
  hash.update(password);
  const hashpwd = hash.digest('hex');
  
  const sql = "insert into `admin` (`account`,`password`) values (?,?)"
  let {err,rows} = await db.async.run(sql,[account,hashpwd])
  if(err == null) {
    res.send({
      code:200,
      msg:"注册成功"
    })
  }else{
    res.send({
      code:500,
      msg:"注册失败"
    })
  }
});

module.exports = router;
