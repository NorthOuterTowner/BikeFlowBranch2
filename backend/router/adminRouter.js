const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")
const crypto = require('crypto');
const fs = require('fs');
const nodemailer = require("nodemailer");
const redis = require("redis")
const redisClient = require("../db/redis")

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

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
  let {account,password,email} = req.body
  
  /*参数正确性 */
  if (!account || !password || !email) {
    return res.send({ code: 400, msg: "参数不能为空" });
  }
  if (!emailRegex.test(email)) {
    return res.send({ code: 400, msg: "邮箱格式不合法" });
  }

  /*避免同一账号重复注册 */
  try {
    const checkSql = "SELECT count(*) as `cnt` FROM `admin` WHERE `account` = ? OR `email` = ?";
    const { rows: exists } = await db.async.all(checkSql, [account, email]);

    if (exists[0].cnt > 0) {
      return res.send({ code: 409, msg: "账号或邮箱已被注册" });
  }}catch (e) {
    res.send({ 
      code: 500, 
      msg: "服务器内部错误", 
      error: e.message 
    });
  }
  
  //hash
  const hash = crypto.createHash('sha256');
  hash.update(password);
  const hashpwd = hash.digest('hex');
  
  // generate verify code
  const verifyCode = crypto.randomBytes(16).toString("hex");
  const expiresAt = 60 * 30; // 秒为单位

  const userData = JSON.stringify({
    account,
    password: hashpwd,
    email
  });

  // store to Redis（设置 30 分钟过期）
  await redisClient.setEx(`register:${verifyCode}`, expiresAt, userData);

  // send email from 163
  const transporter = nodemailer.createTransport({
    service: '163',
    auth: {
      user: 'lrz08302005@163.com',
      pass: 'FVGRCXYRKVQGDIEE'
    }
  });

  const verifyUrl = `http://localhost:3000/admin/verify?code=${verifyCode}`;

  await transporter.sendMail({
    from: 'lrz08302005@163.com',
    to: email,
    subject: '注册验证 - 请确认你的邮箱',
    //html: `<p>点击下面链接完成注册：</p><a href="${verifyUrl}">${verifyUrl}</a><p>30分钟内有效</p>`,
    html:`<html>
            <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; margin: 0; padding: 0;">
              <table align="center" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px;">
                <tr>
                  <td style="text-align: center; padding-bottom: 20px;">
                    <h2 style="color: #333333;">欢迎注册！</h2>
                    <p style="color: #555555; font-size: 16px; margin: 0;">请点击下面的按钮完成邮箱验证，30分钟内有效。</p>
                  </td>
                </tr>
                <tr>
                  <td style="text-align: center; padding: 20px 0;">
                    <a href="${verifyUrl}" style="background-color:rgb(76, 175, 79); color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                      验证邮箱
                    </a>
                  </td>
                </tr>
                <tr>
                  <td style="text-align: center; color: #999999; font-size: 12px; padding-top: 10px;">
                    <p>如果按钮无法点击，请复制下面链接到浏览器打开：</p>
                    <p style="word-break: break-all;">${verifyUrl}</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding-top: 30px; font-size: 12px; color: #aaaaaa; text-align: center;">
                    <p>如果你没有发起本次请求，请忽略此邮件。</p>
                    <p>© 2025 BikeFlow 团队</p>
                  </td>
                </tr>
              </table>
            </body>
          </html>`
  });

  res.send({ code: 200, msg: "验证邮件已发送，请查收邮箱" });
});

router.get('/verify', async (req, res) => {
  const { code } = req.query;

  const json = await redisClient.get(`register:${code}`);
  if (!json) {
    return res.send("链接无效或已过期！");
  }

  const { account, password, email } = JSON.parse(json);

  // insert into info of user
  await db.async.run(
    "INSERT INTO `admin` (`account`,`password`,`email`) VALUES (?,?,?)",
    [account, password, email]
  );

  await redisClient.del(`register:${code}`);

  res.send({
    code:200,
    msg:"注册成功"
  });
});


module.exports = router;
