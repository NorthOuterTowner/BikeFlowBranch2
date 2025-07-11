const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")
const crypto = require('crypto');
const fs = require('fs');
const nodemailer = require("nodemailer");
const redis = require("redis")
const redisClient = require("../db/redis");

const sequelize = require('../orm/sequelize'); // 确保路径正确
const { DataTypes } = require('sequelize')
const AdminModel = require('../orm/models/Admin');
const Admin = AdminModel(sequelize,DataTypes)

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * 登录功能：
 * 输入用户名和密码即可，该接口返回token，前端应使用工具记录以验证登录状态
 */
router.post('/login', async (req, res) => {
  let {account,password} = req.body;

  const hash = crypto.createHash('sha256');
  hash.update(password);
  const hashpwd = hash.digest('hex');
  
  //ORM
  let AdminContent;
  try{
    AdminContent = await Admin.findOne({
      attributes: ['account', 'password'], 
      where:{
        account:account,
        password:hashpwd
      },
      raw:true
    });
  }catch(err){
    console.log("AdminContent Wrong")
  }

  if (AdminContent!=null/*err == null && rows.length > 0*/){
    let login_account = AdminContent.account
    let login_token = uuidv4();
    console.log("3")
    try{
      await Admin.update(
        { token: login_token },                // 要更新的字段
        { where: { account:login_account } }    // 条件
      );
    }catch(e){
      console.log(e)
    }

    let admin_info = AdminContent
    admin_info.password = ""
    admin_info.token = login_token

    res.status(200).send({
      code:200,
      msg:"登陆成功",
      data:admin_info
    })
  }else{
      res.status(500).send({
        code:500,
        error:"登录失败"
      })
    }
  }
);

/**
 * 注册功能：
 * 需在进行注册后点击邮箱链接才可完成注册
 */
router.post('/register', async (req, res) => {
  //
  let {account,password,email} = req.body
  //
  /*参数正确性 */
  if (!account || !password || !email) {
    return res.status(400).send({ 
      error: "参数不能为空" 
    });
  }
  if (!emailRegex.test(email)) {
    return res.status(400).send({  
      error: "邮箱格式不合法" 
    });
  }

  /*避免同一账号重复注册 */
  try {
    const count = await Admin.count({
      where: {
        [Op.or]: [
          { account: account },
          { email: email }
        ]
      }
    });
    //const checkSql = "SELECT count(*) as `cnt` FROM `admin` WHERE `account` = ? OR `email` = ?";
    //const { rows: exists } = await db.async.all(checkSql, [account, email]);

    if (count > 0) {
      return res.status(409).send({ 
        error: "账号或邮箱已被注册" 
      });
  }}catch (e) {
    res.status(500).send({ 
      code:500,
      error: "服务器内部错误"
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

  res.status(200).send({ 
    code:200,
    error: "验证邮件已发送，请查收邮箱" 
  });
});

/**
 * 验证点击的链接所含的验证码是否有效
 * （该API不在前端进行调用，而是通过邮件中链接跳转访问）
 */
router.get('/verify', async (req, res) => {
  const { code } = req.query;

  const json = await redisClient.get(`register:${code}`);
  if (!json) {
    return res.send(`
      <!DOCTYPE html>
      <html lang="zh-CN">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>注册验证失败</title>
          <style>
              body {
                  font-family: 'Arial', sans-serif;
                  background-color: #f5f5f5;
                  display: flex;
                  justify-content: center;
                  align-items: center;
                  height: 100vh;
                  margin: 0;
              }
              .container {
                  background: white;
                  padding: 2rem;
                  border-radius: 8px;
                  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                  text-align: center;
                  max-width: 500px;
              }
              h1 {
                  color: #e74c3c;
              }
              p {
                  color: #555;
                  margin-bottom: 1.5rem;
              }
              .icon {
                  font-size: 3rem;
                  color: #e74c3c;
                  margin-bottom: 1rem;
              }
          </style>
      </head>
      <body>
          <div class="container">
              <div class="icon">❌</div>
              <h1>注册验证失败</h1>
              <p>链接无效或已过期！</p>
              <p>请重新申请注册链接或联系管理员。</p>
          </div>
      </body>
      </html>
    `);
  }

  const { account, password, email } = JSON.parse(json);

  // insert into info of user
  await Admin.create({
    account: account,
    password: password,
    email: email
  });
  /*await db.async.run(
    "INSERT INTO `admin` (`account`,`password`,`email`) VALUES (?,?,?)",
    [account, password, email]
  );*/

  await redisClient.del(`register:${code}`);

  res.send(`
  <!DOCTYPE html>
  <html lang="zh-CN">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>注册成功</title>
      <style>
          body {
              font-family: 'Arial', sans-serif;
              background-color: #f5f5f5;
              display: flex;
              justify-content: center;
              align-items: center;
              height: 100vh;
              margin: 0;
          }
          .container {
              background: white;
              padding: 2rem;
              border-radius: 8px;
              box-shadow: 0 2px 10px rgba(0,0,0,0.1);
              text-align: center;
              max-width: 500px;
          }
          h1 {
              color: #2ecc71;
          }
          p {
              color: #555;
              margin-bottom: 1.5rem;
          }
          .icon {
              font-size: 3rem;
              color: #2ecc71;
              margin-bottom: 1rem;
          }
          .account-info {
              background: #f9f9f9;
              padding: 1rem;
              border-radius: 4px;
              margin: 1rem 0;
              text-align: left;
          }
      </style>
  </head>
  <body>
      <div class="container">
          <div class="icon">✓</div>
          <h1>注册成功</h1>
          <p>您的账户已成功创建！</p>
          
          <div class="account-info">
              <p><strong>账号:</strong> ${account}</p>
              <p><strong>邮箱:</strong> ${email}</p>
          </div>
          
          <p>您现在可以登录您的账户。</p>
      </div>
  </body>
  </html>
`);
});


module.exports = router;
