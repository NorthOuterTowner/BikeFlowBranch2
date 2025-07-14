const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")
const crypto = require('crypto');
const fs = require('fs');
const nodemailer = require("nodemailer");
const redis = require("redis")
const redisClient = require("../db/redis");

const sequelize = require('../orm/sequelize');
const { DataTypes } = require('sequelize')
const AdminModel = require('../orm/models/Admin');
const Admin = AdminModel(sequelize,DataTypes)
const { Op } = require('sequelize');
const { register } = require('module');

const ejs = require('ejs');
const path = require('path');
const templatePath = path.join(__dirname, '../views/emailTemplate.ejs');

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * @api {post} /api/v1/admin/login 登录账号
 * @apiName AdminLogin
 * @apiGroup Admin
 * @apiDescription 使用账号和密码登录，成功后返回 token 和账户信息，前端应保存 token 以维持登录状态。
 *
 * @apiParam {String} account 管理员账号（唯一标识）
 * @apiParam {String} password 管理员密码（将进行 SHA256 加密后校验）
 *
 * @apiSuccess {Number} code 状态码 200 表示登录成功
 * @apiSuccess {String} msg 返回信息
 * @apiSuccess {Object} data 管理员信息
 * @apiSuccess {String} data.account 管理员账号
 * @apiSuccess {String} data.token 登录 token（需前端保存）
 *
 * @apiError (Error 500) {Number} code 状态码 500 表示登录失败
 * @apiError (Error 500) {String} error 错误描述
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
      attributes: ['account', 'password','email'], 
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
    admin_info.email = admin_info.email || ""

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
 * @api {post} /api/v1/admin/register 注册账号（发送邮箱验证）
 * @apiName AdminRegister
 * @apiGroup Admin
 * @apiDescription 注册管理员账号。前端提交账号、密码和邮箱后，系统将发送一封验证邮件。用户需在30分钟内点击验证链接以完成注册。
 *
 * @apiParam {String} account 管理员账号（唯一）
 * @apiParam {String} password 管理员密码（将进行 SHA256 加密）
 * @apiParam {String} email 管理员邮箱地址（用于接收验证邮件）
 *
 * @apiSuccess {Number} code 状态码 200 表示邮件发送成功
 * @apiSuccess {String} msg 返回信息
 *
 * @apiError (Error 400) {String} error 参数缺失或邮箱格式错误
 * @apiError (Error 409) {String} error 账号或邮箱已被注册
 * @apiError (Error 500) {String} error 系统内部错误
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
  }else if (!emailRegex.test(email)) {
    return res.status(400).send({  
      error: "邮箱格式不合法" 
    });
  }else{

  /*避免同一账号重复注册 */
  try {
    console.log("right1")
    const count = await Admin.count({
      where: {
        [Op.or]: [
          { account: account },
          { email: email }
        ]
      }
    });
    if (count > 0) {
      return res.status(409).send({ 
        error: "账号或邮箱已被注册" 
      });
    }else{
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

      const htmlContent = await ejs.renderFile(templatePath, { verifyUrl });
      await transporter.sendMail({
        from: 'lrz08302005@163.com',
        to: email,
        subject: '注册验证 - 请确认你的邮箱',
        html:htmlContent
      });
      res.status(200).send({
        code:200,
        msg:"已发送邮件"
      })
    }
  }catch (e) {
    console.log(e)
      res.status(500).send({ 
        code:500,
        error: "服务器内部错误"
      });
  }}
});

/**
 * @api {get} /api/v1/admin/verify 邮箱验证链接跳转
 * @apiName EmailVerify
 * @apiGroup Admin
 * @apiDescription 用户点击验证链接后触发此接口。若验证码有效，将正式创建账号，并返回一个 HTML 页面提示注册成功。
 *
 * @apiParam {String} code 邮件中附带的验证码（作为 URL 参数传递）
 *
 * @apiSuccessExample {html} 成功页面
 *     HTTP/1.1 200 OK
 *     <!DOCTYPE html>
 *     <html> ... 注册成功的 HTML 页面 ... </html>
 *
 * @apiErrorExample {html} 验证失败页面
 *     HTTP/1.1 200 OK
 *     <!DOCTYPE html>
 *     <html> ... 链接无效或已过期的提示页面 ... </html>
 */

router.get('/verify', async (req, res) => {
  const { code } = req.query;

  const json = await redisClient.get(`register:${code}`);
  if (!json) {
    return res.status(500).render('registerFailed');
  }else{
    const { account, password, email } = JSON.parse(json);

    // insert into info of user
    await db.async.run(
      "INSERT INTO `admin` (`account`,`password`,`email`) VALUES (?,?,?)",
      [account, password, email]
    );

    await redisClient.del(`register:${code}`);

    res.status(200).render('registerSuccess',{ account, email });
  }


});


module.exports = router;
