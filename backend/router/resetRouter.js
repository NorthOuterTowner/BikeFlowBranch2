const express = require('express');
const router = express.Router();
const {db,genid} = require("../db/dbUtils")
const {v4:uuidv4} = require("uuid")
const crypto = require('crypto');
const fs = require('fs');
const nodemailer = require("nodemailer");
const redis = require("redis")
const redisClient = require("../db/redis")
const ejs = require('ejs');
const path = require('path');

const authMiddleware = require("../utils/auth")

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * 重置用户名
 */
router.post('/account',authMiddleware,async (req, res) => {
    let {oldName,newName} = req.body
    const sql = 'select count(*) as cnt from `admin` where `account` = ?'
    let { err, rows } = await db.async.all(sql,[newName])
    if(err == null && rows[0].cnt == 0){
        const sql2 = 'update `admin` set `account` = ? where `account` = ?'
        await db.async.run(sql2,[newName,oldName])
        return res.status(200).send({
            status:200,
            newName,
            msg:"用户名重置成功"
        })
    }else if (rows[0].cnt > 0){
        //用户名已存在
        return res.status(500).send({
            status:500,
            newName,
            msg:"用户名已存在"
        })
    }else{
        return res.status(500).send({
            status:500,
            newName,
            msg:"服务器错误"
        })
    }
});

/**
 * @api {post} /pwd 重置密码
 * @apiDescription 通过邮箱发送重置密码验证码，验证逻辑与注册过程相同。
 * @apiHeader {String} Authorization 用户登录令牌，需通过authMiddleware验证。
 * 
 * @apiParam {String} email 用户注册邮箱。
 * @apiParam {String} newPassword 新密码（明文，服务端将进行哈希处理）。
 *
 * @apiSuccess {Number} code 200
 * @apiSuccess {String} msg "重置验证邮件已发送，请查收邮箱"
 *
 * @apiError (400) {Number} code 400
 * @apiError (400) {String} msg "邮箱和新密码不能为空" 或 "邮箱格式不合法"
 *
 * @apiError (404) {Number} code 404
 * @apiError (404) {String} msg "该邮箱未注册"
 *
 * @apiError (500) {Number} code 500
 * @apiError (500) {String} msg "服务器内部错误"
 * @apiError (500) {String} error 具体错误信息
 */
router.post('/pwd', authMiddleware, async (req, res) => {
  const { email, newPassword } = req.body;

  if (!email || !newPassword) {
    return res.send({ code: 400, msg: "邮箱和新密码不能为空" });
  }
  if (!emailRegex.test(email)) {
    return res.send({ code: 400, msg: "邮箱格式不合法" });
  }

  try {
    // 检查邮箱是否注册
    const checkSql = "SELECT count(*) as `cnt` FROM `admin` WHERE `email` = ?";
    const { rows: exists } = await db.async.all(checkSql, [email]);

    if (exists[0].cnt === 0) {
      return res.status(404).send({ code: 404, msg: "该邮箱未注册" });
    }

    // 生成验证码和哈希密码
    const verifyCode = crypto.randomBytes(16).toString("hex");
    const expiresAt = 60 * 30; // 30分钟有效期
    
    const hash = crypto.createHash('sha256');
    hash.update(newPassword);
    const hashedPassword = hash.digest('hex');

    // 存储到Redis（验证码+新密码）
    await redisClient.setEx(
      `reset-password:${verifyCode}`, 
      expiresAt, 
      JSON.stringify({ email, newPassword: hashedPassword })
    );

    // 发送邮件
    const transporter = nodemailer.createTransport({
      service: '163',
      auth: {
        user: 'lrz08302005@163.com',
        pass: 'FVGRCXYRKVQGDIEE'
      }
    });

    const resetUrl = `http://localhost:3000/reset/verify?code=${verifyCode}`;
    
    const templatePath = path.join(__dirname, '../views/email/resetPasswordEmail.ejs');
    const htmlContent = await ejs.renderFile(templatePath, { resetUrl });

    await transporter.sendMail({
      from: 'lrz08302005@163.com',
      to: email,
      subject: '密码重置确认',
      html: htmlContent
    });

    return res.status(200).send({ 
      code: 200, 
      msg: "重置验证邮件已发送，请查收邮箱" 
    });

  } catch (e) {
    return res.status(500).send({ 
      code: 500, 
      msg: "服务器内部错误", 
      error: e.message 
    });
  }
});

/**
 * @api {get} /verify 验证重置密码链接
 * @apiDescription 用户点击邮件中的链接完成密码重置，链接有效期30分钟。
 *
 * @apiParam {String} code 邮件中的验证码。
 *
 * @apiSuccess {Number} code 200
 * @apiSuccess {String} msg "密码重置成功"
 *
 * @apiError (400/404) {String} msg "链接无效或已过期！"
 *
 * @apiError (500) {Number} code 500
 * @apiError (500) {String} msg "重置失败"
 * @apiError (500) {String} error 具体错误信息
 */
router.get('/verify', async (req, res) => {
  const { code } = req.query;

  try {
    const data = await redisClient.get(`reset-password:${code}`);
    if (!data) {
      return res.send("链接无效或已过期！");
    }

    const { email, newPassword } = JSON.parse(data);

    await db.async.run(
      "UPDATE `admin` SET `password` = ? WHERE `email` = ?",
      [newPassword, email]
    );

    await redisClient.del(`reset-password:${code}`);

    return res.status(200).send({
      code: 200,
      msg: "密码重置成功"
    });

  } catch (e) {
    return res.status(500).send({ 
      code: 500, 
      msg: "重置失败", 
      error: e.message 
    });
  }
});

module.exports = router;
