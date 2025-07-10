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

router.post('/account',async (req, res) => {
    let {oldName,newName} = req.body
    const sql = 'select count(*) as cnt from `admin` where `account` = ?'
    let { err, rows } = await db.async.all(sql,[newName])
    if(err == null && rows[0].cnt == 0){
        const sql2 = 'update `admin` set `account` = ? where `account` = ?'
        await db.async.run(sql2,[newName,oldName])
        res.status(200).send({
            status:200,
            newName,
            msg:"用户名重置成功"
        })
    }else if (rows[0].cnt > 0){
        //用户名已存在
        res.status(500).send({
            status:500,
            newName,
            msg:"用户名已存在"
        })
    }else{
        res.status(500).send({
            status:500,
            newName,
            msg:"服务器错误"
        })
    }
});

router.post('/pwd', async (req, res) => {
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

    await transporter.sendMail({
      from: 'lrz08302005@163.com',
      to: email,
      subject: '密码重置确认',
      html: `<html>
              <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; margin: 0; padding: 0;">
                <table align="center" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px;">
                  <tr>
                    <td style="text-align: center; padding-bottom: 20px;">
                      <h2 style="color: #333333;">密码重置请求</h2>
                      <p style="color: #555555; font-size: 16px; margin: 0;">请点击下方按钮确认重置密码，30分钟内有效。</p>
                    </td>
                  </tr>
                  <tr>
                    <td style="text-align: center; padding: 20px 0;">
                      <a href="${resetUrl}" style="background-color: rgb(76, 175, 79); color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                        确认重置密码
                      </a>
                    </td>
                  </tr>
                  <tr>
                    <td style="text-align: center; color: #999999; font-size: 12px; padding-top: 10px;">
                      <p>如果按钮无法点击，请复制下面链接到浏览器打开：</p>
                      <p style="word-break: break-all;">${resetUrl}</p>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding-top: 30px; font-size: 12px; color: #aaaaaa; text-align: center;">
                      <p>如非本人操作，请立即修改账户密码！</p>
                      <p>© 2025 BikeFlow 团队</p>
                    </td>
                  </tr>
                </table>
              </body>
            </html>`
    });

    res.status(200).send({ 
      code: 200, 
      msg: "重置验证邮件已发送，请查收邮箱" 
    });

  } catch (e) {
    res.status(500).send({ 
      code: 500, 
      msg: "服务器内部错误", 
      error: e.message 
    });
  }
});

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

    res.status(200).send({
      code: 200,
      msg: "密码重置成功"
    });

  } catch (e) {
    res.status(500).send({ 
      code: 500, 
      msg: "重置失败", 
      error: e.message 
    });
  }
});

module.exports = router;
