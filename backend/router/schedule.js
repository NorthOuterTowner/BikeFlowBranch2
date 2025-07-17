const express = require('express');
const router = express.Router();
const { exec } = require('child_process');
const authMiddleware = require("../utils/auth");
const path = require('path');
const pythonExe = 'D:\\Download\\Python3.8.10\\python.exe';

// GET /api/schedule?date=2025-06-13&hour=9
router.get('/',authMiddleware, (req, res) => {
  const { date, hour } = req.query;

  if (!date || !hour) {
    return res.status(400).json({ success: false, message: '缺少 date 或 hour 参数' });
  }

  // 拼接调用命令
  const env = Object.create(process.env);
  env.PYTHONUSERBASE = process.env.PYTHONUSERBASE || `${process.env.USERPROFILE}\\AppData\\Roaming\\Python\\Python38\\site-packages`;

  const scriptPath = path.resolve(__dirname, '../../handle/newdispatch/patch.py');
  const command = `"${pythonExe}" "${scriptPath}" --date ${date} --hour ${hour}`;

  exec(command, (error, stdout, stderr) => {
    console.log('stdout:', stdout);
    console.log('stderr:', stderr);
    if (error) {
      console.error(`[调度错误]`, stderr);
      return res.status(500).json({ success: false, message: '调度执行失败', error: stderr });
    }

    console.log(`[调度成功] 输出:`, stdout);
    return res.status(200).json({ success: true, message: '调度成功', output: stdout });
  });
});

module.exports = router;
