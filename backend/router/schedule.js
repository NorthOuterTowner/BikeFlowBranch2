const express = require('express');
const router = express.Router();
const { exec } = require('child_process');

// GET /api/schedule?date=2025-06-13&hour=9
router.get('/', (req, res) => {
  const { date, hour } = req.query;

  if (!date || !hour) {
    return res.status(400).json({ success: false, message: '缺少 date 或 hour 参数' });
  }

  // 拼接调用命令
  const command = `python D:/myGithub/newBikeFlow/handle/newdispatch/patch.py --date ${date} --hour ${hour}`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`[调度错误]`, stderr);
      return res.status(500).json({ success: false, message: '调度执行失败', error: stderr });
    }

    console.log(`[调度成功] 输出:`, stdout);
    return res.json({ success: true, message: '调度成功', output: stdout });
  });
});

module.exports = router;
