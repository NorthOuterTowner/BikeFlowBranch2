const express = require('express');
const router = express.Router();

router.get('/login', (req, res) => {
  res.json({ msg: 'Hello from /login' });
});

router.get('/register', (req, res) => {
  res.json({ msg: 'Hello from /register' });
});

module.exports = router;
