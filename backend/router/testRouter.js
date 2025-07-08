const express = require('express');
const { genid } = require('../db/dbUtils');
const router = express.Router();

router.get('/', (req, res) => {
  res.send({
    status:200,
    id:genid.NextId()
  })
});

module.exports = router;
