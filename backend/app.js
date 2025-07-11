const express = require('express');
const {rateLimit} = require('express-rate-limit')
const app = express();
const PORT = 3000;
const {db,genid} = require("./db/dbUtils")
const redis = require("redis")
const redisClient = require("./db/redis")
const sequelize = require('./orm/sequelize');

/* 🌟 全局打印收到的所有请求 */
app.use((req, res, next) => {
  console.log(`请求路径: ${req.method} ${req.originalUrl}`)
  next()
})



/* API rate limit */
const limiter = rateLimit({
	windowMs: 1000, // 1 second
	limit: 10, // Limit each IP to 10 requests per `window` (here, per 1 second).
	standardHeaders: 'draft-8', // draft-6: `RateLimit-*` headers; draft-7 & draft-8: combined `RateLimit` header
	legacyHeaders: false, // Disable the `X-RateLimit-*` headers.
	// store: ... , // Redis, Memcached, etc. See below.
})

/* Cross-Origin Requests */
app.use(function(req,res,next){
    const allowedOrigin = process.env.CORS_ORIGIN;
    res.header("Access-Control-Allow-Origin", allowedOrigin);
    res.header("Access-Control-Allow-Headers","*");
    res.header("Access-Control-Allow-Methods","DELETE,PUT,POST,GET,OPTIONS");
    res.header("Access-Control-Allow-Credentials", "true");
    if(req.method == "OPTIONS") res.sendStatus(200);
    else next();
});

app.use(express.json());
app.use(limiter);

sequelize.authenticate().then(() => {
  console.log('Sequelize 已成功连接数据库');
}).catch(err => {
  console.error('连接失败:', err);
});

app.use("/admin",require("./router/adminRouter"));
app.use("/reset",require("./router/resetRouter"));
app.use("/stations",require("./router/stationsRouter"));
app.use("/predict",require("./router/predictRouter"));
app.use("/dispatch",require("./router/dispatch"));
app.use("/schedule", require("./router/schedule"));


app.listen(PORT,'0.0.0.0', () => {
  console.log(`Server is running on port ${PORT}`);
});
