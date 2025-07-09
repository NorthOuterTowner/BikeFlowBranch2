const express = require('express');
const {rateLimit} = require('express-rate-limit')
const app = express();
const PORT = 3000;
const {db,genid} = require("./db/dbUtils")
const redis = require("redis")
const redisClient = require("./db/redis")

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
    res.header("Access-Control-Allow-Origin","*");
    res.header("Access-Control-Allow-Headers","*");
    res.header("Access-Control-Allow-Methods","DELETE,PUT,POST,GET,OPTIONS");
    if(req.method == "OPTIONS") res.sendStatus(200);
    else next();
});

app.use(express.json());
app.use(limiter);

app.use("/test",require("./router/testRouter"));
app.use("/admin",require("./router/adminRouter"));
app.use("/reset",require("./router/resetRouter"));
app.use("/stations",require("./router/stationsRouter"));
app.use("/predict",require("./router/predictRouter"));
app.use("/dispatch",require("./router/dispatch"));

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
