// db/redis.js
const redis = require("redis");

const redisClient = redis.createClient({
  socket: {
    host: "127.0.0.1",
    port: 6379
  }
});

redisClient.on("error", (err) => {
  console.error("Redis 连接失败：", err);
});

redisClient.on("connect", () => {
  console.log("Redis 连接成功");
});

redisClient.connect(); // v4 必须显式调用

module.exports = redisClient;
