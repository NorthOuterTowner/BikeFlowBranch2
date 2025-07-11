const Bull = require("bull");

require("dotenv").config();

const dispatchQueue = new Bull("dispatch-queue", {
  redis: {
    host: process.env.REDIS_HOST,
    port: process.env.REDIS_PORT
  }
});

module.exports = dispatchQueue;
