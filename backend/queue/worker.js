const dispatchQueue = require("./dispatchQueue");
const {db,genid} = require("../db/dbUtils");

dispatchQueue.process(async (job) => {
  const { number, endStation, dispatchDate, dispatchHour, dispatchId } = job.data;

  const changeSql2 = "UPDATE `station_real_data` SET `stock` = `stock` + ? WHERE `station_id` = ? AND `date` = ? AND `hour` = ?";
  
  try {
    await db.async.run(changeSql2, [number, endStation, dispatchDate, dispatchHour]);
  } catch (err) {
    console.error("调度任务执行失败", err);
  }
  
  if (parseInt(dispatchHour) == 23) {
    const finishStatusSql = "UPDATE `station_schedule` SET `status` = 2 WHERE `id` = ?";
    await db.async.run(finishStatusSql, [dispatchId]);
    console.log(`调度任务完成，dispatchId=${dispatchId}`);
  }

});
