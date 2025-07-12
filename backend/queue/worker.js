const dispatchQueue = require("./dispatchQueue");
const {db,genid} = require("../db/dbUtils");

dispatchQueue.process(async (job) => {
  console.log(job.data)
  const {type} = job.data;
  if(type == "dispatch"){
    const { number, startStation,endStation, dispatchDate, dispatchHour, dispatchId } = job.data;

    let dispatchHourInt = parseInt(dispatchHour)

    while(dispatchHourInt <= 23){
      await afterTimeSchedule2(number,endStation,dispatchDate,dispatchHourInt);
      dispatchHourInt+=1;
    }
    const statusFinish = "update `station_schedule` set `status` = 2  where `id` = ? ;"
    await db.async.run(statusFinish,dispatchId)

  }else if (type=="cancel"){
    const { number, startStation,endStation, dispatchDate, dispatchHour, dispatchId } = job.data;

    let dispatchHourInt = parseInt(dispatchHour)
    while(dispatchHourInt <= 23){
      await cancelSchedule(number,startStation,dispatchDate,dispatchHourInt);
      await cancelSchedule2(number,endStation,dispatchDate,dispatchHourInt);
      dispatchHourInt+=1;
    }

    const statusFinish = "update `station_schedule` set `status` = 0  where `id` = ? ;"
    await db.async.run(statusFinish,dispatchId)
  }


});

async function afterTimeSchedule2(number,endStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  await db.async.run(changeSql,[number,endStation,dispatchDate,dispatchHour])
}

async function cancelSchedule(number,startStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
}

async function cancelSchedule2(number,endStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  await db.async.run(changeSql,[number,endStation,dispatchDate,dispatchHour])
}