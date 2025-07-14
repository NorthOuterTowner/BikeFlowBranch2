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

  }else if (type=="cancelStart"){
    const { number, startStation,endStation, dispatchDate, dispatchHour, dispatchId } = job.data;

    let dispatchHourInt = parseInt(dispatchHour)
    while(dispatchHourInt <= 23){
      await cancelSchedule(number,startStation,dispatchDate,dispatchHourInt);
      dispatchHourInt+=1;
    }
    console.log("setStatus")
    const statusFinish = "update `station_schedule` set `status` = 0  where `id` = ? ;"
    await db.async.run(statusFinish,dispatchId)
  }else if (type == "cancelEnd"){
    const { number, startStation,endStation, dispatchDate, dispatchHour, dispatchId } = job.data;

    let dispatchHourInt = parseInt(dispatchHour)
    while(dispatchHourInt <= 23){
      await cancelSchedule2(number,endStation,dispatchDate,dispatchHourInt);
      dispatchHourInt+=1;
    }
  }


});

/**
 * Increase the bike stock at the destination station after a specified delay during scheduling 
 * @param {*} number 
 * @param {*} endStation 
 * @param {*} dispatchDate 
 * @param {*} dispatchHour 
 */
async function afterTimeSchedule2(number,endStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  try{
    await db.async.run(changeSql,[number,endStation,dispatchDate,dispatchHour])
  }catch(err){
    console.log(err)
  }
}

/**
 * Increase the bike stock at the origin station station after a specified delay when scheduling canceled
 * @param {*} number 
 * @param {*} startStation 
 * @param {*} dispatchDate 
 * @param {*} dispatchHour 
 */
async function cancelSchedule(number,startStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` + ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  try{
    await db.async.run(changeSql,[number,startStation,dispatchDate,dispatchHour])
  }catch(err){
    console.log(err)
  }
}

/**
 * Decrease the bike stock at the destination station after a specified delay when scheduling canceled
 * @param {*} number 
 * @param {*} endStation 
 * @param {*} dispatchDate 
 * @param {*} dispatchHour 
 */
async function cancelSchedule2(number,endStation,dispatchDate,dispatchHour) {
  const changeSql = "update `station_real_data` set `stock` = `stock` - ? where `station_id` = ? and `date` = ? and `hour` = ?;"
  try{
    await db.async.run(changeSql,[number,endStation,dispatchDate,dispatchHour])
  }catch(err){
    console.log(err)
  }
}