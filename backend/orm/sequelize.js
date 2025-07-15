const { Sequelize, DataTypes } = require('sequelize');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const sequelize = new Sequelize(
  process.env.DB_NAME,
  process.env.DB_USER,
  process.env.DB_PASSWORD,
  {
    host: process.env.DB_HOST,
    dialect: 'mysql',
    logging: false,
  }
);

const db = {};

db.sequelize = sequelize;
db.Sequelize = Sequelize;

const modelsDir = path.join(__dirname, 'models');
fs.readdirSync(modelsDir)
    .filter(file => file.endsWith('.js')) // 只处理 .js 文件
    .forEach(file => {
        // 引入模型定义函数
        const modelDefinition = require(path.join(modelsDir, file));
        // 调用模型定义函数，传入 sequelize 实例和 DataTypes
        const model = modelDefinition(sequelize, DataTypes);
        // 将初始化的模型以大写驼峰命名存入 db 对象 (e.g., db.StationSchedule)
        db[model.name] = model;
    });


// StationSchedule <-> Station
db.StationSchedule.belongsTo(db.Station, { foreignKey: 'start_id', as: 'startStation' });
db.StationSchedule.belongsTo(db.Station, { foreignKey: 'end_id', as: 'endStation' });
db.Station.hasMany(db.StationSchedule, { foreignKey: 'start_id', as: 'departingSchedules' });
db.Station.hasMany(db.StationSchedule, { foreignKey: 'end_id', as: 'arrivingSchedules' });

// StationHourlyFlow <-> Station
db.StationHourlyFlow.belongsTo(db.Station, { foreignKey: 'station_id', targetKey: 'station_id' });
db.Station.hasMany(db.StationHourlyFlow, { foreignKey: 'station_id', sourceKey: 'station_id' });

// StationRealData <-> Station
db.StationRealData.belongsTo(db.Station, { foreignKey: 'station_id', targetKey: 'station_id' });
db.Station.hasMany(db.StationRealData, { foreignKey: 'station_id', sourceKey: 'station_id' });

db.BikeTrip.belongsTo(db.Station, { foreignKey: 'start_station_id', targetKey: 'station_id', as: 'startStationInfo' });
db.BikeTrip.belongsTo(db.Station, { foreignKey: 'end_station_id', targetKey: 'station_id', as: 'endStationInfo' });
module.exports = sequelize;
