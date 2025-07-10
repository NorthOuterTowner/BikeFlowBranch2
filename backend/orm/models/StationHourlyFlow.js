// models/StationHourlyFlow.js
const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const StationHourlyFlow = sequelize.define('StationHourlyFlow', {
    id: {
      type: DataTypes.INTEGER(11),
      primaryKey: true,
      autoIncrement: true,
      allowNull: false
    },
    station_id: {
      type: DataTypes.STRING(20),
      allowNull: false,
      collate: 'utf8mb4_general_ci'
    },
    timestamp: {
      type: DataTypes.DATE,
      allowNull: false
    },
    inflow: {
      type: DataTypes.INTEGER(11),
      allowNull: false,
      defaultValue: 0
    },
    outflow: {
      type: DataTypes.INTEGER(11),
      allowNull: false,
      defaultValue: 0
    },
    created_at: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW
    }
  }, {
    tableName: 'Station_hourly_flow',
    charset: 'utf8mb4',
    collate: 'utf8mb4_general_ci',
    timestamps: false, // 禁用Sequelize自动管理的时间戳
    underscored: true, // 使用下划线命名风格
    indexes: [
      {
        fields: ['station_id'] // 为station_id添加索引
      },
      {
        fields: ['timestamp'] // 为timestamp添加索引
      }
    ]
  });

  return StationHourlyFlow;
};