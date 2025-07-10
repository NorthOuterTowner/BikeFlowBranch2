const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const Station = sequelize.define('Station', {
    station_id: {
      type: DataTypes.STRING(20),
      primaryKey: true,
      allowNull: false,
      collate: 'utf8mb4_general_ci'
    },
    lat: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    lng: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    capacity: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: 20
    }
  }, {
    tableName: 'station_info', // 这里指定实际的表名
    charset: 'utf8mb4',
    collate: 'utf8mb4_general_ci',
    timestamps: false
  });

  return Station;
};