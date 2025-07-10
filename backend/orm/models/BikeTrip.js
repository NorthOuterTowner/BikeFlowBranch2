const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const bikeTrip = sequelize.define('bikeTrip', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    ride_id: {
      type: DataTypes.STRING(32),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    rideable_type: {
      type: DataTypes.STRING(20),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    started_at: {
      type: DataTypes.DATE,
      allowNull: true
    },
    ended_at: {
      type: DataTypes.DATE,
      allowNull: true
    },
    start_station_name: {
      type: DataTypes.STRING(100),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    start_station_id: {
      type: DataTypes.STRING(20),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    end_station_name: {
      type: DataTypes.STRING(100),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    end_station_id: {
      type: DataTypes.STRING(20),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    },
    start_lat: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    start_lng: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    end_lat: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    end_lng: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    member_casual: {
      type: DataTypes.STRING(20),
      allowNull: true,
      collate: 'utf8mb4_general_ci'
    }
  }, {
    tableName: 'bike_trip',
    charset: 'utf8mb4',
    collate: 'utf8mb4_general_ci',
    timestamps: false
  });

  return StationHourlyFlow;
};