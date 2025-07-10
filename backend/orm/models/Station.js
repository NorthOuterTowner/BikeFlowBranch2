module.exports = (sequelize, DataTypes) => {
  const Station = sequelize.define('Station', {
    station_id: {
      type: DataTypes.STRING(20),
      primaryKey: true,
      allowNull: false
    },
    lat: DataTypes.FLOAT,
    lng: DataTypes.FLOAT,
    capacity: {
      type: DataTypes.INTEGER,
      defaultValue: 20
    }
  }, {
    tableName: 'station_info',
    freezeTableName: true
  });

  return Station;
};