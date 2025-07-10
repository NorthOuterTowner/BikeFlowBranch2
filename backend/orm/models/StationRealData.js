const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const StationRealData = sequelize.define('StationRealData', {
    station_id: {
      type: DataTypes.STRING(20),
      allowNull: false, 
      primaryKey: true 
    },
    date: {
      type: DataTypes.DATEONLY,
      allowNull: false, 
      primaryKey: true 
    },
    hour: {
      type: DataTypes.TINYINT,
      allowNull: false,      
      primaryKey: true     
    },
    stock: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    update_time: {
      type: DataTypes.DATE, 
      allowNull: true
    }
  }, {
    tableName: 'station_real_data',
    charset: 'utf8mb4',         
    collate: 'utf8mb4_general_ci', 
    timestamps: false,          
    underscored: true,         

    indexes: [
      {
        unique: true,
        fields: ['station_id', 'date', 'hour']
      }
    ]
  });

  return StationRealData;
};