const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const StationRealData = sequelize.define('StationRealData', {
    station_id: {
      type: DataTypes.STRING(20),
      allowNull: false,  // 假设station_id不能为空
      primaryKey: true   // 假设是复合主键的一部分
    },
    date: {
      type: DataTypes.DATEONLY,  // Sequelize中DATEONLY对应DATE类型
      allowNull: false,          // 假设date不能为空
      primaryKey: true           // 假设是复合主键的一部分
    },
    hour: {
      type: DataTypes.TINYINT,
      allowNull: false,          // 假设hour不能为空
      primaryKey: true           // 假设是复合主键的一部分
    },
    stock: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    update_time: {
      type: DataTypes.DATE,      // Sequelize中DATE对应DATETIME类型
      allowNull: true
    }
  }, {
    tableName: 'station_real_data', // 指定实际表名
    charset: 'utf8mb4',            // 假设使用utf8mb4字符集
    collate: 'utf8mb4_general_ci', // 假设使用utf8mb4_general_ci校对
    timestamps: false,             // 禁用自动时间戳
    underscored: true,             // 使用下划线命名风格
    // 复合主键设置
    indexes: [
      {
        unique: true,
        fields: ['station_id', 'date', 'hour']
      }
    ]
  });

  return StationRealData;
};