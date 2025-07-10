const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const Admin = sequelize.define('Admin', {
    account: {
      type: DataTypes.STRING, // VARCHAR对应Sequelize中的STRING
      allowNull: false,      // 假设account不能为空
      collate: 'latin1_swedish_ci' // 设置校对规则
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false,       // 假设password不能为空
      collate: 'latin1_swedish_ci'
    },
    token: {
      type: DataTypes.STRING,
      collate: 'latin1_swedish_ci'
      // allowNull默认为true，即可以为空
    },
    email: {
      type: DataTypes.STRING,
      collate: 'latin1_swedish_ci'
      // allowNull默认为true，即可以为空
    }
  }, {
    // 其他模型选项
    tableName: 'admin', // 明确指定表名
    charset: 'latin1',  // 设置字符集
    collate: 'latin1_swedish_ci' // 全局校对规则
  });

  return Admin;
};