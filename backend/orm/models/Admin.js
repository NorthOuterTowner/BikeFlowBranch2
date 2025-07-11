module.exports = (sequelize,DataTypes) => {
  const Admin = sequelize.define('Admin', {
    account: {
      type: DataTypes.STRING,
      allowNull: false, 
      collate: 'latin1_swedish_ci'
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false, 
      collate: 'latin1_swedish_ci'
    },
    token: {
      type: DataTypes.STRING,
      collate: 'latin1_swedish_ci'
    },
    email: {
      type: DataTypes.STRING,
      collate: 'latin1_swedish_ci'
    }
  }, {
    tableName: 'admin', 
    charset: 'latin1',
    collate: 'latin1_swedish_ci' ,
    timestamps: false
  });

  return Admin;
};