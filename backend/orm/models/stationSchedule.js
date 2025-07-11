
/**
 * StationSchedule 模型定义
 * @param {import('sequelize').Sequelize} sequelize - Sequelize 实例
 * @returns {import('sequelize').ModelCtor<Model>}
 */
module.exports = (sequelize,DataTypes) => {
  const StationSchedule = sequelize.define('StationSchedule', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
      allowNull: false
    },
    date: {
      type: DataTypes.DATEONLY,
      allowNull: false
    },
    hour: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    start_id: {
      type: DataTypes.STRING(20),
      allowNull: false,
      references: {
        model: 'station_info',
        key: 'station_id'
      }
    },
    end_id: {
      type: DataTypes.STRING(20),
      allowNull: false,
      references: {
        model: 'station_info',
        key: 'station_id'
      }
    },
    bikes: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: 0
    },
    // --- 状态字段更新 ---
    status: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: 0, // 默认值为 0 (pending)
      comment: '0: pending (待执行), 1: executing (正在执行), 2: completed (已完成)'
    },
    updated_at: {
      type: DataTypes.DATE,
      allowNull: true,
    }
  }, {
    tableName: 'station_schedule',
    charset: 'utf8mb4',
    collate: 'utf8mb4_general_ci',
    timestamps: true,
    createdAt: false,
    updatedAt: 'updated_at',
    underscored: true,

    indexes: [
      {
        name: 'idx_schedule_time',
        fields: ['date', 'hour']
      },
      {
        // 为 status 字段也创建索引，方便快速查询特定状态的任务
        name: 'idx_schedule_status',
        fields: ['status']
      }
    ]
  });

  return StationSchedule;
};