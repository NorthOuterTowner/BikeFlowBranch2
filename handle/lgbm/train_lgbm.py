import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from add_features import add_features  # 确保 add_features.py 文件存在并可导入

# 加载原始样本（和训练同样的来源）
df = pd.read_csv('./handle/lgbm/lgbm_raw_samples.csv', parse_dates=['timestamp'])

# 应用特征工程（时间 + 滞后 + 节假日 + 天气）
df = add_features(df)

# 加载编码器，转换 station_id
encoder = joblib.load('./handle/lgbm/station_id_encoder.pkl')
df['station_id_encoded'] = encoder.transform(df['station_id'].astype(str))

# 准备特征（与训练保持一致）
features = [
    'station_id_encoded', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
    'temp', 'prcp', 'wspd',
    'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
    'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3'
]

# 丢弃缺失样本（由于滞后导致前几条为空）
df.dropna(subset=features + ['inflow', 'outflow'], inplace=True)

X = df[features]
y_true_inflow = df['inflow']
y_true_outflow = df['outflow']

# 加载模型并预测
model_in = joblib.load('./handle/lgbm/inflow_model.pkl')
model_out = joblib.load('./handle/lgbm/outflow_model.pkl')

df['pred_inflow'] = model_in.predict(X)
df['pred_outflow'] = model_out.predict(X)

# 计算误差指标
mae_in = mean_absolute_error(y_true_inflow, df['pred_inflow'])
rmse_in = mean_squared_error(y_true_inflow, df['pred_inflow'], squared=False)

mae_out = mean_absolute_error(y_true_outflow, df['pred_outflow'])
rmse_out = mean_squared_error(y_true_outflow, df['pred_outflow'], squared=False)

print(f"Inflow - MAE: {mae_in:.3f}, RMSE: {rmse_in:.3f}")
print(f"Outflow - MAE: {mae_out:.3f}, RMSE: {rmse_out:.3f}")

# ---------- 可视化对比 ----------
def plot_comparison(true, pred, title, ylabel):
    plt.figure(figsize=(12, 4))
    plt.plot(true.values[:200], label='真实值', marker='o', markersize=3)
    plt.plot(pred[:200], label='预测值', marker='s', markersize=3)
    plt.title(title)
    plt.xlabel('样本点（前200个）')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_comparison(y_true_inflow.values, df['pred_inflow'].values, '流入量预测 vs 真实值', '流入量')
plot_comparison(y_true_outflow.values, df['pred_outflow'].values, '流出量预测 vs 真实值', '流出量')
