import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 读取特征工程后的数据
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])

# 加载编码器并编码 station_id
le = joblib.load('./handle/lgbm/station_id_encoder.pkl')
df['station_id_encoded'] = le.transform(df['station_id'].astype(str))

# 特征列（需与 train_lgbm 中保持一致）
features = [
    'station_id_encoded', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
    'temp', 'prcp', 'wspd',
    'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
    'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3'
]

# 检查是否缺少任何特征
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"特征缺失: {missing}")

X = df[features]
y_in = df['inflow_next']
y_out = df['outflow_next']

# 加载模型
model_in = joblib.load('./handle/lgbm/inflow_model.pkl')
model_out = joblib.load('./handle/lgbm/outflow_model.pkl')

# 预测
df['pred_inflow'] = model_in.predict(X)
df['pred_outflow'] = model_out.predict(X)

# 评估指标
mae_in = mean_absolute_error(y_in, df['pred_inflow'])
rmse_in = mean_squared_error(y_in, df['pred_inflow']) ** 0.5

mae_out = mean_absolute_error(y_out, df['pred_outflow'])
rmse_out = mean_squared_error(y_out, df['pred_outflow']) ** 0.5


print(f"Inflow:  MAE = {mae_in:.2f}, RMSE = {rmse_in:.2f}")
print(f"Outflow: MAE = {mae_out:.2f}, RMSE = {rmse_out:.2f}")
