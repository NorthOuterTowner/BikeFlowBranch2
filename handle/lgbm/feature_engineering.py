# handle/lgbm/feature_engineering.py
import pandas as pd
import os

# 加载 prepare_data.py 保存的原始样本
raw = pd.read_csv('./handle/lgbm/lgbm_raw_samples.csv', parse_dates=['timestamp'])

# 排序
raw = raw.sort_values(['station_id', 'timestamp'])

# 构造未来标签列（即下一小时的 inflow/outflow）
raw['inflow_next'] = raw.groupby('station_id')['inflow'].shift(-1)
raw['outflow_next'] = raw.groupby('station_id')['outflow'].shift(-1)

# 构造时间类特征
raw['dayofweek'] = raw['timestamp'].dt.dayofweek
raw['is_weekend'] = raw['dayofweek'].isin([5, 6]).astype(int)

# 构造 inflow/outflow 的历史滞后特征（过去1/2/3小时）
for lag in [1, 2, 3]:
    raw[f'inflow_lag_{lag}'] = raw.groupby('station_id')['inflow'].shift(lag)
    raw[f'outflow_lag_{lag}'] = raw.groupby('station_id')['outflow'].shift(lag)

# 删除包含 NaN 的行（lag 或 next 可能为空）
raw = raw.dropna(subset=['inflow_next', 'outflow_next',
                         'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
                         'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3'])

# 保存特征工程后的数据
raw.to_csv('./handle/lgbm/lgbm_featured_samples.csv', index=False)
print("特征工程完成，已保存至 ./handle/lgbm/lgbm_featured_samples.csv")
