import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])

# 对station_id进行编码
le = LabelEncoder()
df['station_id'] = le.fit_transform(df['station_id'])

# 保存编码器供预测时使用
joblib.dump(le, './handle/lgbm/station_id_encoder.pkl')

# 特征选择
features = ['station_id', 'hour', 'dayofweek', 'is_weekend',
            'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
            'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3']

X = df[features]
y_in = df['inflow_next']
y_out = df['outflow_next']

# 数据集分割
X_train, X_val, y_in_train, y_in_val = train_test_split(X, y_in, test_size=0.2, random_state=42)
_, _, y_out_train, y_out_val = train_test_split(X, y_out, test_size=0.2, random_state=42)

# 模型训练参数
params = {
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 训练流入模型
model_in = lgb.LGBMRegressor(**params)
model_in.fit(
    X_train, y_in_train,
    eval_set=[(X_val, y_in_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 训练流出模型
model_out = lgb.LGBMRegressor(**params)
model_out.fit(
    X_train, y_out_train,
    eval_set=[(X_val, y_out_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 保存模型
joblib.dump(model_in, './handle/lgbm/inflow_model.pkl')
joblib.dump(model_out, './handle/lgbm/outflow_model.pkl')

print("模型训练完成，已保存")