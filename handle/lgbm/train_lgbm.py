import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import lightgbm as lgb
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])

# 对 station_id 编码
le = LabelEncoder()
df['station_id'] = le.fit_transform(df['station_id'])
joblib.dump(le, './handle/lgbm/station_id_encoder.pkl')

# 特征列
features = ['station_id', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
            'temp', 'prcp', 'wspd',
            'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
            'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3']
X = df[features]
y_in = df['inflow_next']
y_out = df['outflow_next']

# 划分训练集和验证集
X_train, X_val, y_in_train, y_in_val = train_test_split(X, y_in, test_size=0.2, random_state=42)
_, _, y_out_train, y_out_val = train_test_split(X, y_out, test_size=0.2, random_state=42)

# LightGBM 参数
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

# 训练 inflow 模型
model_in = lgb.LGBMRegressor(**params)
model_in.fit(
    X_train, y_in_train,
    eval_set=[(X_val, y_in_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 训练 outflow 模型
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

# ------------------------------
# ✅ SHAP 分析（仅 inflow 模型）
# ------------------------------

# 用训练好的 inflow 模型进行 SHAP 分析
explainer = shap.Explainer(model_in)
shap_values = explainer(X)

# 输出全局特征重要性图（蜂群图）
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('./handle/lgbm/shap_summary_inflow.png')
print("SHAP 分析完成，已保存至 shap_summary_inflow.png")

# 可选保存 SHAP 数据供未来深入分析
joblib.dump((X, shap_values), './handle/lgbm/shap_values_inflow.pkl')
