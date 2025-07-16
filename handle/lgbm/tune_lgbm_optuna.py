import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])

# 编码 station_id
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

# 分割训练集和验证集
X_train, X_val, y_in_train, y_in_val = train_test_split(X, y_in, test_size=0.2, random_state=42)
_, _, y_out_train, y_out_val = train_test_split(X, y_out, test_size=0.2, random_state=42)

# ================================
# 贝叶斯调参目标函数 - inflow
# ================================
def objective_inflow(trial):
    param = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
    }
    model = lgb.LGBMRegressor(**param, n_estimators=500)
    model.fit(X_train, y_in_train,
              eval_set=[(X_val, y_in_val)],
              eval_metric='l2',
              callbacks=[lgb.early_stopping(10)],
              )
    preds = model.predict(X_val)
    return mean_squared_error(y_in_val, preds)

# ================================
# 贝叶斯调参目标函数 - outflow
# ================================
def objective_outflow(trial):
    param = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
    }
    model = lgb.LGBMRegressor(**param, n_estimators=500)
    model.fit(X_train, y_out_train,
              eval_set=[(X_val, y_out_val)],
              eval_metric='l2',
              callbacks=[lgb.early_stopping(10)],
              )
    preds = model.predict(X_val)
    return mean_squared_error(y_out_val, preds)

# ================================
# 启动调参逻辑
# ================================
def main(mode='inflow'):
    if mode == 'inflow':
        print("正在优化 inflow 模型参数...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_inflow, n_trials=50)
        fig = vis.plot_optimization_history(study).figure
        fig.savefig(f'./handle/lgbm/{mode}_optuna_convergence.png', dpi=200)
        print(f"已保存 {mode} 的调参收敛图：{mode}_optuna_convergence.png")
        print("inflow 最佳参数：", study.best_params)

        # 重新训练并保存
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'boosting_type': 'gbdt'
        })
        model = lgb.LGBMRegressor(**best_params, n_estimators=500)
        model.fit(X_train, y_in_train,
                  eval_set=[(X_val, y_in_val)],
                  eval_metric='l2',
                  callbacks=[lgb.early_stopping(10)])
        joblib.dump(model, './handle/lgbm/inflow_model_tuned.pkl')
        print("inflow 模型已保存为 inflow_model_tuned.pkl")

    elif mode == 'outflow':
        print("正在优化 outflow 模型参数...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_outflow, n_trials=50)
        fig = vis.plot_optimization_history(study).figure
        fig.savefig(f'./handle/lgbm/{mode}_optuna_convergence.png', dpi=200)
        print(f"已保存 {mode} 的调参收敛图：{mode}_optuna_convergence.png")
        print("outflow 最佳参数：", study.best_params)

        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'boosting_type': 'gbdt'
        })
        model = lgb.LGBMRegressor(**best_params, n_estimators=500)
        model.fit(X_train, y_out_train,
                  eval_set=[(X_val, y_out_val)],
                  eval_metric='l2',
                  callbacks=[lgb.early_stopping(10)])
        joblib.dump(model, './handle/lgbm/outflow_model_tuned.pkl')
        print("outflow 模型已保存为 outflow_model_tuned.pkl")
    else:
        print("错误：mode 应为 'inflow' 或 'outflow'")

# ================================
# 命令行入口
# ================================
if __name__ == '__main__':
    main('outflow')
