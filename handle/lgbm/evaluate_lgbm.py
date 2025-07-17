import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import datetime, timedelta
import os

# 1. 加载数据和模型
df = pd.read_csv('./handle/lgbm/lgbm_featured_samples.csv', parse_dates=['timestamp'])
encoder = load('./handle/lgbm/station_id_encoder.pkl')
model_in = load('./handle/lgbm/inflow_model_tuned.pkl')
model_out = load('./handle/lgbm/outflow_model_tuned.pkl')

# 2. 数据预处理
df['station_id_encoded'] = encoder.transform(df['station_id'].astype(str))
features = [
    'station_id_encoded', 'hour', 'dayofweek', 'is_weekend', 'is_holiday',
    'temp', 'prcp', 'wspd',
    'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
    'outflow_lag_1', 'outflow_lag_2', 'outflow_lag_3'
]

# 3. 预测并计算指标
df['pred_inflow'] = model_in.predict(df[features])
df['pred_outflow'] = model_out.predict(df[features])

# 4. 输出评估指标
def print_metrics():
    mae_in = mean_absolute_error(df['inflow_next'], df['pred_inflow'])
    rmse_in = np.sqrt(mean_squared_error(df['inflow_next'], df['pred_inflow']))
    mae_out = mean_absolute_error(df['outflow_next'], df['pred_outflow'])
    rmse_out = np.sqrt(mean_squared_error(df['outflow_next'], df['pred_outflow']))
    
    print("\n" + "="*50)
    print("全局评估指标：")
    print(f"Inflow  - MAE: {mae_in:.2f}, RMSE: {rmse_in:.2f}")
    print(f"Outflow - MAE: {mae_out:.2f}, RMSE: {rmse_out:.2f}")
    print("="*50 + "\n")

# 5. 输出可用的时间点信息
def print_available_times():
    print("\n可用时间范围：")
    print(f"最早时间：{df['timestamp'].min()}")
    print(f"最晚时间：{df['timestamp'].max()}")
    
    date_counts = df['timestamp'].dt.date.value_counts().sort_index()
    print("\n日期分布：")
    print(date_counts.head(10))
    
    hour_counts = df['timestamp'].dt.hour.value_counts().sort_index()
    print("\n小时分布：")
    print(hour_counts)

# 6. 绘制指定站点的时序图
def plot_station_trend(station_id, start_date=None, end_date=None):
    station_encoded = encoder.transform([station_id])[0]
    station_data = df[df['station_id_encoded'] == station_encoded].copy()

    if start_date:
        start_date = pd.to_datetime(start_date)
        station_data = station_data[station_data['timestamp'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        station_data = station_data[station_data['timestamp'] <= end_date]

    if station_data.empty:
        print(f"警告：站点 {station_id} 在指定时间段无数据")
        return

    station_data = station_data.sort_values('timestamp')

    # 平滑处理
    station_data['inflow_next_smooth'] = station_data['inflow_next'].rolling(window=3, center=True).mean()
    station_data['pred_inflow_smooth'] = station_data['pred_inflow'].rolling(window=3, center=True).mean()
    station_data['outflow_next_smooth'] = station_data['outflow_next'].rolling(window=3, center=True).mean()
    station_data['pred_outflow_smooth'] = station_data['pred_outflow'].rolling(window=3, center=True).mean()

    # 局部评估
    mae_in = mean_absolute_error(station_data['inflow_next'], station_data['pred_inflow'])
    rmse_in = np.sqrt(mean_squared_error(station_data['inflow_next'], station_data['pred_inflow']))
    mae_out = mean_absolute_error(station_data['outflow_next'], station_data['pred_outflow'])
    rmse_out = np.sqrt(mean_squared_error(station_data['outflow_next'], station_data['pred_outflow']))

    os.makedirs('./handle/lgbm/eval_plot', exist_ok=True)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(station_data['timestamp'], station_data['inflow_next_smooth'], label='True Inflow', color='blue', marker='o', markersize=3)
    plt.plot(station_data['timestamp'], station_data['pred_inflow_smooth'], label='Predicted Inflow', color='red', marker='o', markersize=3)
    plt.title(f'Station {station_id} - Inflow\nMAE: {mae_in:.2f}, RMSE: {rmse_in:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(station_data['timestamp'], station_data['outflow_next_smooth'], label='True Outflow', color='green', marker='o', markersize=3)
    plt.plot(station_data['timestamp'], station_data['pred_outflow_smooth'], label='Predicted Outflow', color='orange', marker='o', markersize=3)
    plt.title(f'Station {station_id} - Outflow\nMAE: {mae_out:.2f}, RMSE: {rmse_out:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./handle/lgbm/eval_plot/station_{station_id}_trend.png', dpi=300)
    plt.show()
    print("图像已保存")

if __name__ == "__main__":
    print_metrics()
    print_available_times()

    target_station = "HB102"
    start_date = "2025-05-13"
    end_date = "2025-05-21"

    plot_station_trend(target_station, start_date, end_date)