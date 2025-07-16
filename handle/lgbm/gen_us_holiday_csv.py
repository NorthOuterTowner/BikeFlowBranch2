import holidays
import pandas as pd

us_holidays = holidays.US(years=[2025])
df = pd.DataFrame([
    {'date': d, 'is_holiday': 1, 'holiday_name': name}
    for d, name in us_holidays.items()
])

# 保存
df.to_csv('./handle/lgbm/us_holidays_2025.csv', index=False)
print("已保存 us_holidays_2025.csv，共", len(df), "个节假日")
