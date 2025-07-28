import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings('ignore')


train_df = pd.read_csv('/Users/shanks/Downloads/store-sales-time-series-forecasting/train.csv')
test_df = pd.read_csv('/Users/shanks/Downloads/store-sales-time-series-forecasting/test.csv')
stores_df = pd.read_csv('/Users/shanks/Downloads/store-sales-time-series-forecasting/stores.csv')
oil_df = pd.read_csv('/Users/shanks/Downloads/store-sales-time-series-forecasting/oil.csv')
holidays_events_df = pd.read_csv('/Users/shanks/Downloads/store-sales-time-series-forecasting/holidays_events.csv')

def add_datepart(df, fldname):
    fld = df[fldname]
    df['Year'] = fld.dt.year
    df['Month'] = fld.dt.month
    df['Day'] = fld.dt.day
    df['Dayofweek'] = fld.dt.dayofweek
    df['Dayofyear'] = fld.dt.dayofyear
    df['Is_month_start'] = fld.dt.is_month_start
    df['Is_month_end'] = fld.dt.is_month_end
    return df

def process_df(df, holidays_df):
    df['date'] = pd.to_datetime(df['date'])
    df = add_datepart(df, 'date')

    national_holidays = holidays_df[(holidays_df['transferred'] == False) & (holidays_df['locale'] == 'National')]
    national_holidays['date'] = pd.to_datetime(national_holidays['date'])
    df = pd.merge(df, national_holidays[['date', 'description']], on='date', how='left')
    df.rename(columns={'description': 'national_holiday'}, inplace=True)

    grouped = df.groupby(['store_nbr', 'family'])

    for days in [7, 14]:
        df[f'sales_lag_{days}'] = grouped['sales'].transform(lambda x: x.shift(days))

    for window in [7, 14]:
        df[f'sales_rolling_mean_{window}'] = grouped['sales'].transform(lambda x: x.shift(7).rolling(window).mean())
        df[f'sales_rolling_std_{window}'] = grouped['sales'].transform(lambda x: x.shift(7).rolling(window).std())

    df.fillna(0, inplace=True)

    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')

    return df

def dow_mean_encoding(df):
    mask = ~((df['date'].dt.month == 12) & (df['date'].dt.day >= 12) |
             (df['date'].dt.month == 1) & (df['date'].dt.day <= 5))
    filtered_df = df[mask]
    filtered_df = filtered_df[filtered_df['date'] >= '2016-01-01']

    aggregated_df = filtered_df.groupby(['store_nbr', 'family', 'Dayofweek'])['sales'].mean().reset_index()
    aggregated_df.rename(columns={'sales': 'dow_mean_sales'}, inplace=True)

    return pd.merge(df, aggregated_df, on=['store_nbr', 'family', 'Dayofweek'], how='left'), aggregated_df

def holiday_mean_encoding(df):
    aggregated_df = df.groupby(['store_nbr', 'family', 'national_holiday'])['sales'].mean().reset_index()
    aggregated_df.rename(columns={'sales': 'holiday_mean_sales'}, inplace=True)

    return pd.merge(df, aggregated_df, on=['store_nbr', 'family', 'national_holiday'], how='left'), aggregated_df

train_df = pd.merge(train_df, stores_df, on='store_nbr', how='left')
train_df = pd.merge(train_df, oil_df, on='date', how='left')
train_df['dcoilwtico'].bfill(inplace=True)

train_processed_df = process_df(train_df, holidays_events_df)
train_processed_df, dow_encoded_df = dow_mean_encoding(train_processed_df)
train_processed_df, holiday_encoded_df = holiday_mean_encoding(train_processed_df)

train_test_date = pd.to_datetime('2017-08-01')

features = [
    'onpromotion', 'store_nbr', 'dcoilwtico',
    'Year', 'Month', 'Day', 'Dayofweek',
    'Is_month_start', 'Is_month_end',
    'sales_lag_7', 'sales_lag_14',
    'sales_rolling_mean_7', 'sales_rolling_std_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'dow_mean_sales', 'holiday_mean_sales'
]

for col in train_processed_df.select_dtypes('category').columns:
    train_processed_df[col] = train_processed_df[col].cat.codes

train_filtered_df = train_processed_df[train_processed_df['date'] >= '2017-01-15']

X_train = train_filtered_df[train_filtered_df['date'] < train_test_date][features]
y_train = train_filtered_df[train_filtered_df['date'] < train_test_date]['sales']

X_val = train_filtered_df[train_filtered_df['date'] >= train_test_date][features]
y_val = train_filtered_df[train_filtered_df['date'] >= train_test_date]['sales']

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=10)

model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)
y_pred_val[y_pred_val < 0] = 0
rmsle_score = np.sqrt(mean_squared_log_error(y_val, y_pred_val))
print(f"\nValidation RMSLE: {rmsle_score:.4f}")

test_df = pd.merge(test_df, stores_df, on='store_nbr', how='left')
test_df = pd.merge(test_df, oil_df, on='date', how='left')
test_df['dcoilwtico'].bfill(inplace=True)

concatenated_df = pd.concat([train_df.drop(columns=stores_df.columns.drop('store_nbr')), test_df], ignore_index=True)

feat_eng_test_df = process_df(concatenated_df, holidays_events_df)
feat_eng_test_df = feat_eng_test_df[feat_eng_test_df['date'] >= test_df['date'].min()]

feat_eng_test_df = pd.merge(feat_eng_test_df, dow_encoded_df, on=['store_nbr', 'family', 'Dayofweek'], how='left')
feat_eng_test_df = pd.merge(feat_eng_test_df, holiday_encoded_df, on=['store_nbr', 'family', 'national_holiday'],
                            how='left')

for col in feat_eng_test_df.select_dtypes('category').columns:
    feat_eng_test_df[col] = feat_eng_test_df[col].cat.codes

feat_eng_test_df.fillna(0, inplace=True)

X_test_sub = feat_eng_test_df[features]
X_test_ids = feat_eng_test_df['id']

final_predictions = model.predict(X_test_sub)
final_predictions[final_predictions < 0] = 0

submission = pd.DataFrame({'id': X_test_ids, 'sales': final_predictions})
submission.to_csv('submission3.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
