import numpy as np
import pandas as pd
from datetime import datetime

train = pd.read_csv('./datasets/train.csv') # https://www.kaggle.com/c/bike-sharing-demand/overview
test = pd.read_csv('./datasets/test.csv') # https://www.kaggle.com/c/bike-sharing-demand/overview

df = pd.concat([train, test])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

time = []
month = []
for i in df['datetime']:
    dt_object2 = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
    time.append(dt_object2.hour)
    month.append(dt_object2.month)
df['time'] = time
df['time'] = df['time'].astype(float)
df['month'] = month
df['month'] = df['month'].astype(float)

df.drop(['datetime', 'holiday', 'atemp'], axis=1, inplace=True)

# One hot encoding on categorical columns.
df = pd.get_dummies(df, columns=['season', 'weather'], drop_first=True)

# Median imputation if there are any null values
for i in df.columns:
    df[i].fillna(value=df[i].median(), inplace=True)


sample_size = 300
print(f'bike sharing dataset was created: {sample_size} sample')
# if not sample, shap puts 1 hour for creating an explanation
df.sample(sample_size).to_csv('./datasets/bike_sharing.csv')