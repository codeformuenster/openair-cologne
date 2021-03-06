"""Feature engineering."""

import pandas as pd

# %% LOAD
df_lanuv = pd.read_parquet('data/df_lanuv.parquet') \
    .groupby(['timestamp', 'station'])['no2'].aggregate('median') \
    .reset_index() \
    .groupby(['timestamp'])['no2'].aggregate('median') \
    .reset_index() \
    .rename(columns={'no2': 'no2_cologne'})

df_openair = pd.read_parquet('data/df_openair.parquet') \
    .drop(columns=['pm10', 'pm25']) \
    .query('r1 != -1 and r2 != -1') \
    .query('hum <= 100 and temp < 45') \
    .assign(feed=lambda d: d.feed.str.split('-').map(lambda x: x[0]))

# %% JOIN DATA
df_joined = pd.merge(df_lanuv, df_openair, how='inner', on=['timestamp'])

# %% TEMPORAL FEATURES
# TODO

# %% WRITE RESULT
df_joined.to_parquet('data/df_features.parquet')
