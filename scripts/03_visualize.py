"""Visualize data before applying model."""

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

# %% LOAD DATA
df = pd.read_parquet('data/df_features.parquet')

# %% BOX PLOTS

# temperature and humidity
plot_box1 = df[['temp', 'hum']].boxplot()
plt.savefig('results/boxplot_temp_hum.png', dpi=150)
plt.close('all')

# sensor readings
plot_box2 = df[['r1', 'r2', 'rssi']].boxplot()
plt.savefig('results/boxplot_sensor.png', dpi=150)
plt.close('all')

# target variable
plot_box3 = df[['no2_cologne']].boxplot()
plt.savefig('results/boxplot_no2_cologne.png', dpi=150)
plt.close('all')

# %% SCATTER PLOTS
plot_scatter = scatter_matrix(df, alpha=0.2, diagonal='kde')
plt.savefig('results/scatterplot.png', dpi=200)
plt.close('all')

for feed in df.feed.unique():
    # only one feed
    df_feed = df.query(f'feed == "{feed}"')
    if df_feed.shape[0] < 2: continue  # if not enough data, continue
    # make scatterplot
    plot_scatter = scatter_matrix(df_feed, alpha=0.2, diagonal='kde')
    plt.savefig(f'results/scatterplot_{feed}.png', dpi=200)
    plt.close('all')

# %% TIME SERIES PLOTS

feeds = df.feed.unique()

for feed in feeds:
    ts_plot = df.query(f'feed == "{feed}"') \
        .plot(subplots=True, figsize=(6, 6))
    plt.savefig(f'results/timeseries_{feed}.png', dpi=200)
    plt.close('all')
