"""Pull data from API and build data frame."""

from influxdb import DataFrameClient
import pandas as pd

LAST_N_DAYS = 30


# %% DATABASE CONNECTION
def get_db_client():
    client_ = DataFrameClient(host='openair.cologne.codefor.de/influx',
                              port=443,
                              database='lanuv')
    return client_


client = get_db_client()

# %% DATA FROM LANUV STATIONS

# no2 data from official stations (last X days)
lanuv_dict = client.query("SELECT WTIME, station, NO2 "
                          "FROM lanuv_f2 "
                          f"WHERE time >= now() - {LAST_N_DAYS}d "
                          "AND time <= now() ")

# make clean data frame
df_lanuv = lanuv_dict['lanuv_f2'] \
    .reset_index(drop=True) \
    .rename(columns={'WTIME': 'timestamp', 'NO2': 'no2'})

# write as Parquet to disk
df_lanuv.to_parquet('data/df_lanuv.parquet')

# %% GET ALL FEEDS FROM OPENAIR

feeds_dict = client.query("SELECT mean(r2) "
                          "from all_openair "
                          "group by feed "
                          "LIMIT 1")

all_feeds = list(set([k[1][0][1] for k in feeds_dict.keys()]))


# %% DATA FROM OPENAIR
def query_openair(last_n_days: int, feed_: str):
    # get data
    client_ = get_db_client()
    openair_dict = client_.query("SELECT "
                                 "median(hum) AS hum, median(pm10) AS pm10, "
                                 "median(pm25) AS pm25, median""(r1) AS r1, "
                                 "median(r2) AS r2, median(rssi) AS rssi, "
                                 "median(temp) AS temp "
                                 "FROM all_openair "
                                 f"WHERE time >= now() - {last_n_days}d "
                                 "AND time <= now() "
                                 f"AND feed = '{feed_}' "
                                 "GROUP BY time(10m) fill(-1)")
    client_.close()
    if 'all_openair' not in openair_dict.keys():
        return pd.DataFrame()
    # post-process
    df_openair = openair_dict['all_openair'] \
        .reset_index().rename(columns={'index': 'timestamp'}) \
        .assign(feed=lambda d: feed_)

    return df_openair


# initialize empty data frame
df_feeds = pd.DataFrame()
# fill data frame with data from all frames
for feed in all_feeds:
    df_feed = query_openair(last_n_days=LAST_N_DAYS, feed_=feed)
    df_feeds = df_feeds.append(df_feed)

# write as Parquet to disk
df_feeds.to_parquet('data/df_lanuv.parquet')

# %% CLOSE DATABASE CONNECTION
client.close()
