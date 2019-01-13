"""Pull data from API and build data frame."""

import pandas as pd

from openair_cologne.influx import query_influx

LAST_N_DAYS = 30

# %% DATA FROM LANUV STATIONS
lanuv_dict = query_influx("SELECT WTIME AS timestamp, station, NO2 AS no2 "
                          "FROM lanuv_f2 "
                          f"WHERE time >= now() - "f"{LAST_N_DAYS}d "
                          "AND time <= now() ")

# make clean data frame
df_lanuv = lanuv_dict['lanuv_f2'] \
    .assign(timestamp=lambda d: pd.to_datetime(d.timestamp,
                                               unit='ms',
                                               utc=True)) \
    .assign(timestamp=lambda d: d.timestamp.dt.floor('10min')) \
    .reset_index(drop=True)

# write as Parquet to disk
df_lanuv.to_parquet('data/df_lanuv.parquet')

# %% DATA FROM OPENAIR
openair_dict_raw = query_influx("SELECT "
                                "median(hum) AS hum, median(pm10) AS pm10, "
                                "median(pm25) AS pm25, median(r1) AS r1, "
                                "median(r2) AS r2, median(rssi) AS rssi, "
                                "median(temp) AS temp "
                                "FROM all_openair "
                                f"WHERE time >= now() - {LAST_N_DAYS}d "
                                "AND time <= now() "
                                "GROUP BY feed, time(10m) fill(-1) ")
# clean dictionary keys
openair_dict = {k[1][0][1]: openair_dict_raw[k]
                for k in openair_dict_raw.keys()}

# initialize empty data frame
df_openair = pd.DataFrame()
# fill data frame with data from all frames
# OPTIONAL: replace for-loop with map-reduce
for feed in list(openair_dict.keys()):
    df_feed = pd.DataFrame.from_dict(openair_dict[feed]) \
        .assign(feed=feed) \
        .rename_axis('timestamp').reset_index()
    df_openair = df_openair.append(df_feed)

# write as Parquet to disk
df_openair.to_parquet('data/df_openair.parquet')
