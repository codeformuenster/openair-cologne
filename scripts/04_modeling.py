"""Applying model to predict NO2 in city from sensor."""

import math

import pandas as pd
import seaborn as sns
import xgboost
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

COLS_DROP = ['timestamp']
COLS_CATEGORICAL = ['feed']
COL_TARGET = 'no2_cologne'

FEED_BLACKLIST = [
    '807f2c96',
    '807f4d84',
    '807f4f46',
    '807f5dba',
    '807f21ce',
    '807f239a',
    '807f273c',
    '807f480c',
    '807f2566',
    '807f3056',
    '807f7084',
]

FEED_WHITELIST = [
    '807f395c',
    '807f49e2',
]

# %% LOAD DATA AND ENCODE CATEGORICAL COLUMNS
df = pd.read_parquet('data/df_features.parquet') \
    .query('feed not in @FEED_BLACKLIST') \
    .query('feed in @FEED_WHITELIST')

# ENCODING TO BOTH DUMMIES AND LABELS
label_encoders = {}

for column in COLS_CATEGORICAL:
    # make fitted label encoder perdistent in global dict
    label_encoders[column] = LabelEncoder().fit(df[column])
    # create dummy columns and label series
    dummies = pd.get_dummies(df[column], prefix=column + '_dummy')
    labels = label_encoders[column].transform(df[column])
    # add encoding to data frame
    df = df.join(dummies) \
        .assign(feed_label=labels)

df = df.drop(columns=COLS_CATEGORICAL)

# %% PARTITION DATA
cutoff = df.timestamp.max() - pd.Timedelta('30d')
df_train = df[df.timestamp < cutoff]
df_test = df[df.timestamp >= cutoff]

X_train = df_train.drop(columns=[COL_TARGET]).drop(columns=['timestamp'])
X_test = df_test.drop(columns=[COL_TARGET]).drop(columns=['timestamp'])

y_train = df_train[COL_TARGET]
y_test = df_test[COL_TARGET]

# %% APPLY LINEAR REGRESSION
lin_reg = linear_model.LinearRegression(fit_intercept=False)

# show parameters
lin_reg.fit(X_train.drop(columns=['feed_label']), y_train)

lin_reg_params = pd.DataFrame({
    'variable': X_train.drop(columns=['feed_label']).columns,
    'weight': lin_reg.coef_.ravel()
}) \
    .sort_values(by=['weight'])

lin_reg_pred = lin_reg.predict(X=X_test.drop(columns=['feed_label'])).ravel()

# %%  APPLY XGBOOST MODEL
xgb_params = {'max_depth': 3,
              'learning_rate': 0.1,
              'n_estimators': 1000,
              'objective': 'reg:linear',
              'silent': 1,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.8,
              'n_jobs': 4}

xgb_reg = xgboost.XGBRegressor(**xgb_params)

xgb_reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=100)

xgb_reg_pred = xgb_reg.predict(X_test)

# plot feature importance
xgboost.plot_importance(xgb_reg, importance_type='gain')
plt.savefig(f'results/feature_importance.png', dpi=100)
plt.close('all')

# %% QUANTIFY ERRORS
model_y_pred = {
    'lin_reg': lin_reg_pred,
    'xgb_reg': xgb_reg_pred,
}
metrics = pd.DataFrame({'model': list(model_y_pred.keys())})

metrics['mae'] = metrics.model.apply(
    lambda model_: mean_absolute_error(y_test, model_y_pred[model_]))
metrics['rmse'] = metrics.model.apply(
    lambda model_: math.sqrt(mean_squared_error(y_test, model_y_pred[model_])))

print(metrics)

# %% VISUALIZE ERRORS
df_pred = df_test \
    .assign(no2_lin_reg=model_y_pred['lin_reg']) \
    .assign(no2_xgb_reg=model_y_pred['xgb_reg']) \
    .assign(feed_label=lambda d:
label_encoders['feed'].inverse_transform(d.feed_label))

# scatterplot of predictions
for model in model_y_pred.keys():
    plot = sns.scatterplot(x='no2_cologne',
                           y=f'no2_{model}',
                           hue='feed_label',
                           alpha=0.5,
                           data=df_pred,
                           palette='colorblind',
                           s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    plt.plot([0, 80], [0, 80], linewidth=1, linestyle='dashed', color='red')
    plt.savefig(f'results/predictions_{model}.png', dpi=150)
    plt.close('all')

# lineplot as predictions
df_long = pd.melt(frame=df_pred,
                  id_vars=['timestamp', 'feed'],
                  value_vars=['no2_cologne', 'no2_xgb_reg', 'no2_lin_reg'])

for feed in df_long.feed.unique():
    df_long_feed = df_long.query(f'feed == "{feed}"')
    sns.lineplot(x="timestamp",
                 y="value",
                 hue="variable",
                 data=df_long_feed)
    plt.savefig(f'results/predictions_time_{feed}.png', dpi=150)
    plt.close('all')
