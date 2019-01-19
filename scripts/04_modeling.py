"""Applying model to predict NO2 in city from sensor."""

import math

import pandas as pd
import seaborn as sns
import xgboost
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix

COLS_DROP = ['timestamp']
COLS_CATEGORICAL = ['feed']
COL_TARGET = ['no2_cologne']

# %% LOAD DATA AND DROP COLUMNS
le = LabelEncoder()
le.fit(pd.read_parquet('data/df_features.parquet').feed)

df = pd.read_parquet('data/df_features.parquet') \
    .assign(feed_label=lambda d: le.transform(d.feed))

# %% APPLY LINEAR REGRESSION
lin_reg = linear_model.LinearRegression(fit_intercept=False)

# prepare data
X = df.drop(columns=['timestamp'])

# dummy encoding
for column in COLS_CATEGORICAL:
    dummies = pd.get_dummies(X[column], prefix=column + '_dummy')
    X = X.join(dummies) \
        .drop(columns=column)

X = X.drop(columns=COL_TARGET)
y = df[COL_TARGET].values.ravel()

# make CV predictions
model_y = {'lin_reg': y}
model_y_pred = {'lin_reg': cross_val_predict(lin_reg, X, y, cv=100,
                                             verbose=True, n_jobs=4)}

# show parameters
lin_reg.fit(X.drop(columns=['feed_label']), y)
pd.DataFrame({'variable': X.drop(columns=['feed_label']).columns,
              'weight': lin_reg.coef_}) \
    .sort_values(by=['weight'])

# %%  APPLY XGBOOST MODEL

cutoff = df.timestamp.max() - pd.Timedelta('7d')
df_train = df[df.timestamp < cutoff].drop(columns=['timestamp', 'feed'])
df_test = df[df.timestamp >= cutoff].drop(columns=['timestamp', 'feed'])

X_train = df_train.drop(columns=COL_TARGET)
X_test = df_test.drop(columns=COL_TARGET)

y_train = df_train[COL_TARGET]
y_test = df_test[COL_TARGET]

xgb_params = {'max_depth': 3,
              'eta': 0.1,
              'objective': 'reg:linear',
              'silent': 1}

bst = xgboost.train(params=xgb_params,
                    dtrain=DMatrix(data=X_train, label=y_train),
                    evals=[(DMatrix(data=X_train, label=y_train), 'train'),
                           (DMatrix(data=X_test, label=y_test), 'test')],
                    early_stopping_rounds=500,
                    num_boost_round=10000,
                    verbose_eval=True)

# plot feature importance
xgboost.plot_importance(bst, importance_type='gain')
plt.savefig(f'results/feature_importance.png', dpi=100)
plt.close('all')

# make predictions
xgb_pred = bst.predict(DMatrix(data=X_test, label=y_test),
                       ntree_limit=bst.best_ntree_limit)
model_y['xgboost'] = y_test.values.ravel()
model_y_pred['xgboost'] = xgb_pred

# %% QUANTIFY ERRORS
metrics = pd.DataFrame({'model': list(model_y_pred.keys())})

metrics['mae'] = metrics.model.apply(
    lambda model_: mean_absolute_error(model_y[model_], model_y_pred[model_]))
metrics['rmse'] = metrics.model.apply(
    lambda model_: math.sqrt(mean_squared_error(model_y[model_],
                                                model_y_pred[model_])))

print(metrics)

# %% VISUALIZE ERRORS
model_feed = {
    'lin_reg': df.feed,
    'xgboost': le.inverse_transform(df_test.feed_label),
}

for model in model_y_pred.keys():
    df_pred = pd.DataFrame({'no2_cologne': model_y[model],
                            'no2_pred': model_y_pred[model],
                            'feed_label': model_feed[model]})
    plot = sns.scatterplot(x='no2_cologne', y='no2_pred', hue='feed_label',
                           alpha=0.5, data=df_pred, palette='colorblind',
                           s=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    plt.plot([0, 80], [0, 80], linewidth=1, linestyle='dashed', color='red')
    plt.savefig(f'results/predictions_{model}.png', dpi=150)
    plt.close('all')
