"""Applying model to predict NO2 in city from sensor."""

import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

COLS_DROP = ['timestamp']
COLS_CATEGORICAL = ['feed']
COL_TARGET = ['no2_cologne']


# %% LOAD DATA AND DROP COLUMNS
le = LabelEncoder()
le.fit(pd.read_parquet('data/df_features.parquet').feed)

df = pd.read_parquet('data/df_features.parquet') \
    .drop(columns=['timestamp']) \
    .assign(feed_label=lambda d: le.transform(d.feed))

# categorical encoding
for column in COLS_CATEGORICAL:
    dummies = pd.get_dummies(df[column], prefix=column + '_dummy')
    df = df.join(dummies) \
        .drop(columns=column)

# %% SEPARATE FEATURE MATRIX FROM TARGET VECTOR
X = df.drop(columns=COL_TARGET)
y = df[COL_TARGET].values.ravel()

# %% APPLY LINEAR REGRESSION
lin_reg = linear_model.LinearRegression(fit_intercept=False)

# make CV predictions
model_y_pred = {'lin_reg': cross_val_predict(lin_reg, X, y, cv=100,
                                             verbose=True, n_jobs=4)}

# show parameters
lin_reg.fit(X.drop(columns=['feed_label']), y)
pd.DataFrame({'variable': X.drop(columns=['feed_label']).columns,
              'weight': lin_reg.coef_}) \
    .sort_values(by=['weight'])

# %% QUANTIFY ERRORS
eval = pd.DataFrame({'model': list(model_y_pred.keys())})

eval['mae'] = eval.model.apply(
    lambda model: mean_absolute_error(y, model_y_pred[model]))
eval['rmse'] = eval.model.apply(
    lambda model: math.sqrt(mean_squared_error(y, model_y_pred[model])))

print(eval)

# %% VISUALIZE ERRORS
for model in model_y_pred.keys():
    df_pred = df.assign(no2_pred=model_y_pred[model])
    # df_pred = pd.DataFrame({'y_true': y, 'y_pred': model_y_pred[model]})
    df_pred.plot.scatter(x='no2_cologne', y='no2_pred', s=0.5)
    plt.plot([0, 80], [0, 80], linewidth=1, linestyle='dashed', color='red')
    plt.savefig(f'results/predictions_{model}.png', dpi=100)
    plt.close('all')
