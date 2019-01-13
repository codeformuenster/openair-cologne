"""Applying model to predict NO2 in city from sensor."""

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict

COLS_DROP = ['timestamp']
COLS_CATEGORICAL = ['feed']
COL_TARGET = ['no2_cologne']

MODELS = {
    'LinearRegression': linear_model.LinearRegression(fit_intercept=False),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=300),
    'RandomForestRegressor': RandomForestRegressor(max_depth=8,
                                                   n_estimators=100)
}

# %% LOAD DATA AND DROP COLUMNS
df = pd.read_parquet('data/df_features.parquet') \
    .drop(columns=['timestamp'])

# categorical encoding
for column in COLS_CATEGORICAL:
    dummies = pd.get_dummies(df[column], prefix=column)
    df = df.join(dummies) \
        .drop(columns=column)

# %% SEPARATE FEATURE MATRIX FROM TARGET VECTOR
X = df.drop(columns=COL_TARGET)
y = df[COL_TARGET].values.ravel()

# %% TRAIN CROSS-VALIDATION
model_y_pred = {model: cross_val_predict(MODELS[model], X, y, cv=10,
                                         verbose=True, n_jobs=4)
                for model in MODELS.keys()}

# %% STACKING: AVERAGE PREDICTIONS FROM ALL MODELS

model_y_pred['AllModels_mean'] = \
    np.mean(np.array(list(model_y_pred.values())), axis=0)
model_y_pred['AllModels_median'] = \
    np.median(np.array(list(model_y_pred.values())), axis=0)

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
