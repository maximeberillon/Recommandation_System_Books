# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import numpy as np

from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.score_types.base import BaseScoreType

problem_title = 'Recommendation system for books'

_train = 'data_train.csv'
_test = 'data_test.csv'

quick_mode = os.getenv('RAMP_TEST_MODE', 0)

if(quick_mode):
    _train = 'data_train_small.csv'
    _test = 'data_test_small.csv'
    

_target_column_name = 'ratingScore'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=['ratingScore'])
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true[:, 0] - y_pred[:, 0])))


class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='MAE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.mean(np.absolute(y_true[:, 0] - y_pred[:, 0]))
    

score_types = [RMSE(), MAE()]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits = 4, test_size = 0.2)
    res = cv.split(X)
    for train_index, test_index in res:
        pass
    return res


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    meta_data=pd.read_csv(os.path.join(path, 'data', 'meta.csv'))
    data=data.join(meta_data.set_index('productID'),on='productID')
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'data_train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'data_test.csv'
    return _read_data(path, f_name)
    
    
    
