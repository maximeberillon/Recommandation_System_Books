import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.a=0

    def fit(self, X, y):
        return self

    def predict(self, X):
        y_pred = np.zeros(len(X))
        return y_pred
