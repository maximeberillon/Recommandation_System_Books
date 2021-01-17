import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator


def svd_prediction(matrix, array, mean, u, i):
    '''
    Given a user-product matrix and the corresponding 2D array
    Return the predicted ratingScore
    If its a new user or a new product it returns the mean ratingScore
    '''
    if i in matrix.index:
        idx_i = matrix.index.get_loc(i)
        idx_u = matrix.columns.get_loc(u)
        pred = array[idx_i][idx_u]
    else:
        pred = mean
    return pred


class Regressor(BaseEstimator):
    def __init__(self):
        self.imp = IterativeImputer(missing_values=0,
                                    random_state=0,
                                    min_value=1,
                                    max_value=5)

    def fit(self, X, y):
        self.mean_ratingScore = np.mean(y)
        data = X.loc[:, ['reviewerID', 'productID']]
        data['ratingScore'] = y
        self.ratings_matrix = data.pivot_table(values='ratingScore',
                                               index='productID',
                                               columns='reviewerID',
                                               fill_value=0)
        ratings_array = self.ratings_matrix.values
        self.ratings_array_predicted = self.imp.fit_transform(ratings_array)
        return self

    def predict(self, X):
        y_pred = [svd_prediction(self.ratings_matrix,
                                 self.ratings_array_predicted,
                                 self.mean_ratingScore,
                                 x[0], x[1]) for x in X.values]
        y_pred=np.array(y_pred)
        return y_pred.reshape((y_pred.shape[0],-1))
