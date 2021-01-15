import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

mean_ratingScore = np.mean(y_train)

def svd_prediction(matrix, array, u,i):
    '''
    Given a user-product matrix and the corresponding 2D array
    Return the predicted ratingScore
    If its a new user or a new product it returns the mean ratingScore
    '''
    if i in ratings_matrix.index:
        idx_i = matrix.index.get_loc(i)
        idx_u = matrix.columns.get_loc(u)
        pred = array[idx_i][idx_u]
    else:
        pred = mean_ratingScore
    return pred

def estimator(X_train, y_train, X_test):
    #Preprocessing the data a bit
    data = data = X_train.copy()
    data['ratingScore'] = y_train

    #Getting our user-product matrix
    ratings_matrix = data.pivot_table(values='ratingScore', index='productID', columns='reviewerID', fill_value=0)
    ratings_array = ratings_matrix.values

    #Running the SVD on it
    imp = IterativeImputer(missing_values=0, random_state=0, min_value=1, max_value=5)
    ratings_array_predicted = imp.fit_transform(ratings_array)

    #Returning the prediciton
    y_pred = [svd_prediction(x[0], x[1]) for x in X_test.values]
    return y_pred