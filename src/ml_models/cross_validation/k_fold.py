# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:26:11 2026

@author: Debabrata Ghorai, Ph.D.

K-fold cross validation.

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler


class GroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits)

    def split(self, X, y, groups):
        unique_groups = np.unique(groups)
        for train_group_idx, test_group_idx in self.kf.split(unique_groups):
            train_idx = np.isin(groups, unique_groups[train_group_idx])
            test_idx = np.isin(groups, unique_groups[test_group_idx])
            yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class CustomStratifiedKFold:
    def __init__(self, n_splits=5, oversample_func=None, random_state=None):
        self.n_splits = n_splits
        self.oversample_func = oversample_func
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def split(self, X, y):
        for train_index, test_index in self.skf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]

            # Oversample only the training data
            if self.oversample_func is not None:
                X_train, y_train = self.oversample_func().fit_resample(X_train, y_train)

            yield train_index, test_index

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class CustomTimeSeriesSplit:
    def __init__(self, n_splits=5, max_train_size=None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_size)

    def split(self, X, y=None, groups=None):
        for train_index, test_index in self.tscv.split(X):
            yield train_index, test_index

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits



def oversample_func():
    return RandomOverSampler()


# ======= EXAMPLE ===========
def example_standard_k_fold():
    data = fetch_california_housing()
    # Preparing the data
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    # Setting up K-Fold Cross-Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # Initializing the model
    model = LinearRegression()
    # Performing cross-validation
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    # Calculating the average R2 score
    average_r2 = np.mean(scores)
    # Displaying the final results
    print(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
    print(f"Average R² across {k} folds: {average_r2:.2f}")
    return


def example_group_k_fold():
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=2, 
        n_redundant=2, 
        n_clusters_per_class=1, 
        random_state=42
    )
    groups = np.random.randint(0, 20, size=1000)  # 20 different groups
    custom_cv = GroupKFold(n_splits=5)
    rfc = RandomForestClassifier(random_state=42)
    scores = cross_val_score(
        estimator=rfc, 
        X=X, 
        y=y, 
        cv=custom_cv.split(X, y, groups), 
        scoring='accuracy', 
        n_jobs=-1
    )
    print("Cross-validation scores: ", scores)
    print("Mean cross-validation score: ", scores.mean())
    return


def example_grid_search_group_k_fold():
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=2, 
        n_redundant=2, 
        n_clusters_per_class=1, 
        random_state=42
    )
    groups = np.random.randint(0, 20, size=1000)  # 20 different groups
    custom_cv = GroupKFold(n_splits=5)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None]
    }
    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid, 
        cv=custom_cv.split(X, y, groups), 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X, y)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    return

def example_stratified_k_fold():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    custom_cv = CustomStratifiedKFold(n_splits=5, oversample_func=oversample_func, random_state=42)
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=custom_cv.split(X, y))
    print("Cross-validation scores with custom stratified oversampling:", scores)
    return

def example_time_series_k_fold():
    # Example usage:
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    # Instantiate custom generator
    custom_cv = CustomTimeSeriesSplit(n_splits=5)
    model = Ridge()
    scores = cross_val_score(model, X, y, cv=custom_cv.split(X, y))
    print("Cross-validation scores with custom time series split:", scores)
    return


# https://www.kaggle.com/code/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
