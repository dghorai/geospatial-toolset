import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


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


def example():
    # Example usage:
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    # Instantiate custom generator
    custom_cv = CustomTimeSeriesSplit(n_splits=5)
    model = Ridge()
    scores = cross_val_score(model, X, y, cv=custom_cv.split(X, y))
    print("Cross-validation scores with custom time series split:", scores)
    return

# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
