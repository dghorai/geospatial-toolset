import numpy as np

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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


def oversample_func():
    return RandomOverSampler()


def example():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    custom_cv = CustomStratifiedKFold(n_splits=5, oversample_func=oversample_func, random_state=42)
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=custom_cv.split(X, y))
    print("Cross-validation scores with custom stratified oversampling:", scores)
    return

# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
