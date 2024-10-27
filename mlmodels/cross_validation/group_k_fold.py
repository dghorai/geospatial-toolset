import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


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


def example():
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


def example_grid_search():
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

# https://www.kaggle.com/code/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
# https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/
