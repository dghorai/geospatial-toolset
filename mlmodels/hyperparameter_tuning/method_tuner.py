# Necessary imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification



def grid_search_cv():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

    # Creating the hyperparameter grid
    c_space = np.logspace(-5, 8, 15)
    param_grid = {'C': c_space}

    # Instantiating logistic regression classifier
    logreg = LogisticRegression()

    # Instantiating the GridSearchCV object
    logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

    # Assuming X and y are your feature matrix and target variable
    # Fit the GridSearchCV object to the data
    logreg_cv.fit(X, y)

    # Print the tuned parameters and score
    print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
    print("Best score is {}".format(logreg_cv.best_score_))


def randomized_search_cv():
    import numpy as np
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset for illustration
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

    # Rest of your code (including the RandomizedSearchCV part)
    from scipy.stats import randint
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "max_depth": [3, None],
        "max_features": randint(1, 9),
        "min_samples_leaf": randint(1, 9),
        "criterion": ["gini", "entropy"]
    }

    tree = DecisionTreeClassifier()
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
    tree_cv.fit(X, y)

    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))

def bayesian_optimization():
    pass





