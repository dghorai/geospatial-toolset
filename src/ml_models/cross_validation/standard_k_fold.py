import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def example():
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
