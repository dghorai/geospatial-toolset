"""
Cross-Validation
======================

Standard K-Fold vs. Other Cross-Validation Methods
===================================================

#1 https://www.datacamp.com/tutorial/k-fold-cross-validation

1) Standard K-Fold Cross-Validation
    a) Usage -> Both regression and classification
    b) Description -> Splits the dataset into k equal-sized folds. Each fold is used once as a test set.
    c) When to use -> Best for balanced datasets to ensure comprehensive model evaluation.
2) Stratified K-Fold Cross-Validation
    a) Usage -> Primarily classification
    b) Description -> Maintains the same proportion of class labels in each fold as the original dataset.
    c) When to use -> Great for classification tasks with imbalanced classes to maintain group proportions.
3) Leave-One-Out Cross-Validation (LOOCV)
    a) Usage -> Both regression and classification
    b) Description -> Each data point is used once as a test set, with the rest as training.
    c) When to use -> Great for small datasets to maximize training data, though computationally intensive.
4) Leave-P-Out Cross-Validation
    a) Usage -> Both regression and classification
    b) Description -> Similar to LOOCV, but leaves out p data points for the test set.
    c) When to use -> Ideal for small datasets to test how changes in the data samples affect model stability.
5) Group K-Fold Cross-Validation
    a) Usage -> Both regression and classification with groups
    b) Description -> Ensures no group is in both training and test sets, which is useful when data points are not independent.
    c) When to use -> Great for datasets with logical groupings to test performance on independent groups.
6) Stratified Group K-Fold Cross-Validation
    a) Usage -> Primarily classification with grouped data
    b) Description -> Combines stratification and group integrity, ensuring that groups are not split across folds.
    c) When to use -> Great for grouped and imbalanced datasets to maintain both class and group integrity.


Why Custom Cross-Validation Generators?
=========================================

#2 https://www.geeksforgeeks.org/creating-custom-cross-validation-generators-in-scikit-learn/

While Scikit-learn provides robust cross-validation techniques, certain situations demand customization:
1) Imbalanced Datasets:
    Standard methods might not handle class imbalance well, requiring techniques like oversampling during training.
2) Time Series Data:
    Temporal dependencies in time series data necessitate special handling to prevent information leakage.
3) Grouped Data:
    When data is grouped by certain features, maintaining these groups during cross-validation is crucial.
4) Oversampling:
    In cases of imbalanced datasets, oversampling the minority class during training can be beneficial.
    A custom generator can be designed to handle this.
5) Custom Splitting Logic:
    Sometimes, the splitting logic needs to be customized based on specific requirements,
    such as grouping by certain features or handling missing data.

"""