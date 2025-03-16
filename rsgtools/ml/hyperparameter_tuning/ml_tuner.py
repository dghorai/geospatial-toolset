# ml models
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBRFClassifier
# data processing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from mlmodels.cross_validation.group_k_fold import GroupKFold
from mlmodels.cross_validation.standard_k_fold import KFold
from mlmodels.cross_validation.stratified_k_fold import CustomStratifiedKFold, oversample_func
from mlmodels.cross_validation.time_series_split import CustomTimeSeriesSplit


# https://www.geeksforgeeks.org/hyperparameter-tuning/


# def svm_hyperparameter():
#     _c = []
#     kernel = []
#     gamma = []
#     pass


# def xgboost_hyperparameter():
#     learning_rate = []
#     n_estimators = []
#     max_depth = []
#     min_child_weight = []
#     sub_sample = []
#     pass


def model_specific_search_params(model_name=None, model=None, tune_params_dict=None, is_scale=False, is_reduction=False):
    model_name = model_name.lower()
    if model_name is 'gnb':
        is_reduction = False
    _add = [[('scalar', StandardScaler()), is_scale], [('pca', PCA(n_components=2)), is_reduction]]
    _pipe_params = [p[0] for p in _add if p[1] is True] + [(model_name, model)]
    _pipe = Pipeline(_pipe_params)
    _tune_params = tune_params_dict
    res = {"pipeline": _pipe, "tune_parameters": _tune_params}
    return res


def k_fold_method(n_split=3, kfold_type=None):
    if kfold_type == 'kfold':
        custom_cv = KFold(n_splits=n_split, shuffle=True, random_state=42)
    elif kfold_type == 'custom_stratified_kfold':
        custom_cv = CustomStratifiedKFold(n_splits=n_split, oversample_func=oversample_func, random_state=42)
    elif kfold_type == 'custom_group_kfold':
        custom_cv = GroupKFold(n_splits=n_split)
    else:
        custom_cv = CustomTimeSeriesSplit(n_splits=5)
    return custom_cv


def tuner(
        X_train, y_train, X_test, y_test, groups, 
        model_name=None, model=None, tune_params_dict=None, 
        label_dict=None, metric="f1_macro", n_split=None, kfold_type=None, 
        is_scale=False, is_reduction=False):
    """
    X_train, y_train:: training data
    X_test, y_test:: test data
    groups:: cluster id data
    model_name:: model name
    model:: sklearn model
    tune_params_dict:: provide a dict of the model parameters, example: {"rf__n_estimators": [50, 100, 200, 300], "rf__criterion": ["gini", "entropy"], "rf__max_depth": [5, 10, 20, 30]}
    label_dict:: provide label names and its encoded value, example: {'agriculture': 0, 'water': 1, 'forest': 2}
    metric:: provide metric available in sklearn ("accuracy", "recall_macro", "precision_macro", "f1_macro")
    n_split:: no. of K-Fold
    kfold_type:: provide following K-Fold method for data split: 'kfold' / 'custom_stratified_kfold' / 'custom_group_kfold' / 'custom_timeseries_kfold'
    is_scale:: make it True if feature data needs to be scaled
    is_reduction:: make it True if feature data dimension needs to be reduced

    """
    search_params = model_specific_search_params(
        model_name=model_name, 
        model=model, 
        tune_params_dict=tune_params_dict, 
        is_scale=is_scale, 
        is_reduction=is_reduction
    )
    if kfold_type in ['kfold', 'custom_stratified_kfold', 'custom_group_kfold', 'custom_timeseries_kfold']:
        custom_cv = k_fold_method(n_split=n_split, kfold_type=kfold_type)
    else:
        raise ValueError("expected kfold_type: kfold, custom_stratified_kfold, custom_group_kfold, custom_timeseries_kfold")
    grid_search = GridSearchCV(
        search_params["pipeline"], search_params["tune_parameters"],
        n_jobs=8,
        verbose=0,
        cv=custom_cv.split(X_train, y_train, groups),
        scoring=["accuracy", "recall_macro", "precision_macro", "f1_macro"],
        refit=metric
    )
    grid_search.fit(X_train, y_train)
    best_index = grid_search.cv_results_[f"rank_test_{metric}"].argmin()
    scores = [key for key in grid_search.cv_results_.keys() if "split" in key]
    best_scores = {}
    for score in scores:
        best_scores[score] = grid_search.cv_results_[score][best_index]
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    # prediction
    y_pred1 = best_estimator.predict(X_train)
    y_pred2 = best_estimator.predict(X_test)
    # accuracy
    train_acc = accuracy_score(y_train, y_pred1)
    test_acc = accuracy_score(y_test, y_pred2)
    # conf-matrix
    conf_matrix = confusion_matrix(y_test, y_pred2, labels=range(len(label_dict)))
    # output
    report = {
        "model_name": model_name,
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_scores": best_scores,
        "train_acc": round(train_acc, 2),
        "test_acc": round(test_acc, 2),
        "test_conf_matrix": conf_matrix
    }
    return report
