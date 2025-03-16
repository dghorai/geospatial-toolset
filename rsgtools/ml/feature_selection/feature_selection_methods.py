# Build a voting selector
import warnings
import pandas as pd
import numpy as np

from logger import logger
from sklearn.model_selection import train_test_split  # , GridSearchCV, KFold
from sklearn.feature_selection import RFE, chi2, mutual_info_regression, mutual_info_classif, SelectFromModel  # , SelectKBest
# , LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.inspection import permutation_importance
# , GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
# from mlxtend.feature_selection import ExhaustiveFeatureSelector
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from statsmodels.tools.tools import add_constant

warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.reset_option('all')
# warnings.filterwarnings("error")
# pd.options.future.infer_string = True
pd.options.mode.chained_assignment = None  # default='warn'

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class FeatureSelectorMethod:
    """Define feature selection methods"""

    def __init__(self):
        pass

    @staticmethod
    def series_to_df(series):
        df = series.to_frame()
        df.reset_index(drop=False, inplace=True)
        df.columns = ['feature', 'score']
        return df

    @classmethod
    def random_forest_importance(cls, X, y, supervised_type=None):
        """features with higher scores are best features"""

        if supervised_type == "Classification":
            forest = RandomForestClassifier(random_state=0)
        else:
            forest = RandomForestRegressor(random_state=0)

        forest.fit(X, y)
        sorted_idx = forest.feature_importances_.argsort()
        cols = X.columns[sorted_idx]
        importances = forest.feature_importances_[sorted_idx]
        series = pd.Series(importances, index=cols)
        series.sort_values(ascending=False, inplace=True)

        # series to dataframe
        rf_score = cls.series_to_df(series)

        return rf_score

    @classmethod
    def mutual_information(cls, X, y, supervised_type=None):
        """features with higher scores are best features"""

        if supervised_type == 'Classification':
            mutual_info = mutual_info_classif(X, y)
        else:
            mutual_info = mutual_info_regression(X, y)

        series = pd.Series(mutual_info, index=X.columns)
        series.sort_values(ascending=False, inplace=True)

        # series to dataframe
        mi_score = cls.series_to_df(series)

        return mi_score

    @classmethod
    def lasso_regularization(cls, X, y, supervised_type=None):
        """coefficient equal to 0 can be removed from feature selection"""

        # this is only for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0, stratify=y)

        if supervised_type == 'Classification':
            alphas = np.arange(1e-10, 0.1, 0.01)
            mxacc = 0
            best_alpha = None

            for i in alphas:
                lso = Lasso(alpha=i)
                lso.fit(X_train, y_train)
                y_pred = lso.predict(X_test)
                y_pred = np.int_(np.round_(y_pred))
                acc = accuracy_score(y_test.to_numpy(), y_pred)
                if acc > mxacc:
                    mxacc = acc
                    best_alpha = i

            # remember to set the seed, the random state in this function
            selector = SelectFromModel(
                Lasso(alpha=best_alpha, random_state=0))
            selector.fit(X_train, y_train)

            # selector.get_support()
            lasso1_coef = np.abs(selector.estimator_.coef_)
            # this is how we can make a list of the selected features
            lasso_feature = list(X.columns[(selector.get_support())])
            series = pd.Series(lasso1_coef, index=lasso_feature)
            series.sort_values(ascending=False, inplace=True)

            # series to dataframe
            lasso_score = cls.series_to_df(series)
        else:
            lasso_score = pd.DataFrame()

        return lasso_score

    @classmethod
    def chi2_test(cls, X, y, supervised_type=None):
        """features with higher scores are best features"""

        # Chi-square test is used for categorical features in a dataset
        if supervised_type == 'Classification':
            chi2_scores, p_value = chi2(X, y)
            series = pd.Series(chi2_scores, index=X.columns)
            series.sort_values(ascending=False, inplace=True)
            # series to dataframe
            chi2_score = cls.series_to_df(series)
        else:
            chi2_score = pd.DataFrame()

        return chi2_score

    @staticmethod
    def recursive_features_elimination(X, y, supervised_type=None, req_feature=None):
        """provide feature count to select best no. of features"""

        if supervised_type == 'Classification':
            rfe = RFE(estimator=DecisionTreeClassifier(),
                      n_features_to_select=req_feature)
        else:
            rfe = RFE(estimator=DecisionTreeRegressor(),
                      n_features_to_select=req_feature)

        rfe.fit(X, y)

        cols = X.columns.tolist()
        supports = rfe.support_
        ranks = rfe.ranking_  # selected feature rank always 1

        features = []
        for col, sup, rank in zip(cols, supports, ranks):
            if sup:
                features.append([col, rank])

        rfe_score = pd.DataFrame(features, columns=['feature', 'score'])

        return rfe_score

    @classmethod
    def multicolinearity_test(cls, X, y=None, supervised_type=None):
        """features with lower scores are best features"""

        # const = add_constant(X)
        # vif_scores = pd.DataFrame()
        # vif_scores["feature"] = const.columns
        # vif_scores['score'] = [variance_inflation_factor(
        #     const.values, feature) for feature in range(len(const.columns))]
        # vif_scores = vif_scores.sort_values(by=['score'], ascending=True)
        df_cor = X.corr()
        series = pd.Series(np.linalg.inv(
            X.corr().values).diagonal(), index=df_cor.index)
        vif_scores = cls.series_to_df(series)
        vif_scores = vif_scores.sort_values(by=['score'], ascending=True)
        vif_scores.reset_index(drop=True, inplace=True)
        return vif_scores


class BestFeatureSelector(FeatureSelectorMethod):
    """Select best features for ml model"""

    def __init__(self):
        super().__init__()

        self.selectors = {
            "random_forest_importance": self._select_forest,
            "mutual_information": self._select_mutual,
            "chi2_test": self._select_chi2,
            "recursive_features_elimination": self._select_rfe,
            "multi_collinearity_test": self._select_colinearity,
            "lasso_regularization": self._select_l1,
        }

        self.best_features = None

    @classmethod
    def _select_forest(cls, X, y, **kwargs):
        selector = cls.random_forest_importance(
            X, y, supervised_type=kwargs.get("supervised_type"))
        return selector

    @classmethod
    def _select_mutual(cls, X, y, **kwargs):
        selector = cls.mutual_information(
            X, y, supervised_type=kwargs.get("supervised_type"))
        return selector

    @classmethod
    def _select_chi2(cls, X, y, **kwargs):
        selector = cls.chi2_test(
            X, y, supervised_type=kwargs.get("supervised_type"))
        return selector

    @classmethod
    def _select_rfe(cls, X, y, **kwargs):
        selector = cls.recursive_features_elimination(X, y, supervised_type=kwargs.get(
            "supervised_type"), req_feature=kwargs.get("n_feature_to_select"))
        return selector

    @classmethod
    def _select_colinearity(cls, X, y, **kwargs):
        selector = cls.multicolinearity_test(
            X, y, supervised_type=kwargs.get("supervised_type"))
        return selector

    @classmethod
    def _select_l1(cls, X, y, **kwargs):
        selector = cls.lasso_regularization(
            X, y, supervised_type=kwargs.get("supervised_type"))
        return selector

    def select(self, X, y, **kwargs):
        n_feature = kwargs.get("n_feature_to_select")
        # selection_option = kwargs.get("selection_criteria")

        dfs = []
        for selector_name, selector_method in self.selectors.items():
            logger.info(f"Running best feature selector: {selector_name}")

            if selector_name == 'recursive_features_elimination':
                result = selector_method(X, y, **kwargs)
                result.loc[:, 'ranks'] = n_feature
                dfs.append(result)
            else:
                result = selector_method(X, y, **kwargs)
                selector = result.head(n_feature)
                selector.loc[:, 'ranks'] = [
                    i+1 for i in reversed(range(n_feature))]
                dfs.append(selector)

        # concat
        mdf = pd.concat(dfs, axis=0)
        return mdf
