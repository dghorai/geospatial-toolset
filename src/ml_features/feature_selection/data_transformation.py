# Build a voting selector
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# from src.logger import logger

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class NumericNormalize(BaseEstimator, TransformerMixin):
    def __init__(self, variables, label_id):
        self.variables = variables
        self.label_id = label_id

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.variables)
        numerical_columns = list(X.columns[X.dtypes != 'object'])
        for col in numerical_columns:
            if col not in [self.label_id]:
                minv = X[col].min()
                maxv = X[col].max()
                if (minv < 0) or (maxv > 1):
                    X[col] = (X[col] - minv) / (maxv - minv)
        return X


class DataProcessing(NumericNormalize):
    """Prepare data"""

    def __init__(self):
        pass

    def processing(self, df, label_id_col, label_name_col, drop_columns: list = None):
        not_useful_cols = drop_columns
        df2 = df.drop(not_useful_cols, axis=1)
        label_encoder = LabelEncoder()

        # df2['class_id'] = label_encoder.fit_transform(df2['class_name'])
        df2[label_id_col] = label_encoder.fit_transform(df2[label_name_col])

        # seperate x and y data
        X, y = df2.drop([label_name_col, label_id_col],
                        axis=1), df2[label_id_col]

        # normalize numerical_columns
        df3 = X.copy()
        columns = df3.columns.tolist()
        normalize_variables = NumericNormalize(
            variables=columns, label_id=label_id_col)

        # find fields
        numerical_columns = list(X.columns[X.dtypes != 'object'])
        categorical_columns = list(X.columns[X.dtypes == 'object'])

        # find sub-category of categorical columns
        sub_category = []
        for col in categorical_columns:
            if col not in [label_name_col]:
                subcat = df3[col].unique().tolist()
                sub_category.append(subcat)

        # data transformation
        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', normalize_variables)
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(
                    strategy='constant', fill_value='unknown')),
                ('ordinalencoder', OrdinalEncoder(categories=sub_category)),
                ('scaler', StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ]
        )

        X = pd.DataFrame(preprocessor.fit_transform(df3),
                         columns=X.columns.tolist())

        return X, y
