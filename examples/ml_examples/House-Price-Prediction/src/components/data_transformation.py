import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging, PROJECT_DIR
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join(PROJECT_DIR, 'artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = []
            numerical_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                              'RM', 'AGE', 'RAD', 'PTRATIO', 'B', 'LSTAT']

            # Does not required to define the custom ranking for each ordinal variable as in this data there is no categorical variable
            logging.info('Pipeline initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())

                ]
            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[])),
                    ('scaler', StandardScaler())
                ]
            )

            # Preprocessor
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline completed')
            return preprocessor
        except Exception as e:
            logging.info("Error in data trnasformation")
            raise CustomException(e, sys)

    def initaite_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(
                f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(
                f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'MEDV'
            drop_columns = [target_column_name, 'NOX', 'DIS', 'TAX']

            # Train DF
            input_feature_train_df = train_df.drop(
                columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Test DF
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Trnasformating using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            logging.info(
                "Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            # Save transformed train/test data
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_filepath
        except Exception as e:
            logging.info(
                "Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)
