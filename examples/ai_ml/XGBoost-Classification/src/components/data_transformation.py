import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

from src.exception import CustomException
from src.logger import logging, project_dir
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_filepath = os.path.join(
        project_dir, "artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')

            # define columns and their types
            categorical_cols = ['workclass', 'education', 'marital_status',
                                'occupation', 'relationship', 'race', 'sex', 'native_country']
            numerical_cols = ['age', 'fnlwgt', 'education_num',
                              'capital_gain', 'capital_loss', 'hours_per_week']

            logging.info('Pipeline initiated')

            # Define the custom ranking for each ordinal variable
            workclass_categories = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
                                    'Local-gov', 'unknown', 'Self-emp-inc', 'Without-pay', 'Never-worked']
            education_categories = ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-acdm',
                                    'Assoc-voc', 'Doctorate', 'Prof-school', 'Preschool', 'Primary Education']
            marital_status_categories = ['Never-married', 'Married-civ-spouse', 'Divorced',
                                         'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
            occupation_categories = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                                     'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'unknown', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
            relationship_categories = [
                'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']
            race_categories = [
                'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
            sex_categories = ['Male', 'Female']
            native_country_categories = [
                'North America', 'Central America', 'Asia', 'unknown', 'South America', 'European Union']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[workclass_categories, education_categories, marital_status_categories,
                     occupation_categories, relationship_categories, race_categories, sex_categories, native_country_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            # preprocessor
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline completed')
            return preprocessor
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_dp, test_dp):
        try:
            # read train and test data
            train_df = pd.read_csv(train_dp)
            test_df = pd.read_csv(test_dp)

            logging.info('Read train and test data completed')
            logging.info(f'Train DF Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DF Head: \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessing_object = self.get_data_transformation_object()

            target_column_name = 'wage_class_id'
            drop_columns = [target_column_name]

            # train df
            input_feat_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feat_train_df = train_df[target_column_name]

            # test df
            input_feat_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feat_test_df = test_df[target_column_name]

            # transforming using preprocessor object
            train_arr = preprocessing_object.fit_transform(input_feat_train_df)
            test_arr = preprocessing_object.transform(input_feat_test_df)

            logging.info(
                'Applying preprocessing object on training and test datasets')

            train_arrc = np.c_[train_arr, np.array(target_feat_train_df)]
            test_arrc = np.c_[test_arr, np.array(target_feat_test_df)]

            # save transformed data
            file_path = self.data_transformation_config.preprocessor_filepath
            save_object(
                file_path=file_path,
                obj=preprocessing_object
            )
            logging.info('Preprocess pickle file saved')
            return train_arrc, test_arrc, file_path
        except Exception as e:
            logging.info(
                "Exception occured in the initiate_data_transformation")
            raise CustomException(e, sys)
