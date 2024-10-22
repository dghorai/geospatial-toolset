# import os
import sys
import numpy as np
import pandas as pd
import warnings

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import feature_engineering

warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass
class DataTransformationConfig:
    final_columns = ['Married', 'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
                     'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_data_array(self, featdict, x_feature, y_feature):
        X_train1 = pd.DataFrame(featdict['X_train'])
        X_train1.columns = x_feature
        X_train2 = pd.DataFrame(featdict['X_test'])
        X_train2.columns = x_feature
        y_train1 = pd.DataFrame(featdict['y_train'])
        y_train1.columns = y_feature
        y_train2 = pd.DataFrame(featdict['y_test'])
        y_train2.columns = y_feature

        X_train = pd.concat([X_train1, X_train2], axis=0).to_numpy()
        y_train = pd.concat([y_train1, y_train2], axis=0).to_numpy()

        train_arrc = np.c_[X_train, y_train]
        return train_arrc

    def initiate_data_transformation(self, train_dp, test_dp):
        try:
            # read train and test data
            train_df = pd.read_csv(train_dp)
            test_df = pd.read_csv(test_dp)
            features = self.data_transformation_config.final_columns

            logging.info('Read train and test data completed')
            logging.info(f'Train DF Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DF Head: \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            target_feature = [features[-1]]
            train_feature = features[:-1]
            train_df = train_df[features]
            test_df = test_df[features]

            train_xysplit_cs = feature_engineering(
                train_df, colslist=features, return_type='CustomScaler')
            test_xysplit_cs = feature_engineering(
                test_df, colslist=features, return_type='CustomScaler')

            train_xysplit_ss = feature_engineering(
                train_df, colslist=features, return_type='StandardScaler')
            test_xysplit_ss = feature_engineering(
                test_df, colslist=features, return_type='StandardScaler')

            logging.info('Concatenate numpy array')
            train_arrc_cs = self.create_data_array(
                train_xysplit_cs, train_feature, target_feature)
            test_arrc_cs = self.create_data_array(
                test_xysplit_cs, train_feature, target_feature)

            train_arrc_ss = self.create_data_array(
                train_xysplit_ss, train_feature, target_feature)
            test_arrc_ss = self.create_data_array(
                test_xysplit_ss, train_feature, target_feature)

            return train_arrc_cs, test_arrc_cs, train_arrc_ss, test_arrc_ss
        except Exception as e:
            logging.info(
                "Exception occured in the initiate_data_transformation")
            raise CustomException(e, sys)
