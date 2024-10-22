import os
import sys


import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd

# from patsy import dmatrix  # one-hot-encoding
from logger import logging, CustomException
from classification.xgboost.utils import load_object, prepare_data, feature_clipping
from config import PRJ_DIR


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            # get file path
            preprocessor_file_path = os.path.join(PRJ_DIR, "artifacts", "classification", "xgboost", "preprocessor.pkl")
            model_file_path = os.path.join(PRJ_DIR, "artifacts", "classification", "xgboost", "model.pkl")

            # create object
            preprocessor = load_object(preprocessor_file_path)
            model = load_object(model_file_path)

            # transformed data
            data_scaled = preprocessor.transform(features)
            # prediction
            y_pred = model.predict(data_scaled)

            if y_pred[0] == 0:
                res = "person makes below 50K per year"
            else:
                res = "person makes over 50K per year"

            return res
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.p1 = kwargs['age']
        self.p2 = kwargs['workclass']
        self.p3 = kwargs['fnlwgt']
        self.p4 = kwargs['education']
        self.p5 = kwargs['education_num']
        self.p6 = kwargs['marital_status']
        self.p7 = kwargs['occupation']
        self.p8 = kwargs['relationship']
        self.p9 = kwargs['race']
        self.p10 = kwargs['sex']
        self.p11 = kwargs['capital_gain']
        self.p12 = kwargs['capital_loss']
        self.p13 = kwargs['hours_per_week']
        self.p14 = kwargs['native_country']

    def format_user_data(self, user_data_dict):
        # create a df
        df = pd.DataFrame(user_data_dict)
        fdf = prepare_data(df)
        train_df = pd.read_csv(os.path.join(PRJ_DIR, 'notebooks', 'data', 'wage_train_set.csv'), skipinitialspace=True)
        train_df = prepare_data(train_df)
        lower, upper = feature_clipping(train_df, col='fnlwgt')
        fdf['fnlwgt'] = fdf['fnlwgt'].clip(lower, upper)
        return fdf

    def get_user_inputs(self):
        try:
            user_data_dict = {
                'age': [self.p1],
                'workclass': [self.p2],
                'fnlwgt': [self.p3],
                'education': [self.p4],
                'education_num': [self.p5],
                'marital_status': [self.p6],
                'occupation': [self.p7],
                'relationship': [self.p8],
                'race': [self.p9],
                'sex': [self.p10],
                'capital_gain': [self.p11],
                'capital_loss': [self.p12],
                'hours_per_week': [self.p13],
                'native_country': [self.p14]
            }
            # create a df
            df = self.format_user_data(user_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)
