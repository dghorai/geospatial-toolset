import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging, project_dir
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            # get file path
            preprocessor_file_path = os.path.join(
                project_dir, "artifacts", "preprocessor.pkl")
            model_file_path = os.path.join(
                project_dir, "artifacts", "model.pkl")

            # create object
            preprocessor = load_object(preprocessor_file_path)
            model = load_object(model_file_path)

            # transformed data
            data_scaled = preprocessor.transform(features)

            # prediction
            y_pred = model.predict(data_scaled)

            return y_pred
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.p1 = kwargs['value1']
        self.p2 = kwargs['value2']
        self.p3 = kwargs['value3']
        self.p4 = kwargs['value4']
        self.p5 = kwargs['value5']
        self.p6 = kwargs['value6']
        self.p7 = kwargs['value7']
        self.p8 = kwargs['value8']
        self.p9 = kwargs['value9']
        # const
        self.CRIM_LOWER = -5.31051125
        self.CRIM_UPPER = 9.06963875
        self.ZN_LOWER = -18.75
        self.ZN_UPPER = 31.25
        self.B_LOWER = 344.10624999999993
        self.B_UPPER = 427.49625000000003

    def format_user_data(self, user_data_dict):
        # create a df
        df = pd.DataFrame(user_data_dict)
        df = df[['CRIM', 'ZN', 'INDUS', 'CHAS',
                 'AGE', 'DIS', 'RAD', 'B', 'LSTAT']]
        # feature clipping
        for col in ['CRIM', 'ZN', 'B']:
            if col == 'CRIM':
                df[col] = df[col].clip(self.CRIM_LOWER, self.CRIM_UPPER)

            if col == 'ZN':
                df[col] = df[col].clip(self.ZN_LOWER, self.ZN_UPPER)

            if col == 'B':
                df[col] = df[col].clip(self.B_LOWER, self.B_UPPER)

        return df

    def get_user_inputs(self):
        try:
            user_data_dict = {
                'CRIM': [self.p1],
                'ZN': [self.p2],
                'INDUS': [self.p3],
                'CHAS': [self.p4],
                'AGE': [self.p5],
                'DIS': [self.p6],
                'RAD': [self.p7],
                'B': [self.p8],
                'LSTAT': [self.p9]
            }
            # create a df
            df = self.format_user_data(user_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)
