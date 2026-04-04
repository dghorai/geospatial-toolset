import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging, PROJECT_DIR
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join(
                PROJECT_DIR, 'artifacts', 'preprocessor.pkl')
            model_path = os.path.join(PROJECT_DIR, 'artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.crim = kwargs['crim']
        self.zn = kwargs['zn']
        self.indus = kwargs['indus']
        self.chas = kwargs['chas']
        self.nox = kwargs['nox']
        self.rm = kwargs['rm']
        self.age = kwargs['age']
        self.dis = kwargs['dis']
        self.rad = kwargs['rad']
        self.tax = kwargs['tax']
        self.ptratio = kwargs['ptratio']
        self.b = kwargs['b']
        self.lstat = kwargs['lstat']

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CRIM': [self.crim],
                'ZN': [self.zn],
                'INDUS': [self.indus],
                'CHAS': [self.chas],
                'NOX': [self.nox],
                'RM': [self.rm],
                'AGE': [self.age],
                'DIS': [self.dis],
                'RAD': [self.rad],
                'TAX': [self.tax],
                'PTRATIO': [self.ptratio],
                'B': [self.b],
                'LSTAT': [self.lstat]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)
