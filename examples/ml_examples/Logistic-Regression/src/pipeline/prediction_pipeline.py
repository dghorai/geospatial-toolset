import os
import sys
import pandas as pd

from patsy import dmatrix

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

            if y_pred == 1.0:
                results = 'affair'
            else:
                results = 'no affair'
            return results
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.p1 = kwargs['rate_marriage']
        self.p2 = kwargs['age']
        self.p3 = kwargs['yrs_married']
        self.p4 = kwargs['children']
        self.p5 = kwargs['religious']
        self.p6 = kwargs['educ']
        self.p7 = kwargs['occupation']
        self.p8 = kwargs['occupation_husb']

    def format_user_data(self, user_data_dict):
        # create a df
        df = pd.DataFrame(user_data_dict)

        # one-hot-encoding
        scale_data = {"occupation": [1, 2, 3, 4, 5, 6],
                      "occupation_husb": [1, 2, 3, 4, 5, 6]}
        ohe = dmatrix('C(occupation) + C(occupation_husb)',
                      scale_data, return_type="dataframe")
        ohe = ohe.rename(columns={
            'C(occupation)[T.2]': 'occ_2',
            'C(occupation)[T.3]': 'occ_3',
            'C(occupation)[T.4]': 'occ_4',
            'C(occupation)[T.5]': 'occ_5',
            'C(occupation)[T.6]': 'occ_6',
            'C(occupation_husb)[T.2]': 'occ_husb_2',
            'C(occupation_husb)[T.3]': 'occ_husb_3',
            'C(occupation_husb)[T.4]': 'occ_husb_4',
            'C(occupation_husb)[T.5]': 'occ_husb_5',
            'C(occupation_husb)[T.6]': 'occ_husb_6'
        }
        )
        ohe_occupation = ohe[['occ_2', 'occ_3', 'occ_4',
                              'occ_5', 'occ_6']][ohe.index == user_data_dict['occupation'][0]-1]
        ohe_occupation.reset_index(drop=True, inplace=True)
        ohe_occupation_husb = ohe[['occ_husb_2', 'occ_husb_3', 'occ_husb_4', 'occ_husb_5',
                                   'occ_husb_6']][ohe.index == user_data_dict['occupation_husb'][0]-1]
        ohe_occupation_husb.reset_index(drop=True, inplace=True)
        df_numerical = df[['rate_marriage', 'age',
                           'yrs_married', 'children', 'religious', 'educ']]
        df_numerical.reset_index(drop=True, inplace=True)
        fdf = pd.concat(
            [ohe_occupation, ohe_occupation_husb, df_numerical], axis=1)
        return fdf

    def get_user_inputs(self):
        try:
            user_data_dict = {
                'rate_marriage': [self.p1],
                'age': [self.p2],
                'yrs_married': [self.p3],
                'children': [self.p4],
                'religious': [self.p5],
                'educ': [self.p6],
                'occupation': [self.p7],
                'occupation_husb': [self.p8]
            }
            # create a df
            df = self.format_user_data(user_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)
