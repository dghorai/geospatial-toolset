import os
import sys
import pandas as pd

from scipy import stats

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

            # display result
            if y_pred == 1.0:
                results = 'Titanic passenger has survived'
            else:
                results = 'Titanic passenger has not survived'
            return results
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, **kwargs):
        self.p1 = kwargs['Pclass']
        self.p2 = kwargs['Sex']
        self.p3 = kwargs['Age']
        self.p4 = kwargs['SibSp']
        self.p5 = kwargs['Parch']
        self.p6 = kwargs['Fare']
        self.original_data = os.path.join(
            project_dir, 'notebooks', 'data', 'titanic-train.csv')

    def format_user_data(self, user_data_dict):
        # find the best_lambda from original data
        org_df = pd.read_csv(self.original_data)
        org_df = org_df[['Fare']]
        org_df.loc[org_df['Fare'] >= 100, 'Fare'] = 100
        _, fitted_lambda = stats.boxcox(org_df['Fare'] + 0.5)
        logging.info(f'best box-cox-transformation lambda: {fitted_lambda}')

        # create a df
        df = pd.DataFrame(user_data_dict)
        logging.info(f'User Data Original: \n{df}')

        # replace outliers from Fare column
        df.loc[df['Fare'] >= 100, 'Fare'] = 100

        # box-cox-transformation (https://www.statology.org/box-cox-transformation-python/)
        df['Fare'] = (df['Fare']**fitted_lambda - 1) / fitted_lambda
        logging.info(f'User Data Transformed: \n{df}')

        return df

    def get_user_inputs(self):
        try:
            user_data_dict = {
                'Pclass': [self.p1],
                'Sex': [self.p2],
                'Age': [self.p3],
                'SibSp': [self.p4],
                'Parch': [self.p5],
                'Fare': [self.p6]
            }
            # create a df
            df = self.format_user_data(user_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e, sys)
