import os
import sys
import pandas as pd
import numpy as np

# from patsy import dmatrices
from scipy import stats
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging, project_dir
from src.exception import CustomException


# initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join(project_dir, "artifacts", "raw.csv")
    train_data_path = os.path.join(project_dir, "artifacts", "train.csv")
    test_data_path = os.path.join(project_dir, "artifacts", "test.csv")


# create a class for data ingestion
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def prepare_data(self, df):
        # PassengerId is just a numerical number and not related to survival prediction and
        # it does not contribute to a persons survival probability. Similarly, Name, Ticket, Cabin,
        # and Embarked fields are non relevant to survival prediction of a person.

        cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
        new_df = df[cols]  # clean and format data
        # replace outliers from Fare column
        new_df.loc[new_df['Fare'] >= 100, 'Fare'] = 100
        # transform Fare column to convert to normal distribution
        new_df['Fare'], _ = stats.boxcox(new_df['Fare'] + 0.5)

        return new_df

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')

        try:
            df = pd.read_csv(os.path.join(
                project_dir, 'notebooks', 'data', 'titanic-train.csv'))
            logging.info('Dataset read as pandas dataframe')

            logging.info('Clean and format initial data')
            df = self.prepare_data(df)

            raw_dp = self.ingestion_config.raw_data_path
            train_dp = self.ingestion_config.train_data_path
            test_dp = self.ingestion_config.test_data_path
            artifacts_dir = os.path.dirname(raw_dp)
            os.makedirs(artifacts_dir, exist_ok=True)

            if os.path.exists(raw_dp):
                logging.info('Raw data file already exists')
            else:
                df.to_csv(raw_dp, index=False)

            logging.info('Train test split')
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            if os.path.exists(train_dp):
                logging.info('Training data file already exists')
            else:
                train_set.to_csv(train_dp, index=False, header=True)

            if os.path.exists(test_dp):
                logging.info('Test data file already exists')
            else:
                test_set.to_csv(test_dp, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return train_dp, test_dp
        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e, sys)
