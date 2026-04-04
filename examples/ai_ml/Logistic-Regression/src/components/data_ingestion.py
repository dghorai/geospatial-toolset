import os
import sys
import pandas as pd
import numpy as np

from patsy import dmatrices
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
        # Add "affair" column: 1 -> represents having affairs, 0 -> represents not having affairs
        df['affair'] = (df.affairs > 0).astype(
            int)  # converting into binary features

        y, X = dmatrices(
            'affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', df, return_type="dataframe")

        X = X.rename(columns={
            'C(occupation)[T.2.0]': 'occ_2',
            'C(occupation)[T.3.0]': 'occ_3',
            'C(occupation)[T.4.0]': 'occ_4',
            'C(occupation)[T.5.0]': 'occ_5',
            'C(occupation)[T.6.0]': 'occ_6',
            'C(occupation_husb)[T.2.0]': 'occ_husb_2',
            'C(occupation_husb)[T.3.0]': 'occ_husb_3',
            'C(occupation_husb)[T.4.0]': 'occ_husb_4',
            'C(occupation_husb)[T.5.0]': 'occ_husb_5',
            'C(occupation_husb)[T.6.0]': 'occ_husb_6'
        }
        )
        # concatenate X and y
        new_df = pd.concat([X, y], axis=1)

        return new_df

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')

        try:
            df = pd.read_csv(os.path.join(
                project_dir, 'notebooks', 'data', 'affair.csv'))
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
