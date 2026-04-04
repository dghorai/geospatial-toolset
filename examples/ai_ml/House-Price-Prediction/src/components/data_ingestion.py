import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging, PROJECT_DIR
from src.exception import CustomException


# Intitialize the Data Ingetion Configuration
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(PROJECT_DIR, 'artifacts', 'raw.csv')
    train_data_path: str = os.path.join(PROJECT_DIR, 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(PROJECT_DIR, 'artifacts', 'test.csv')


# create a class for Data Ingestion
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')

        try:
            df = pd.read_csv(os.path.join(PROJECT_DIR, 'notebooks', 'data', 'boston.csv'))
            logging.info('Dataset read as pandas data-frame')

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)

            if os.path.exists(self.ingestion_config.raw_data_path):
                logging.info("Raw data file already exists")
            else:
                df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train test split')
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            if os.path.exists(self.ingestion_config.train_data_path):
                logging.info("Training data file already exists")
            else:
                train_set.to_csv(
                    self.ingestion_config.train_data_path, index=False, header=True)

            if os.path.exists(self.ingestion_config.test_data_path):
                logging.info("Test data file already exists")
            else:
                test_set.to_csv(self.ingestion_config.test_data_path,
                                index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            logging.info('Exception occured at data-ingestion stage')
            raise CustomException(e, sys)
