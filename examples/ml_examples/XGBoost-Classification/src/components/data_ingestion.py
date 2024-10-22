import os
import sys
import pandas as pd
# import numpy as np

# from patsy import dmatrices
# from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging, project_dir
from src.exception import CustomException
from src.utils import prepare_data, feature_clipping


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

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')

        try:
            train_df = pd.read_csv(os.path.join(
                project_dir, 'notebooks', 'data', 'train_set.csv'), skipinitialspace=True)
            test_df = pd.read_csv(os.path.join(
                project_dir, 'notebooks', 'data', 'test_set.csv'), skipinitialspace=True)
            logging.info('Dataset read as pandas dataframe')

            logging.info('Clean and format initial data')

            train_df = prepare_data(train_df, is_train=True)
            test_df = prepare_data(test_df, is_train=True)

            # feature clipping
            lower, upper = feature_clipping(train_df, col='fnlwgt')
            train_df['fnlwgt'] = train_df['fnlwgt'].clip(lower, upper)
            test_df['fnlwgt'] = test_df['fnlwgt'].clip(lower, upper)

            # concatenate dfs
            df = pd.concat([train_df, test_df], axis=0)

            # define paths
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
            # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set, test_set = train_df, test_df

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
