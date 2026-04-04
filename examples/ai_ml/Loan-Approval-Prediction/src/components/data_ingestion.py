import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging, project_dir
from src.exception import CustomException
from src.utils import input_dataframe, feature_engineering


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
            logging.info('Dataset read as pandas dataframe')
            df = input_dataframe(os.path.join(
                project_dir, 'notebooks', 'data', 'input_training_data.csv'))
            cols = df.columns.tolist()[1:]
            # print(cols)
            df2 = feature_engineering(df, colslist=cols, return_type='XY')
            xdf = pd.DataFrame(df2['X'])
            xdf.columns = cols[:-1]
            ydf = pd.DataFrame(df2['y'])
            ydf.columns = [cols[-1]]
            df3 = pd.concat([xdf, ydf], axis=1)
            # print(df3)

            # define datapaths
            raw_dp = self.ingestion_config.raw_data_path
            train_dp = self.ingestion_config.train_data_path
            test_dp = self.ingestion_config.test_data_path
            artifacts_dir = os.path.dirname(raw_dp)
            os.makedirs(artifacts_dir, exist_ok=True)

            if os.path.exists(raw_dp):
                logging.info('Raw data file already exists')
            else:
                df3.to_csv(raw_dp, index=False)

            logging.info('Train test split')
            train_set, test_set = train_test_split(
                df3, test_size=0.2, random_state=1)

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
