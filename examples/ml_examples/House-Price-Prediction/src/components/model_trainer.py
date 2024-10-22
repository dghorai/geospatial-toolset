import os
import sys

from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging, PROJECT_DIR
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join(PROJECT_DIR, 'artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info(
                'Splitting dependent and independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression()
            }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)
            print(model_report)
            print(
                '\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get best model score and name from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(
                f'Best Model Found -> Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print(
                '\n====================================================================================\n')
            logging.info(
                f'Best Model Found -> Model Name : {best_model_name}, R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
