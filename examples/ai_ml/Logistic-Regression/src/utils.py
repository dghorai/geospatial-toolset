import os
import sys
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_object function - utils")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # train model
            model.fit(X_train, y_train)

            # predict on test data
            y_pred = model.predict(X_test)

            # get accuracy score for test data
            # test_accuracy = accuracy_score(y_test, y_pred)

            # get classification report for test data
            test_classification_report = classification_report(
                y_test, y_pred)

            report[list(models.keys())[i]] = test_classification_report

        return report
    except Exception as e:
        logging.info(
            'Exception occured during model training/model evaluation')
        raise CustomException(e, sys)
