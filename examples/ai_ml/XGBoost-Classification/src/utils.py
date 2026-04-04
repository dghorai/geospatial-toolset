import os
import sys
import pickle
# import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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


def primary_education(x):
    # education variable has several sub-class that can be grouped to make an one sub-class
    if x in ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'Primary Education'
    else:
        return x


def geography_location(country):
    if country in ['United-States', 'Canada']:
        return 'North America'
    elif country in ['Puerto-Rico', 'El-Salvador', 'Cuba', 'Jamaica', 'Dominican-Republic', 'Guatemala', 'Haiti', 'Nicaragua', 'Trinadad&Tobago', 'Honduras']:
        return 'Central America'
    elif country in ['Mexico', 'Columbia', 'Vietnam', 'Peru', 'Ecuador', 'South', 'Outlying-US(Guam-USVI-etc)']:
        return 'South America'
    elif country in ['Germany', 'England', 'Italy', 'Poland', 'Portugal', 'Greece', 'Yugoslavia', 'France', 'Ireland', 'Scotland', 'Hungary', 'Holand-Netherlands']:
        return 'European Union'
    elif country in ['India', 'Iran', 'China', 'Japan', 'Thailand', 'Hong', 'Cambodia', 'Laos', 'Philippines', 'Taiwan']:
        return 'Asia'
    else:
        return country


def feature_clipping(df, col: str = None):
    # feature clipping for outlier variables
    df2 = df.copy()

    # IQR
    Q1 = df2[col].quantile(0.25)
    Q3 = df2[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    return lower, upper


def prepare_data(df, is_train=False):
    # replacing '?' with 'unknown' as value
    df.replace(' ?', 'unknown', inplace=True)
    df.replace('?', 'unknown', inplace=True)

    if is_train:
        # rectify string in target variable
        df.wage_class.replace('<=50K.', '<=50K', inplace=True)
        df.wage_class.replace('>50K.', '>50K', inplace=True)

        # add numerical identity to target class
        df['wage_class_id'] = df['wage_class'].replace({'<=50K': 0, '>50K': 1})
        df.drop(['wage_class'], axis=1, inplace=True)

    # group education variable
    df['education'] = df['education'].apply(primary_education)

    # native_country variable has many sub-class (country) and this can be group into few sub-class based on their geographical position.
    df['native_country'] = df['native_country'].apply(geography_location)

    return df


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get accuracy_score for test data
            test_model_score = accuracy_score(y_test, y_test_pred)

            # get classification report for test data
            test_classification_report = classification_report(
                y_test, y_test_pred)

            # get confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)

            res = {'accuracy_score': test_model_score,
                   'classification_report': test_classification_report,
                   'confusion_matrix': cm}

            report[list(models.keys())[i]] = res

        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)
