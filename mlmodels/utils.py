import os
import sys
import pickle
import yaml
import json
import joblib
import base64

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# !pip install python-box==4.0.3
# !pip install python-box==4.0.3
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from typing import Any
from pathlib import Path
from logger import logging, CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"json file saved at: {path}")
    return


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    logging.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_model(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logging.info(f"binary file saved at: {path}")
    return


@ensure_annotations
def load_model(path: Path) -> Any:
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
    return


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_object function - utils")
        raise CustomException(e, sys)
    return


@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def create_directory(path_to_directory: list, verbose=True):
    for path in path_to_directory:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"create directory at: {path}")
    return


def decode_image(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()
    return


def encode_image_into_base64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    return
    

def evaluate_model(X_train, y_train, X_test, y_test, models, ml_type=None):
    try:
        report = {}
        if ml_type == 'regression':
            for i in range(len(models)):
                model = list(models.values())[i]
                # Train model
                model.fit(X_train, y_train)
                # Predict Testing data
                y_test_pred = model.predict(X_test)
                # Get R2 scores for train and test data
                test_model_score = r2_score(y_test, y_test_pred)
                report[list(models.keys())[i]] = test_model_score
        elif ml_type == 'classification':
            pass
        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)
