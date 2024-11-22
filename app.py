# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

Flask Application - Manage PyGeoML Projects.
"""

import os
import sys
sys.path.append('src')

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from pathlib import Path

from ml_models.prediction_pipeline import CustomData, PredictPipeline
from classification.xgboost.pipeline.prediction_pipeline import CustomData, PredictPipeline
# from classification.cnn_classifier.pipeline.predict import PredictionPipeline
# from classification.cnn_classifier.utils.utilities import decode_image
# from config import PRJ_DIR


# app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
CORS(app)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# config
DEG_TO_KM = 111.0  # https://education.nationalgeographic.org/resource/latitude/
PRJ_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = Path("config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")


# class ClientApp:
#     def __init__(self):
#         self.filename = os.path.join(PRJ_DIR, "artifacts", "classification", "cnn_classifier", "test_data", "input_image.jpg")
#         self.classifier = PredictionPipeline(self.filename)


# define the home page of the site
@app.route('/')  # this sets the route to this page
def home():
    data = {}
    return render_template('index.html', data=data)


@app.route('/regression')
def regression_home_page():
    return render_template('/regression/random_forest/index.html')


@app.route('/regression/predict1', methods=['GET', 'POST'])
def regression_predict_user_data():
    if request.method == 'GET':
        return render_template('/regression/random_forest/form.html')
    else:
        user_inputs = {
            'value1': float(request.form['crim']),
            'value2': float(request.form['zn']),
            'value3': float(request.form['indus']),
            'value4': float(request.form['chas']),
            'value5': float(request.form['age']),
            'value6': float(request.form['dis']),
            'value7': float(request.form['rad']),
            'value8': float(request.form['b']),
            'value9': float(request.form['lstat'])
        }
        user_data = CustomData(**user_inputs)
        y_test = user_data.get_user_inputs()
        predict_pipeline = PredictPipeline()
        res = predict_pipeline.predict(y_test)
        results = round(res[0], 2)
        return render_template('/regression/random_forest/results.html', final_result=results)


        
@app.route('/classification')
def classification_home_page():
    return render_template('/classification/xgboost/index.html')


@app.route('/classification/predict1', methods=['GET', 'POST'])
def classification_predict_user_data():
    if request.method == 'GET':
        return render_template('/classification/xgboost/form.html')
    else:
        user_inputs = {
            'age': float(request.form['age']),
            'workclass': str(request.form['workclass']),
            'fnlwgt': float(request.form['fnlwgt']),
            'education': str(request.form['education']),
            'education_num': float(request.form['education_num']),
            'marital_status': str(request.form['marital_status']),
            'occupation': str(request.form['occupation']),
            'relationship': str(request.form['relationship']),
            'race': str(request.form['race']),
            'sex': str(request.form['sex']),
            'capital_gain': float(request.form['capital_gain']),
            'capital_loss': float(request.form['capital_loss']),
            'hours_per_week': float(request.form['hours_per_week']),
            'native_country': str(request.form['native_country'])
        }
        user_data = CustomData(**user_inputs)
        y_test = user_data.get_user_inputs()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(y_test)
        return render_template('/classification/xgboost/results.html', final_result=results)
    

# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('/classification/cnn_classifier/index.html')

# @app.route("/train", methods=['GET', 'POST'])
# @cross_origin()
# def trainRoute():
#     os.system("python main.py")
#     return "Training done successfully!"

# @app.route("/predict", methods=['POST'])
# @cross_origin()
# def predictRoute():
#     image = request.json['image']
#     decode_image(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)
# # call client app
# clApp = ClientApp()
                    


if __name__ == '__main__':
    app.run(debug=True)  # "debug=True": allows possible Python errors to appear on the web page.
