# run flask application
from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_user_data():
    if request.method == 'GET':
        return render_template('form.html')
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
        return render_template('results.html', final_result=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
