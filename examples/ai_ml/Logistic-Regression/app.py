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
            'rate_marriage': float(request.form['rate_marriage']),
            'age': float(request.form['age']),
            'yrs_married': float(request.form['yrs_married']),
            'children': float(request.form['children']),
            'religious': float(request.form['religious']),
            'educ': float(request.form['educ']),
            'occupation': float(request.form['occupation']),
            'occupation_husb': float(request.form['occupation_husb'])
        }
        user_data = CustomData(**user_inputs)
        y_test = user_data.get_user_inputs()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(y_test)
        return render_template('results.html', final_result=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
