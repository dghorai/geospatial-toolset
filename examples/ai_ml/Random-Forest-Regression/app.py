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
        return render_template('results.html', final_result=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
