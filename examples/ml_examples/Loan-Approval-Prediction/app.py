# run flask application
from flask import Flask, request, render_template
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
            'Married': str(request.form['Married']),
            'Dependents': str(request.form['Dependents']),
            'Education': str(request.form['Education']),
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': int(request.form['Loan_Amount_Term']),
            'Credit_History': int(request.form['Credit_History']),
            'Property_Area': str(request.form['Property_Area'])
        }
        user_data = CustomData(**user_inputs)
        x_test_custom, x_test_standard = user_data.get_user_inputs()
        print(x_test_custom)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(x_test_custom, x_test_standard)
        return render_template('results.html', final_result=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
