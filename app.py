
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the logistic regression model
model = joblib.load('logistic_regression_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = request.form.get('age')
    sex = request.form.get('sex')
    chol = request.form.get('chol')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')

    # Create DataFrame for model input
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chol': [chol],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Create a response message
    message = "You might have heart disease" if prediction == 1 else "You might not have heart disease"

    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
