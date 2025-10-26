from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model/best_model.joblib")
scaler = joblib.load("model/scaler.joblib")

# Mapping numeric prediction to text
label_mapping = {0: "Healthy", 1: "Pre-Diabetic", 2: "Diabetic"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        gender = int(request.form['gender'])
        smoking_history = int(request.form['smoking_history'])
        hba1c = float(request.form['HbA1c_level'])
        blood_glucose = float(request.form['blood_glucose_level'])

        # Prepare data for model
        input_data = np.array([[age, bmi, hypertension, heart_disease,
                                gender, smoking_history, hba1c, blood_glucose]])

        # Scale data
        if scaler:
            input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)[0]

        # Map numeric prediction to label
        result = label_mapping.get(prediction, "Unknown")

        # Prepare inputs dictionary to display in result.html
        user_inputs = {
            "Age": age,
            "BMI": bmi,
            "Hypertension": "Yes" if hypertension == 1 else "No",
            "Heart Disease": "Yes" if heart_disease == 1 else "No",
            "Gender": "Male" if gender == 1 else "Female",
            "Smoking History": {0: "Never", 1: "Former", 2: "Current"}.get(smoking_history, "Unknown"),
            "HbA1c Level": hba1c,
            "Blood Glucose Level": blood_glucose
        }

        return render_template('result.html', result=result, inputs=user_inputs)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
