import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained XGBoost model
model_path = os.path.join(os.path.dirname(__file__), 'churning_customer.pkl')
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("WARNING: Model failed to load. Are you using the virtual environment? Ensure 'xgboost' is installed.")
    model = None

# Feature names exactly as expected by the model
FEATURE_NAMES = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 
    'TotalCharges', 'gender_Female', 'gender_Male', 'MultipleLines_No', 
    'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 
    'InternetService_Fiber optic', 'InternetService_No', 'Contract_Month-to-month', 
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check', 'ServicesUsed', 'HeavyUser'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded properly.'}), 500

    try:
        data = request.json
        
        # Initialize input dictionary with 0s
        input_data = {col: 0 for col in FEATURE_NAMES}

        # ------------------------------------------
        # NUMERICAL INPUTS
        # ------------------------------------------
        tenure = float(data.get('tenure', 0))
        input_data['tenure'] = tenure / 72.0

        monthly = float(data.get('MonthlyCharges', 0))
        input_data['MonthlyCharges'] = (monthly - 18.25) / (118.75 - 18.25)

        total = float(data.get('TotalCharges', 0))
        input_data['TotalCharges'] = (total - 18.8) / (8684.8 - 18.8)

        # ------------------------------------------
        # BINARY INPUTS
        # ------------------------------------------
        binary_cols = [
            'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
        ]
        for col in binary_cols:
            input_data[col] = int(data.get(col, 0))

        # ------------------------------------------
        # GENDER
        # ------------------------------------------
        gender = data.get('gender', '')
        if gender == 'Male':
            input_data['gender_Male'] = 1
        elif gender == 'Female':
            input_data['gender_Female'] = 1

        # ------------------------------------------
        # MULTIPLE LINES
        # ------------------------------------------
        multiple = data.get('MultipleLines', '')
        if multiple == 'No':
            input_data['MultipleLines_No'] = 1
        elif multiple == 'No phone service':
            input_data['MultipleLines_No phone service'] = 1
        elif multiple == 'Yes':
            input_data['MultipleLines_Yes'] = 1

        # ------------------------------------------
        # INTERNET SERVICE
        # ------------------------------------------
        internet = data.get('InternetService', '')
        if internet == 'DSL':
            input_data['InternetService_DSL'] = 1
        elif internet == 'Fiber optic':
            input_data['InternetService_Fiber optic'] = 1
        elif internet == 'No':
            input_data['InternetService_No'] = 1

        # ------------------------------------------
        # CONTRACT
        # ------------------------------------------
        contract = data.get('Contract', '')
        if contract == 'Month-to-month':
            input_data['Contract_Month-to-month'] = 1
        elif contract == 'One year':
            input_data['Contract_One year'] = 1
        elif contract == 'Two year':
            input_data['Contract_Two year'] = 1

        # ------------------------------------------
        # PAYMENT METHOD
        # ------------------------------------------
        payment = data.get('PaymentMethod', '')
        if payment == 'Bank transfer (automatic)':
            input_data['PaymentMethod_Bank transfer (automatic)'] = 1
        elif payment == 'Credit card (automatic)':
            input_data['PaymentMethod_Credit card (automatic)'] = 1
        elif payment == 'Electronic check':
            input_data['PaymentMethod_Electronic check'] = 1
        elif payment == 'Mailed check':
            input_data['PaymentMethod_Mailed check'] = 1

        # ------------------------------------------
        # FEATURE ENGINEERING
        # ------------------------------------------
        services_used = (
            input_data['PhoneService'] +
            input_data['OnlineSecurity'] +
            input_data['OnlineBackup'] +
            input_data['DeviceProtection'] +
            input_data['TechSupport'] +
            input_data['StreamingTV'] +
            input_data['StreamingMovies']
        )
        input_data['ServicesUsed'] = services_used
        input_data['HeavyUser'] = 1 if services_used >= 3 else 0

        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches exactly what the model expects
        input_df = input_df[FEATURE_NAMES]

        # Predict
        prob = model.predict_proba(input_df)
        churn_prob = float(prob[0][1]) * 100

        # Risk Analysis
        if churn_prob >= 70:
            category = "High Churn Risk"
            risk_level = "high"
        elif churn_prob >= 30:
            category = "Moderate Churn Risk"
            risk_level = "medium"
        else:
            category = "Low Churn Risk"
            risk_level = "low"

        return jsonify({
            'success': True,
            'probability': churn_prob,
            'category': category,
            'risk_level': risk_level
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
