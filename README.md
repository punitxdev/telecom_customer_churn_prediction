# Telecom Customer Churn Prediction

## Project Overview

Telecom Customer Churn Prediction is a machine learning web application that estimates the probability that a telecom customer will leave the service. The project combines a trained XGBoost classifier with a Flask web application, allowing users to enter customer profile details and receive an immediate churn-risk prediction.

The application is designed for customer retention analysis. It converts demographic, account, billing, and service subscription information into the exact feature format expected by the trained model, then returns a churn probability and a practical risk category.

## Key Features

- Predicts customer churn probability using a trained XGBoost classifier.
- Provides three risk levels:
  - Low Churn Risk: less than 30 percent probability
  - Moderate Churn Risk: 30 percent to 69 percent probability
  - High Churn Risk: 70 percent probability or higher
- Includes a Flask backend with a `/predict` JSON endpoint.
- Provides a responsive web interface for entering customer details.
- Performs the same preprocessing and feature engineering used during model training.
- Includes the original dataset, training notebook, extraction script, and serialized model file.

## Dataset

The project uses the Telco Customer Churn dataset, commonly used for customer retention modeling.

Dataset source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The local dataset file is `telecom_customer.csv`.

Dataset summary:

- Rows: 7,043 customers
- Columns: 21 original columns
- Target column: `Churn`
- Churn distribution:
  - No: 5,174 customers
  - Yes: 1,869 customers
- `TotalCharges` contains 11 blank values, which are removed during preprocessing.

## Input Features

The original dataset contains customer identifiers, demographic fields, account details, service subscriptions, billing information, and the churn label.

Main raw columns include:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

The application does not use `customerID` for prediction because it is only an identifier.

## Machine Learning Workflow

The model development workflow is documented in `telecom.ipynb`.

The workflow includes:

1. Loading the dataset from `telecom_customer.csv`.
2. Dropping the `customerID` column.
3. Encoding binary categorical columns such as `Partner`, `Dependents`, `PhoneService`, and `PaperlessBilling`.
4. One-hot encoding multi-class categorical columns:
   - `gender`
   - `MultipleLines`
   - `InternetService`
   - `Contract`
   - `PaymentMethod`
5. Cleaning `TotalCharges` by removing blank values and converting the column to numeric format.
6. Normalizing numeric columns:
   - `tenure`
   - `MonthlyCharges`
   - `TotalCharges`
7. Creating engineered features:
   - `ServicesUsed`: count of selected customer services.
   - `HeavyUser`: binary flag set to 1 when the customer uses at least three services.
8. Splitting the data into training and testing sets using `train_test_split`.
9. Training an `XGBClassifier`.
10. Saving the trained model as `churning_customer.pkl`.

## Model Details

The deployed model is an XGBoost classifier loaded from `churning_customer.pkl`.

Model configuration:

- Algorithm: `XGBClassifier`
- Estimators: 300
- Learning rate: 0.05
- Maximum depth: 6
- Random state: 42
- Test score recorded in the notebook: approximately 0.7552

The model predicts two classes:

- `0`: customer is not expected to churn
- `1`: customer is expected to churn

The web application uses `predict_proba` and reports the probability of class `1` as the churn probability.

## Final Model Feature Set

The model expects 31 processed features in a fixed order:

```text
SeniorCitizen
Partner
Dependents
tenure
PhoneService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
PaperlessBilling
MonthlyCharges
TotalCharges
gender_Female
gender_Male
MultipleLines_No
MultipleLines_No phone service
MultipleLines_Yes
InternetService_DSL
InternetService_Fiber optic
InternetService_No
Contract_Month-to-month
Contract_One year
Contract_Two year
PaymentMethod_Bank transfer (automatic)
PaymentMethod_Credit card (automatic)
PaymentMethod_Electronic check
PaymentMethod_Mailed check
ServicesUsed
HeavyUser
```

The Flask application constructs this feature set from the web form before passing it to the model.

## Application Architecture

The project has three main parts:

- Model training and preparation: `telecom.ipynb` and `extract.py`
- Backend prediction service: `app.py`
- Frontend user interface: `templates/index.html` and `static/css/style.css`

Request flow:

1. The user enters customer information in the web form.
2. JavaScript collects form values and sends them as JSON to `/predict`.
3. Flask validates and transforms the input into the trained model feature format.
4. The XGBoost model returns churn class probabilities.
5. Flask converts the churn probability into a percentage and assigns a risk category.
6. The frontend displays the probability, risk label, and suggested action.

## Risk Classification Logic

The application converts the model probability into a business-friendly risk level:

| Churn Probability | Risk Category | Suggested Action |
| --- | --- | --- |
| Less than 30 percent | Low Churn Risk | No immediate action |
| 30 percent to 69 percent | Moderate Churn Risk | Offer incentives |
| 70 percent or higher | High Churn Risk | Immediate contact |

## API Endpoint

### `POST /predict`

Accepts customer details as JSON and returns the churn probability.

Example request:

```json
{
  "gender": "Male",
  "SeniorCitizen": "0",
  "Partner": "1",
  "Dependents": "0",
  "tenure": "12",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "1",
  "MonthlyCharges": "50.00",
  "TotalCharges": "600.00",
  "PhoneService": "1",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "0",
  "OnlineBackup": "0",
  "DeviceProtection": "0",
  "TechSupport": "0",
  "StreamingTV": "0",
  "StreamingMovies": "0"
}
```

Example response:

```json
{
  "success": true,
  "probability": 42.15,
  "category": "Moderate Churn Risk",
  "risk_level": "medium"
}
```

## Project Structure

```text
telecom_customer_churn_prediction/
|-- app.py
|-- churning_customer.pkl
|-- extract.py
|-- README.md
|-- telecom.ipynb
|-- telecom_customer.csv
|-- static/
|   `-- css/
|       `-- style.css
`-- templates/
    `-- index.html
```

File descriptions:

- `app.py`: Flask application, model loader, preprocessing logic, and prediction endpoint.
- `churning_customer.pkl`: Serialized trained XGBoost model.
- `telecom.ipynb`: Jupyter notebook containing data preprocessing, feature engineering, model training, and model export.
- `extract.py`: Python script exported from the notebook workflow.
- `telecom_customer.csv`: Local copy of the Telco Customer Churn dataset.
- `templates/index.html`: Main web interface and client-side prediction handling.
- `static/css/style.css`: Styling for the responsive frontend.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/punitxdev/telecom_customer_churn_prediction.git
cd telecom_customer_churn_prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install flask pandas scikit-learn xgboost joblib
```

The local environment used for this project includes:

- Flask
- pandas
- scikit-learn
- XGBoost
- joblib
- NumPy
- SciPy

## Running the Application

Start the Flask server:

```bash
python app.py
```

If your system uses `python3` as the Python executable:

```bash
python3 app.py
```

The application runs on:

```text
http://127.0.0.1:5001
```

Open this URL in a browser, complete the customer profile form, and select `Analyze Risk` to generate a prediction.

## Reproducing Model Training

To retrain or inspect the model development process:

1. Open `telecom.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells in order.
3. Review preprocessing, feature engineering, and model training steps.
4. Export the trained model with `joblib.dump(model, "churning_customer.pkl")`.

Important: The deployed Flask application expects the same feature names and preprocessing logic used during training. If the notebook preprocessing changes, update `FEATURE_NAMES` and the transformation logic in `app.py` accordingly.

## Technology Stack

- Python
- Flask
- pandas
- scikit-learn
- XGBoost
- joblib
- HTML
- CSS
- JavaScript
