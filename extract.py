import pandas as pd
df = pd.read_csv("telecom_customer.csv")
df.head()
print(df.columns
     )
df.tail()
df = df.drop(["customerID"], axis=1)
df
df = pd.get_dummies(df,columns=["gender"])
df
df["Partner"] = df["Partner"].map({
     "Yes": 1,
     "No":0

})
df
df["Dependents"] = df["Dependents"].map({
     "Yes": 1,
     "No":0

})
df
tenure_max = df["tenure"].min()
tenure_min = df["tenure"].max()
df["tenure"] = (df["tenure"] - tenure_min)/(tenure_max - tenure_min)
df

df["PhoneService"] = df["PhoneService"].map({
     "Yes": 1,
     "No":0

})
df
df["MultipleLines"].unique()
df = pd.get_dummies(df, columns=["MultipleLines"])
df
df["InternetService"].unique()
df = pd.get_dummies(df, columns=["InternetService"])
df
df["OnlineSecurity"] = df["OnlineSecurity"].map({
    "No":0,
    "Yes":1
})
df["OnlineBackup"] = df["OnlineBackup"].map({
    "No":0,
    "Yes":1
})
df["DeviceProtection"] = df["DeviceProtection"].map({
    "No":0,
    "Yes":1
})
df
df.columns
print(df.dtypes)
binary_cols = ["TechSupport", "StreamingTV", "PaperlessBilling", "StreamingMovies"]
for i in binary_cols:
    df[i] = df[i].map({
        "No":0,
        "Yes":1
    })
    
print(df.dtypes)
df["PaymentMethod"].unique()
df = pd.get_dummies(df, columns=["Contract","PaymentMethod"])
df
df[["MonthlyCharges", "TotalCharges"]]
df = df[df['TotalCharges'].str.strip() != '']
import numpy as np
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
min_MC = df["MonthlyCharges"].min()
max_MC = df["MonthlyCharges"].max()
min_TC = float(df["TotalCharges"].min())
max_TC = float(df["TotalCharges"].max())
print(f" min MC = {min_MC} max MC = {max_MC} min_TC ={ min_TC} max TC = { max_TC}")
df["TotalCharges"].min()
df["MonthlyCharges"] = (df["MonthlyCharges"]-min_MC)/(max_MC - min_MC)
df.dropna(inplace=True)
df
print(df.dtypes)
pd.set_option('display.max_columns', None)
df
df["TotalCharges"] = (df["TotalCharges"]-min_TC)/(max_TC - min_TC)
df
df1 = df.copy()
df["Churn"] = df["Churn"].map({
    'No':0,
    'Yes':1
})
df
y = df["Churn"]
y
x = df.drop(["Churn"], axis=1)
x["ServicesUsed"] = df["PhoneService"] + df["OnlineSecurity"] + df["OnlineBackup"] + df["TechSupport"] + df["StreamingTV"] + df["StreamingMovies"]
print(x.columns)
x["ServicesUsed"].max()
x["HeavyUser"] = (
    x["ServicesUsed"] >= 3
).astype(int)
x
print(x.columns)
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state = 42)
# model = RandomForestClassifier(
#     n_estimators=500,
#     max_depth = 10,
#     min_samples_split=5,
#     random_state=42
# )

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(x_train, y_train)
model.score(x_test,y_test)
df.columns
import pandas as pd

# ==========================================
# CREATE INPUT DICTIONARY
# ==========================================

input_data = dict.fromkeys(x.columns, 0)

# ==========================================
# NUMERICAL INPUTS
# ==========================================

# tenure
tenure = float(input("Enter Tenure (0-72 months): "))

input_data['tenure'] = tenure / 72

# ------------------------------------------

# MonthlyCharges
# Min = 18.25
# Max = 118.75

monthly = float(input("Enter Monthly Charges: "))

input_data['MonthlyCharges'] = (
    (monthly - 18.25) /
    (118.75 - 18.25)
)

# ------------------------------------------

# TotalCharges
# Min = 18.8
# Max = 8684.8

total = float(input("Enter Total Charges: "))

input_data['TotalCharges'] = (
    (total - 18.8) /
    (8684.8 - 18.8)
)

# ==========================================
# BINARY INPUTS
# ==========================================

binary_cols = [
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling'
]

for col in binary_cols:

    value = int(
        input(f"{col} (0=No, 1=Yes): ")
    )

    input_data[col] = value

# ==========================================
# GENDER
# ==========================================

print("""
Gender:
1 = Male
2 = Female
""")

gender = int(input("Enter Gender: "))

if gender == 1:
    input_data['gender_Male'] = 1

elif gender == 2:
    input_data['gender_Female'] = 1

# ==========================================
# MULTIPLE LINES
# ==========================================

print("""
Multiple Lines:
1 = No
2 = No phone service
3 = Yes
""")

multiple = int(input("Enter Multiple Lines Option: "))

if multiple == 1:
    input_data['MultipleLines_No'] = 1

elif multiple == 2:
    input_data['MultipleLines_No phone service'] = 1

elif multiple == 3:
    input_data['MultipleLines_Yes'] = 1

# ==========================================
# INTERNET SERVICE
# ==========================================

print("""
Internet Service:
1 = DSL
2 = Fiber optic
3 = No
""")

internet = int(input("Enter Internet Service: "))

if internet == 1:
    input_data['InternetService_DSL'] = 1

elif internet == 2:
    input_data['InternetService_Fiber optic'] = 1

elif internet == 3:
    input_data['InternetService_No'] = 1

# ==========================================
# CONTRACT
# ==========================================

print("""
Contract:
1 = Month-to-month
2 = One year
3 = Two year
""")

contract = int(input("Enter Contract Type: "))

if contract == 1:
    input_data['Contract_Month-to-month'] = 1

elif contract == 2:
    input_data['Contract_One year'] = 1

elif contract == 3:
    input_data['Contract_Two year'] = 1

# ==========================================
# PAYMENT METHOD
# ==========================================

print("""
Payment Method:
1 = Bank transfer (automatic)
2 = Credit card (automatic)
3 = Electronic check
4 = Mailed check
""")

payment = int(input("Enter Payment Method: "))

if payment == 1:
    input_data[
        'PaymentMethod_Bank transfer (automatic)'
    ] = 1

elif payment == 2:
    input_data[
        'PaymentMethod_Credit card (automatic)'
    ] = 1

elif payment == 3:
    input_data[
        'PaymentMethod_Electronic check'
    ] = 1

elif payment == 4:
    input_data[
        'PaymentMethod_Mailed check'
    ] = 1

# ==========================================
# FEATURE ENGINEERING
# ==========================================

services_used = (
    input_data['PhoneService'] +
    input_data['OnlineSecurity'] +
    input_data['OnlineBackup'] +
    input_data['DeviceProtection'] +
    input_data['TechSupport'] +
    input_data['StreamingTV'] +
    input_data['StreamingMovies']
)

# Total Services Used
input_data['ServicesUsed'] = services_used

# Heavy User Feature
input_data['HeavyUser'] = (
    1 if services_used >= 3 else 0
)

# ==========================================
# CONVERT TO DATAFRAME
# ==========================================

input_df = pd.DataFrame([input_data])

# ==========================================
# PREDICT PROBABILITY
# ==========================================

prob = model.predict_proba(input_df)

churn_prob = float(prob[0][1]) * 100

# ==========================================
# DISPLAY RESULT
# ==========================================

print(f"\nServices Used: {services_used}")

print(f"Heavy User: {input_data['HeavyUser']}")

print(f"\nChurn Probability: {churn_prob:.2f}%")

# ==========================================
# RISK ANALYSIS
# ==========================================

if churn_prob >= 70:
    print("Customer Will Churn")

elif churn_prob >= 30:
    print("Customer Needs Intervention")

else:
    print("Customer Will Stay")
print(df['Churn'].value_counts())
prob = model.predict_proba(input_df)

print(prob)
from sklearn.metrics import classification_report

pred = model.predict(x_test)

print(classification_report(y_test, pred))
import joblib
joblib.dump(model, "churning_customer.pkl")


