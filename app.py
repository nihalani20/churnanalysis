import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('random_forest_model.pkl')


# Initialize LabelEncoders
le_gender = joblib.load('label_encoder_gender.pkl')
le_partner = joblib.load('label_encoder_partner.pkl')
le_dependents = joblib.load('label_encoder_dependents.pkl')
le_phone_service = joblib.load('label_encoder_phone_service.pkl')
le_multiple_lines = joblib.load('label_encoder_multiple_lines.pkl')
le_internet_service = joblib.load('label_encoder_internet_service.pkl')
le_online_security = joblib.load('label_encoder_online_security.pkl')
le_online_backup = joblib.load('label_encoder_online_backup.pkl')
le_device_protection = joblib.load('label_encoder_device_protection.pkl')
le_tech_support = joblib.load('label_encoder_tech_support.pkl')
le_streaming_tv = joblib.load('label_encoder_streaming_tv.pkl')
le_streaming_movies = joblib.load('label_encoder_streaming_movies.pkl')
le_contract = joblib.load('label_encoder_contract.pkl')
le_paperless_billing = joblib.load('label_encoder_paperless_billing.pkl')
le_payment_method = joblib.load('label_encoder_payment_method.pkl')

# Define a function to preprocess input data
def preprocess_input(data):
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'

    ]
    encoders = {
        'gender': le_gender,
        'Partner': le_partner,
        'Dependents': le_dependents,
        'PhoneService': le_phone_service,
        'MultipleLines': le_multiple_lines,
        'InternetService': le_internet_service,
        'OnlineSecurity': le_online_security,
        'OnlineBackup': le_online_backup,
        'DeviceProtection': le_device_protection,
        'TechSupport': le_tech_support,
        'StreamingTV': le_streaming_tv,
        'StreamingMovies': le_streaming_movies,
        'Contract': le_contract,
        'PaperlessBilling': le_paperless_billing,
        'PaymentMethod': le_payment_method
    }

    # Transform categorical features
    for feature, encoder in encoders.items():
        if feature in data:
            try:
                data[feature] = encoder.transform([data[feature]])[0]
            except ValueError:
                st.error(f'Invalid input for {feature}')
                return pd.DataFrame(columns=feature_order)

    # Ensure 'SeniorCitizen' is in the correct format
    data['SeniorCitizen'] = int(data['SeniorCitizen'])

    # Reorder columns to match training data
    df_input = pd.DataFrame([data])
    df_input = df_input.reindex(columns=feature_order, fill_value=0)
    return df_input


st.title('Customer Churn Prediction')

# User inputs for customer details
# Collecting user inputs for all features

gender = st.selectbox("Gender",["Male","Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, step=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges",min_value = 10,max_value=200,step=1)
total_charges = st.number_input("Total Charges", min_value=10, max_value=10000, step=1)


# Predict button
if st.button('Predict'):
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    processed_input = preprocess_input(input_data)  # Renaming df_input to processed_input
    if not processed_input.empty:
        prediction = model.predict(processed_input)[0]
        st.write('Prediction: ', 'Churn' if prediction == 1 else 'Not Churn')








