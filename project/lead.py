import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import load

model = load('gbm_model.joblib')

print("Model loaded successfully.")

scaler = joblib.load('scaler.pkl')  # Assuming a scaler was used and saved

def make_predictions(balance_range):
    # Create a DataFrame for the balance range
    data = pd.DataFrame(balance_range, columns=['Balance'])
    
    # Set fixed values for the other features (customize these based on your dataset's needs)
    fixed_features = {
        'CreditScore': 650,  # Example fixed credit score
        'Age': 30,           # Example fixed age
        'Tenure': 5,         # Example fixed tenure
        'NumOfProducts': 2,  # Example fixed number of products
        'Gender': 1,         # Example fixed gender (0 for Female, 1 for Male)
        'EstimatedSalary': 50000,  # Example fixed salary
        'Geo_France': 1,     # Example: assuming customer is from France
        'Geo_Germany': 0,
        'Geo_Spain': 0
    }
    
    # Add fixed features to the DataFrame
    for feature, value in fixed_features.items():
        data[feature] = value
    
    # Scale the data according to the model's scaling
    scaled_data = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    # Add predictions to the DataFrame
    data['Prediction'] = predictions
    
    return data

# Streamlit webpage content
st.title('Credit Card Purchase Prediction Across Balance Range')
st.write('This application predicts the likelihood of customers purchasing a credit card based on balance range.')

# Input fields for the balance range
balance_start = st.number_input('Starting Balance', min_value=0.0, max_value=100000.0, value=10000.0)
balance_end = st.number_input('Ending Balance', min_value=0.0, max_value=100000.0, value=50000.0)
increment = st.number_input('Increment', min_value=100.0, max_value=5000.0, value=1000.0)

if st.button('Predict'):
    balance_range = np.arange(balance_start, balance_end + increment, increment)
    results = make_predictions(balance_range)
    st.write(results)

