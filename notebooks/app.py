from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # notebooks/
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model_compressed.pkl")
pipe = joblib.load(MODEL_PATH)

st.title('Smart Premium : Insurance Premium Prediction')
st.header('Enter Customer & Policy Details')

#numerical fields
age=st.number_input('Age',min_value=18,max_value=100,value=30)
annual_income=st.number_input('Annual Income',min_value=0,value=50000,max_value=1000000)
num_dependents=st.number_input('Number of Dependents',min_value=0,max_value=10,value=0)
health_score=st.slider('Health Score',min_value=0,max_value=100,value=70)
vehicle_age=st.number_input('Vehicle Age (Years)',min_value=0,max_value=30,value=5)
credit_score=st.number_input('Credit Score',min_value=300,max_value=850, value=600)
insurance_duration=st.number_input('Insurance Duration (Years)',min_value=1,max_value=30,value=5)
previous_claims=st.number_input('Previous Claims',min_value=0,max_value=50,value=0)

#categorical fields
gender=st.selectbox('Gender',['Male','Female'])
marital_status=st.selectbox('Marital Status',['Single','Married','Divorced'])
education=st.selectbox('Education Level',['High School','Bachelor\'s','Master\'s','PhD'])
occupation=st.selectbox('Occupation',['Employed','Self-Employed','Unemployed'])
location=st.selectbox('Location',['Urban','Suburban','Rural'])
policy_type=st.selectbox('Policy Type',['Basic','Comprehensive','Premium'])
smoking_status=st.radio('Smoking Status',['Yes','No'])
exercise_freq=st.selectbox('Exercise Frequency',['Daily','Weekly','Monthly','Rarely'])
property_type=st.selectbox('Property Type',['House','Apartment','Condo'])

#date fields
policy_start_date=st.date_input('Policy Start Date',value=date(2020,1,1))
policy_start_year=policy_start_date.year
policy_start_month=policy_start_date.month
policy_start_day=policy_start_date.day
day_of_week=policy_start_date.weekday()

input_data=pd.DataFrame(
    {
        'Age':[age],
        'Gender':[gender],
        'Annual Income':[annual_income],
        'Marital Status':[marital_status],
        'Number of Dependents':[num_dependents],
        'Education Level':[education],
        'Occupation':[occupation],
        'Health Score':[health_score],
        'Location':[location],
        'Policy Type':[policy_type],
        'Previous Claims':[previous_claims],
        'Vehicle Age':[vehicle_age],
        'Credit Score':[credit_score],
        'Insurance Duration':[insurance_duration],
        'Smoking Status':[smoking_status],
        'Exercise Frequency':[exercise_freq],
        'Property Type':[property_type],
        'Policy Start year':[policy_start_year],
        'Policy Start Month':[policy_start_month],
        'Policy Start Day':[policy_start_day],
        'Day of Week':[day_of_week]
    }
)

st.write('### Preview of Input Data')
st.dataframe(input_data)

if st.button('Predict Premium'):
    prediction = pipe.predict(input_data)[0] 
    lower = max(prediction * 0.9, 0)
    upper = prediction * 1.1
    st.success(f'Estimated Insurance Premium : Rs.{prediction:.2f}')
