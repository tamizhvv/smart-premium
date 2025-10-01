#Smart Premium: Insurance Premium Prediction

##Overview:

Smart Premium is an end-to-end machine learning project that predicts insurance premiums for customers based on personal, financial, and policy-related data. It leverages feature engineering, multiple regression models, and model tracking using MLflow, providing interpretable and accurate predictions.

##Key Features:

-Handles both numerical and categorical features including age, income, health score, policy type, etc.

-Implements a pipeline with preprocessing and model in a single workflow.

-Tracks experiments with MLflow, logging metrics like RMSE, MAE, and R².

-Uses joblib to serialize the best performing model.

-Deployed as a Streamlit web application for interactive premium prediction.

##Tech Stack:

Python: pandas, numpy, scikit-learn, xgboost, joblib, mlflow

###Deployment: Streamlit

###Version Control: GitHub

##Project Highlights:

-Trained multiple regression models and selected the best one based on validation RMSE.

-Retrained the best model on the full dataset for final deployment.

-Developed a user-friendly Streamlit interface for inputting customer details and generating premium predictions.

##Project repository: GitHub Link

##Live demo: Streamlit App Link
 https://smart-premium-hp4artz746thktnerzbwtg.streamlit.app/

##Outcome:

The project demonstrates practical end-to-end ML workflow skills including data preprocessing, model evaluation, tracking, serialization, and deployment. It is suitable for insurance companies or financial analytics applications looking to automate premium calculation.