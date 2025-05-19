
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/air_quality_model.pkl")

st.title("Air Quality Category Predictor")

uploaded_file = st.file_uploader("Upload CSV file with air quality readings", type=["csv"])
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    try:
        features = new_data.drop(columns=["AQI_Category", "Combined_Pollutant"], errors="ignore")
        predictions = model.predict(features)
        new_data["Predicted AQI Category"] = predictions
        st.write(new_data)
    except Exception as e:
        st.error(f"Error: {e}")
