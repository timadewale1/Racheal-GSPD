import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
# Make sure you saved your model earlier with:
# pickle.dump(lasso_model, open("lasso_model.pkl", "wb"))
model = joblib.load("Random_Forest_Regressor.pkl")
st.title("🌍 Malaria Cases Prediction App")
st.write("Predict malaria cases using rainfall, temperature, and lagged features.")

# Inputs the user WILL provide
# -------------------------------
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
rainfall_lag1 = st.number_input("Rainfall Lag 1", min_value=0.0, step=0.1)
rainfall_lag2 = st.number_input("Rainfall Lag 2", min_value=0.0, step=0.1)
temperature_lag5 = st.number_input("Temperature Lag 5", min_value=0.0, step=0.1)
temperature_lag6 = st.number_input("Temperature Lag 6", min_value=0.0, step=0.1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Malaria Cases"):
    features = np.array([[
                          rainfall,
                          temperature,
                          rainfall_lag1,
                          rainfall_lag2,
                          temperature_lag5,
                          temperature_lag6]])

    prediction = model.predict(features)[0]
    st.success(f"✅ Predicted Malaria Cases: {int(prediction)}")

