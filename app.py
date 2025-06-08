import streamlit as st
import pandas as pd
import joblib
import os

# Configure page layout
st.set_page_config(page_title="Smoking & Drinking Predictor", layout="centered")

# Get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
smoking_model_path = os.path.join(BASE_DIR, "rf_smoking_model.pkl")
drinking_model_path = os.path.join(BASE_DIR, "gb_drinking_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load models and scaler
try:
    rf_model = joblib.load(smoking_model_path)
    gb_model = joblib.load(drinking_model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
except FileNotFoundError:
    st.error("❌ Model or scaler files not found. Please make sure 'rf_smoking_model.pkl', 'gb_drinking_model.pkl', and 'scaler.pkl' exist.")
    model_loaded = False

# UI layout
st.title("🚭🍷 Smoking & Drinking Status Predictor")
st.markdown("Use this tool to predict **smoking** and **drinking** status based on biometric inputs.")

# Sidebar inputs
with st.sidebar:
    st.header("📋 Enter Your Health Info")
    age = st.slider("Age", 10, 100, 30)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 40.0, 22.0)
    gamma_GTP = st.number_input("Gamma GTP (Liver Enzyme)", min_value=0.0, max_value=500.0, value=30.0)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.0)

# Prediction logic
if model_loaded:
    input_data = pd.DataFrame({
        "age": [age],
        "BMI": [bmi],
        "gamma_GTP": [gamma_GTP],
        "hemoglobin": [hemoglobin]
    })

    try:
        # Apply same scaling as during model training
        input_scaled = scaler.transform(input_data)

        # Smoking prediction
        smoking_pred = rf_model.predict(input_scaled)[0]
        smoking_prob = rf_model.predict_proba(input_scaled)[0].max()

        # Drinking prediction
        drinking_pred = gb_model.predict(input_scaled)[0]
        drinking_prob = gb_model.predict_proba(input_scaled)[0].max()

        # Results
        st.subheader("🔍 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Smoking Status",
                value="🚬 Smoker" if smoking_pred == 1 else "✅ Non-Smoker",
                delta=f"Confidence: {smoking_prob:.2f}"
            )

        with col2:
            st.metric(
                label="Drinking Status",
                value="🍷 Drinker" if drinking_pred == 1 else "✅ Non-Drinker",
                delta=f"Confidence: {drinking_prob:.2f}"
            )

        st.markdown("---")
        st.info("📌 This tool is for **educational purposes** only and does not replace medical advice.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")