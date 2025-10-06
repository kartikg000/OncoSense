import streamlit as st
import joblib
from sklearn.datasets import load_breast_cancer
import pandas as pd
import os
import numpy as np

# --- Configuration for all pages ---
st.set_page_config(
    page_title="Breast Cancer Prediction ML App",
    page_icon="🎗️",
    layout="wide"
)

# --- Caching Functions (Load Model/Scaler Once) ---
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and feature info."""
    model_path = 'model/rf_model.joblib'
    scaler_path = 'model/scaler.joblib'
    feature_info_path = 'model/feature_info.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or Scaler files not found. Please run `python model_training/train_model.py` first.")
        st.stop()
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_info = joblib.load(feature_info_path)
        return model, scaler, feature_info
    except Exception as e:
        st.error(f"Error loading ML assets. Check the 'model/' directory. Error: {e}")
        st.stop()
        
model, scaler, feature_info = load_assets()

@st.cache_data
def load_raw_data():
    """Loads the full Wisconsin dataset for exploration."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df['diagnosis'] = df['target'].map({0: 'Benign', 1: 'Malignant'})
    df = df.drop(columns=['target']).rename(columns={'diagnosis': 'Target Diagnosis'})
    return df

@st.cache_data
def get_feature_importance_df(_model, feature_names):
    """Calculates and returns a DataFrame of feature importance for the model.
    The underscore in _model tells Streamlit not to hash this argument.
    """
    importance = _model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    return df_importance

# --- Home Page Content ---
st.title("🎗️ Wisconsin Breast Cancer Diagnosis Predictor")
st.markdown("---")

st.header("Welcome to the ML Diagnostic Assistant")
st.markdown("""
This application uses a trained Machine Learning model (a **Random Forest Classifier**) 
to predict whether a breast mass is **Benign** (non-cancerous) or **Malignant** (cancerous), 
based on 10 key physical measurements of the cell nuclei.

### Project Flow:
1.  **Data Collection:** Used the Wisconsin Breast Cancer (Diagnostic) Dataset.
2.  **Preprocessing:** Feature scaling was performed using `StandardScaler`.
3.  **Model Training:** A **Random Forest Classifier** was trained on the 10 'mean' features.
4.  **Deployment:** The trained model and scaler are loaded into this Streamlit application.

### Navigate:
-   **📊 Data Exploration:** View the raw dataset, feature statistics, and visualizations.
-   **🔮 Prediction:** Input 10 cellular measurements using sliders to get a real-time diagnosis prediction.
-   **🧠 Model Explained:** Understand the mathematics behind **Standard Scaling** and the **Random Forest** algorithm, including which features were most important.
""")

st.warning("Disclaimer: This tool is for educational and illustrative purposes only and should NOT be used for actual medical diagnosis. Always consult with qualified healthcare professionals.")