import streamlit as st
import numpy as np
import pandas as pd
from app import load_assets

# Load model, scaler, and feature info (cached from app.py)
model, scaler, feature_info = load_assets() 

st.set_page_config(page_title="Prediction", page_icon="🔮")

st.title("🔮 Real-Time Cancer Prediction")
st.markdown("---")

# --- Function to build the sidebar/form and collect input ---
def get_user_input():
    st.sidebar.header("🔬 Cellular Measurements Input")
    st.sidebar.markdown("Adjust the 10 'mean' features below. Use the default values (medians from the dataset) as a starting point.")
    
    user_data = {}
    col1, col2 = st.sidebar.columns(2)
    
    # Iterate through the 10 features used by the model
    for i, (feature, stats) in enumerate(feature_info.items()):
        
        display_name = feature.replace('_mean', '').replace('_', ' ').title()
        target_col = col1 if i % 2 == 0 else col2
        
        # Create a slider for each feature
        value = target_col.slider(
            label=f"{display_name}",
            min_value=float(stats['min']),
            max_value=float(stats['max']),
            value=float(stats['default']),
            # Set dynamic step size for better control
            step=0.001 if stats['max'] < 1 else (0.1 if stats['max'] < 100 else 1.0), 
            format="%.3f" if stats['max'] < 1 else "%.2f"
        )
        user_data[feature] = value
        
    return pd.DataFrame([user_data])

# --- Get Input and Display ---
input_df = get_user_input()

st.subheader("Current Input Data (10 Mean Features)")
st.dataframe(input_df.T.rename(columns={0: 'Value'}).style.format(lambda x: f"{x:.3f}"))
st.markdown("---")

# --- Prediction Logic ---
scaled_input = scaler.transform(input_df)

# Make Prediction and Get Probabilities
prediction_raw = model.predict(scaled_input)[0]
prediction_proba = model.predict_proba(scaled_input)[0]

# Diagnosis Mapping (0 is Benign, 1 is Malignant)
DIAGNOSIS_MAP = {0: "Benign", 1: "Malignant"}

final_diagnosis = DIAGNOSIS_MAP[prediction_raw]
prob_malignant = prediction_proba[1] 
prob_benign = prediction_proba[0]    


st.header("Final Diagnosis Result")
col1, col2 = st.columns(2)

with col1:
    if final_diagnosis == "Malignant":
        st.error(f"❗ Predicted Class: MALIGNANT")
        st.metric(label="Malignant Confidence", value=f"{prob_malignant*100:.2f}%")
        
        # Calmer, professional feedback for Malignant
        st.markdown(f"""
        <div style='background-color: #fef2f2; padding: 15px; border-left: 5px solid #ef4444; border-radius: 8px;'>
            <p style='color: #b91c1c; font-weight: bold; font-size: 1.1em;'>
                <i class="fas fa-exclamation-triangle"></i> Assessment: High Malignant Likelihood
            </p>
            <p style='color: #b91c1c;'>
                The model indicates a high probability that the mass is Malignant (Cancerous) based on the inputs. 
                Features like **Concave Points** and **Radius** appear high, often correlating with malignancy.
            </p>
            <p style='margin-top: 10px; font-weight: bold;'>
                **Action:** This requires immediate clinical follow-up and verification.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Predicted Class: BENIGN")
        st.metric(label="Benign Confidence", value=f"{prob_benign*100:.2f}%")
        
        # Calmer, professional feedback for Benign
        st.markdown(f"""
        <div style='background-color: #f0fdf4; padding: 15px; border-left: 5px solid #22c55e; border-radius: 8px;'>
            <p style='color: #166534; font-weight: bold; font-size: 1.1em;'>
                <i class="fas fa-check-circle"></i> Assessment: High Benign Likelihood
            </p>
            <p style='color: #166534;'>
                The model indicates a high probability that the mass is Benign (Non-Cancerous). 
                The cellular measurements fall within the typical range for non-harmful growths in this dataset.
            </p>
            <p style='margin-top: 10px; font-weight: bold;'>
                **Action:** While the risk appears low, continuous monitoring and standard diagnostic procedures are always recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
with col2:
    st.subheader("Probability Breakdown")
    
    prob_df = pd.DataFrame({
        'Diagnosis': ['Benign', 'Malignant'],
        'Probability': [prob_benign, prob_malignant]
    })
    
    st.bar_chart(prob_df.set_index('Diagnosis'))
    st.caption("Visual representation of the model's confidence scores.")