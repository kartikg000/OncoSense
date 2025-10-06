import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app import load_assets, get_feature_importance_df

# Load the cached model and features
model, scaler, feature_info = load_assets()
feature_names = list(feature_info.keys())

st.set_page_config(page_title="Model Explained", page_icon="🧠", layout="wide")

st.title("🧠 Model Explained: How the Prediction Works")
st.markdown("---")

# --- 1. Random Forest Feature Importance ---
st.header("1. Random Forest: Feature Importance")
st.markdown("""
The **Random Forest** model determines a score for each feature, indicating how useful it was in accurately classifying the tumors. 
Features with higher scores are the most critical predictors of malignancy.
""")

try:
    df_importance = get_feature_importance_df(model, feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importance, palette="viridis", ax=ax)
    ax.set_title("Feature Importance in Random Forest Model")
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Feature (Mean Measurements)")
    st.pyplot(fig)

    st.subheader("Key Insights from Feature Importance:")
    st.markdown(f"""
    1.  **Top Predictor:** **`{df_importance.iloc[0]['Feature'].replace('_', ' ').title()}`** is the single most powerful feature for classification.
    2.  **Weakest Predictor:** **`{df_importance.iloc[-1]['Feature'].replace('_', ' ').title()}`** contributes the least to the model's decision.
    """)
    st.markdown("""
    *Generally, features related to size (`perimeter`, `area`, `radius`) and irregularity (`concave points`, `concavity`) are the most predictive of a malignant mass.*
    """)

except Exception as e:
    st.error(f"Could not generate Feature Importance plot. Ensure the model is loaded correctly. Error: {e}")

st.markdown("---")

# --- 2. The Random Forest Algorithm ---
st.header("2. How Random Forest Works")
st.markdown("""
The **Random Forest** is an **ensemble learning** method, meaning it combines the results of many simpler models (Decision Trees) to make a final prediction.

1.  **Many Trees:** It builds hundreds of independent decision trees.
2.  **Randomness:** Each tree is trained on a random subset of the data (bootstrapping) and only considers a random subset of features at each decision split. This ensures diversity among the trees.
3.  **Voting:** To make a final diagnosis, the model collects the prediction from every single tree.
4.  **Final Result:** The class that receives the most votes (**Malignant** or **Benign**) wins.

This voting process makes the model highly robust, accurate, and resistant to noise or overfitting.
""")

st.markdown("---")

# --- 3. Feature Scaling (StandardScaler) ---
st.header("3. Why We Scale the Data")
st.markdown(r"""
The measurements in the dataset are on vastly different scales:
* `radius_mean`: typically between **7** and **28**.
* `area_mean`: typically between **140** and **2500**.

If we don't scale the data, the model might incorrectly assume that features with larger numerical values (like `area_mean`) are inherently more important than features with smaller values (like `compactness_mean`).

We use **Standard Scaling** to normalize all features so they have:
* A mean ($\mu$) of **0**.
* A standard deviation ($\sigma$) of **1**.

The formula for Standard Scaling (Z-score) is:

$$
Z = \frac{X - \mu}{\sigma}
$$

This ensures that every feature contributes equally to the training process, leading to a fairer and more accurate model.
""")