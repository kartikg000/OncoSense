import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from app import load_raw_data

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Data Exploration")
st.markdown("Explore the Wisconsin Breast Cancer Dataset with interactive visualizations.")

df = load_raw_data()
numerical_cols = df.drop(columns=['Target Diagnosis']).columns.tolist()

st.header("1. Dataset Overview")
if st.checkbox("Show Raw Data (First 50 Rows)"):
    st.dataframe(df.head(50))
st.info(f"The dataset has **{df.shape[0]} rows** (patients) and **{df.shape[1]-1} measurement features**.")

# --- 2. Target Distribution ---
st.header("2. Target Distribution")
diagnosis_counts = df['Target Diagnosis'].value_counts()
fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
palette = {'Benign': '#4ade80', 'Malignant': '#f87171'}
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, ax=ax_dist, palette=palette)
ax_dist.set_title("Distribution of Diagnosis (Target)", fontsize=16)
ax_dist.set_ylabel("Count", fontsize=12)
ax_dist.set_xlabel("Diagnosis Type", fontsize=12)
ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_dist)
st.markdown(f"""
- **Benign:** {diagnosis_counts['Benign']} cases
- **Malignant:** {diagnosis_counts['Malignant']} cases
""")
st.markdown("---")


# --- 3. Interactive Feature Distribution (Histogram) ---
st.header("3. Interactive Feature Distribution (Histogram)")
st.markdown("Select any feature to visualize its distribution for Benign vs. Malignant cases. Overlap indicates a less predictive feature, separation indicates high predictive power.")

default_col_hist = 'radius_mean' if 'radius_mean' in numerical_cols else numerical_cols[0]
selected_feature_hist = st.selectbox(
    'Select Feature for Histogram:',
    options=numerical_cols,
    index=numerical_cols.index(default_col_hist)
)

fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
sns.histplot(
    data=df, 
    x=selected_feature_hist, 
    hue='Target Diagnosis', 
    kde=True, 
    palette={'Benign': '#4ade80', 'Malignant': '#f87171'},
    ax=ax_hist
)
ax_hist.set_title(f"Distribution of {selected_feature_hist.replace('_', ' ').title()}")
st.pyplot(fig_hist)
st.markdown("---")


# --- 4. Interactive Feature Comparison (Box Plot) ---
st.header("4. Interactive Feature Comparison (Box Plot)")
st.markdown("Box plots show the range and quartiles for a feature across the two diagnosis groups. A clear separation of the boxes indicates the feature's strength as a predictor.")

default_col_box = 'radius_mean' if 'radius_mean' in numerical_cols else numerical_cols[0]
selected_feature_box = st.selectbox(
    'Select Feature for Box Plot:',
    options=numerical_cols,
    index=numerical_cols.index(default_col_box),
    key='box_plot_selector'
)

fig_box, ax_box = plt.subplots(figsize=(8, 6))
sns.boxplot(
    x='Target Diagnosis', 
    y=selected_feature_box, 
    data=df, 
    palette={'Benign': '#4ade80', 'Malignant': '#f87171'},
    ax=ax_box
)
ax_box.set_title(f"Comparison of {selected_feature_box.replace('_', ' ').title()} by Diagnosis")
st.pyplot(fig_box)
st.markdown("---")


# --- 5. Interactive Scatter Plot ---
st.header("5. Interactive Scatter Plot")
st.markdown("Visualize the relationship between two features. Look for clear separation between the green (Benign) and red (Malignant) clusters.")

col_x, col_y = st.columns(2)

default_x = 'radius_mean' if 'radius_mean' in numerical_cols else numerical_cols[0]
default_y = 'texture_mean' if 'texture_mean' in numerical_cols else numerical_cols[1]


x_axis_feature = col_x.selectbox(
    'X-Axis Feature:',
    options=numerical_cols,
    index=numerical_cols.index(default_x)
)

y_axis_feature = col_y.selectbox(
    'Y-Axis Feature:',
    options=numerical_cols,
    index=numerical_cols.index(default_y)
)

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x=x_axis_feature, 
    y=y_axis_feature, 
    hue='Target Diagnosis', 
    palette={'Benign': '#4ade80', 'Malignant': '#f87171'},
    ax=ax_scatter
)
ax_scatter.set_title(f'{x_axis_feature.title()} vs. {y_axis_feature.title()}')
st.pyplot(fig_scatter)
st.markdown("---")


# --- 6. Feature Correlation ---
st.header("6. Feature Correlation Heatmap")
st.markdown("A heatmap showing the correlation between all numerical features. Values closer to 1 (bright red) or -1 (dark blue) indicate a strong linear relationship.")
numerical_df = df[numerical_cols]
df_corr = numerical_df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(df_corr, annot=False, cmap='coolwarm', fmt=".2f", ax=ax_corr, cbar_kws={'label': 'Correlation Coefficient'})
ax_corr.set_title("Feature Correlation Heatmap", fontsize=18)
st.pyplot(fig_corr)
st.markdown("---")


# --- 7. Descriptive Statistics ---
st.header("7. Descriptive Statistics")
st.markdown("Summary statistics for all measurement features in the dataset.")
st.dataframe(df.describe().T.style.format('{:.3f}'))