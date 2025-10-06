import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, 'feature_info.joblib')
RANDOM_STATE = 42

def prepare_data():
    print("Loading data...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    mean_features = [col for col in df.columns if 'mean' in col]
    X = df[mean_features]
    y = df['target']
    print(f"Features used for training: {list(X.columns)}")
    return X, y

def train_and_save_model(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print("Fitting and saving StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    
    print(f"Saving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    feature_info = {}
    for col in X.columns:
        feature_info[col] = {
            'min': X[col].min(),
            'max': X[col].max(),
            'default': X[col].median()
        }
    joblib.dump(feature_info, FEATURE_INFO_PATH)
    print(f"Feature info saved to {FEATURE_INFO_PATH}")
    
    print("\nTraining complete! Model and Scaler saved.")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = prepare_data()
    train_and_save_model(X, y)