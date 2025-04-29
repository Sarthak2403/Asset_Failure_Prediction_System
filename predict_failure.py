import pandas as pd
import joblib
import numpy as np
import os

# --- Load Model ---
MODEL_PATH = os.path.join("model", "random_forest_failure_model.pkl")
model = joblib.load(MODEL_PATH)

# --- Sample Input (simulate new data) ---
sample = {
    'sensor_1': [0.53], 'sensor_2': [0.42], 'sensor_3': [0.31],
    'sensor_4': [0.21], 'sensor_5': [0.45], 'sensor_6': [0.36],
    'sensor_7': [0.10], 'sensor_8': [0.05], 'sensor_9': [0.02],
    'sensor_10': [0.61], 'sensor_11': [0.29], 'sensor_12': [0.48],
    'sensor_13': [0.30], 'sensor_14': [0.19], 'sensor_15': [0.08],
    'sensor_16': [0.52], 'sensor_17': [0.24], 'sensor_18': [0.07],
    'sensor_19': [0.41], 'sensor_20': [0.33], 'sensor_21': [0.27]
}
input_df = pd.DataFrame(sample)

# --- Predict ---
failure_prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

# --- Display ---
print("🔍 Predicted Failure Probability:", round(failure_prob, 4))
print("⚠️ Failure Risk:", "HIGH" if prediction == 1 else "LOW")
