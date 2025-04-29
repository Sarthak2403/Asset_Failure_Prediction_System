import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Load Model ---
MODEL_PATH = os.path.join("model", "random_forest_failure_model.pkl")
model = joblib.load(MODEL_PATH)

# --- Streamlit App ---
st.set_page_config(page_title="Asset Failure Prediction Dashboard", page_icon="🚀")
st.title("🚀 Asset Failure Prediction Dashboard")

st.write("""
Upload new sensor data and predict failure risk for each asset.

- **HIGH Risk** means immediate attention is needed.
- **LOW Risk** means normal operations.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded file
    input_df = pd.read_csv(uploaded_file)

    # Ensure required sensor columns are present
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    if not all(col in input_df.columns for col in sensor_cols):
        st.error("❌ Uploaded CSV must contain columns sensor_1 to sensor_21.")
    else:
        # Filter only sensor columns
        X = input_df[sensor_cols]

        # Predict
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        # Add results to original dataframe
        input_df['Failure Probability'] = np.round(probs, 4)
        input_df['Risk Level'] = np.where(preds == 1, "HIGH", "LOW")

        # Display predictions
        st.success("✅ Predictions completed.")
        st.dataframe(input_df)

        # Risk distribution
        st.write("### 📊 Risk Level Distribution")
        st.bar_chart(input_df['Risk Level'].value_counts())

        # Failure probability histogram
        st.write("### 📈 Failure Probability Histogram")
        fig, ax = plt.subplots()
        ax.hist(probs, bins=20, color='orange', edgecolor='black')
        ax.set_xlabel("Failure Probability")
        ax.set_ylabel("Asset Count")
        ax.set_title("Distribution of Failure Probabilities")
        st.pyplot(fig)

        # Download results
        csv_download = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_download,
            file_name='predicted_failures.csv',
            mime='text/csv',
        )

else:
    st.info("Please upload a CSV file to begin.")