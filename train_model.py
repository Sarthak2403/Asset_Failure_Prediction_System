import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import joblib
import os

# --- Load Preprocessed Data ---
DATA_FILE = os.path.join("data", "processed_data.csv")
df = pd.read_csv(DATA_FILE)

# --- Feature Selection ---
sensor_cols = [col for col in df.columns if "sensor_" in col]
X = df[sensor_cols]
y = df["failure_label"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train Random Forest Classifier ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save Model ---
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_failure_model.pkl")
joblib.dump(clf, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")