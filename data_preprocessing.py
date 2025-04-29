import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Paths
DATA_DIR = "data"
RAW_TRAIN = os.path.join(DATA_DIR, "train_FD001.txt")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed_data.csv")

# Columns (1 engine id, 1 cycle, 3 op settings, 21 sensors)
columns = ["engine_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
df = pd.read_csv(RAW_TRAIN, sep="\s+", header=None, names=columns)

# Compute RUL (max cycle - current cycle for each engine)
rul_df = df.groupby("engine_id")["cycle"].max().reset_index()
rul_df.columns = ["engine_id", "max_cycle"]
df = df.merge(rul_df, on="engine_id", how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]
df.drop("max_cycle", axis=1, inplace=True)

# Optional: classify RUL into binary failure label (if < threshold)
FAILURE_THRESHOLD = 30
df["failure_label"] = np.where(df["RUL"] <= FAILURE_THRESHOLD, 1, 0)

# Normalize sensor columns
sensor_cols = [col for col in df.columns if "sensor_" in col]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

# Save preprocessed data
df.to_csv(PROCESSED_CSV, index=False)
print(f"✅ Processed data saved to {PROCESSED_CSV}")