import joblib
import pandas as pd
import json

# LOAD ARTIFACTS DIRECTLY
gmm = joblib.load("artifacts/gmm_models/gmm_udder.joblib")
scaler = joblib.load("artifacts/scalers/scaler_udder.joblib")

with open("artifacts/config/thresholds.json") as f:
    thresholds = json.load(f)

threshold = float(thresholds["udder"])

print("SCALER MEAN:", scaler.mean_)
print("SCALER SCALE:", scaler.scale_)
print("THRESHOLD:", threshold)

# TEST A KNOWN-HEALTHY POINT
x = pd.DataFrame(
    [[1.5, 3.0, 1.5]],
    columns=["delta_mean", "delta_max", "frame_std"]
)

x_scaled = scaler.transform(x)
score = gmm.score_samples(x_scaled)[0]

print("SCALED INPUT:", x_scaled)
print("GMM SCORE:", score)
