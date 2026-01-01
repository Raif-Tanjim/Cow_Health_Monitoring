import time
import json
import joblib
import numpy as np
import board
import busio
import adafruit_mlx90640
from datetime import datetime

# ---------------------------
# Load artifacts
# ---------------------------
gmm = joblib.load("artifacts/gmm_models/gmm_udder.joblib")
scaler = joblib.load("artifacts/scalers/scaler_udder.joblib")

with open("artifacts/config/thresholds.json") as f:
    thresholds = json.load(f)

THRESHOLD = thresholds["udder"]

# ---------------------------
# Setup MLX90640
# ---------------------------
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

frame = np.zeros((32 * 24,))

# ---------------------------
# Real-time loop
# ---------------------------
while True:
    try:
        mlx.getFrame(frame)

        mean_temp = np.mean(frame)
        max_temp = np.max(frame)
        std_temp = np.std(frame)

        ambient = mean_temp
        delta_mean = mean_temp - ambient
        delta_max = max_temp - ambient

        X = [[delta_mean, delta_max, std_temp, 70.0]]  # humidity placeholder
        X_scaled = scaler.transform(X)

        score = gmm.score_samples(X_scaled)[0]
        abnormal = score < THRESHOLD
        confidence = round(min(1.0, abs(score) / 6), 2)

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "cow_part": "udder",
            "condition": "mastitis_suspected" if abnormal else "normal",
            "severity": "HIGH" if confidence > 0.6 else "NONE",
            "confidence": confidence,
            "gmm_score": round(score, 2)
        }

        # 🔑 Write result for Streamlit
        with open("latest_result.json", "w") as f:
            json.dump(result, f, indent=2)

        time.sleep(2)

    except Exception as e:
        print("Sensor error:", e)
        time.sleep(2)
