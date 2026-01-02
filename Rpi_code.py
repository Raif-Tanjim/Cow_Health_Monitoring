# ============================================================
# Cow Health Monitoring – Raspberry Pi Inference (UART MLX90640)
# ============================================================

import serial
import time
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
import board
import adafruit_dht

# =========================
# PATHS & DIRECTORIES
# =========================
BASE_DIR = Path(__file__).parent
SHARED_DIR = BASE_DIR / "shared"
CONTROL_DIR = SHARED_DIR / "control"
OUTPUT_DIR = SHARED_DIR / "output"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CONTROL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_FILE = CONTROL_DIR / "body_part.json"
RESULT_FILE = OUTPUT_DIR / "latest_result.json"
IMAGE_FILE = OUTPUT_DIR / "latest_image.png"

# Default control file
if not CONTROL_FILE.exists():
    CONTROL_FILE.write_text('{"body_part": "udder"}')

# =========================
# MLX90640 UART SETTINGS
# =========================
PORT = "/dev/ttyAMA0"
BAUD = 115200
FRAME_SIZE = 1544
PIXELS = 32 * 24
CAPTURE_FRAMES = 20

# =========================
# BODY PART TEMP RANGES (FOR VISUALIZATION)
# =========================
PART_TEMP_RANGES = {
    "eye":   (30.0, 38.0),
    "udder": (28.0, 40.0),
    "leg":   (26.0, 36.0),
    "hoof":  (24.0, 34.0),
    "etc":   (25.0, 40.0),
}

PARTS = list(PART_TEMP_RANGES.keys())
FEATURES = ["delta_mean", "delta_max", "frame_std", "humidity"]

# =========================
# LOAD MODELS & SCALERS
# =========================
gmm_models = {}
scalers = {}

for part in PARTS:
    gmm_models[part] = joblib.load(ARTIFACTS_DIR / f"gmm_models/gmm_{part}.joblib")
    scalers[part] = joblib.load(ARTIFACTS_DIR / f"scalers/scaler_{part}.joblib")

with open(ARTIFACTS_DIR / "config/thresholds.json") as f:
    thresholds = json.load(f)

# =========================
# DHT22 (AMBIENT SENSOR)
# =========================
dht = adafruit_dht.DHT22(board.D4, use_pulseio=False)

def read_ambient():
    try:
        return dht.temperature, dht.humidity
    except RuntimeError:
        return None, None

# =========================
# UART MLX90640 INIT
# =========================
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(1)

# Set MLX frame rate & mode (same as your capture script)
ser.write(bytes([0xA5, 0x25, 0x01, 0xCB]))  # 4Hz
time.sleep(0.1)
ser.write(bytes([0xA5, 0x35, 0x02, 0xDC]))  # framed auto
time.sleep(0.2)
ser.reset_input_buffer()

def read_frame():
    data = ser.read(FRAME_SIZE)
    if len(data) != FRAME_SIZE:
        return None
    if data[0] != 0x5A or data[1] != 0x5A:
        return None
    raw = data[4:4 + PIXELS * 2]
    temps = np.frombuffer(raw, np.int16).astype(np.float32) / 100.0
    return temps.reshape(24, 32)

# =========================
# ADVICE LOGIC
# =========================
def generate_advice(condition, severity):
    if condition == "normal":
        return (
            "No abnormal thermal pattern detected.",
            "Continue routine monitoring."
        )

    if condition == "mastitis_suspected":
        exp = "Elevated temperature detected in the udder region."
        act = "Inspect udder and milk quality."
    elif condition == "lameness_suspected":
        exp = "Localized heat detected in the leg or hoof."
        act = "Inspect hooves and reduce movement."
    else:
        exp = "Abnormal thermal pattern detected."
        act = "Recheck and observe closely."

    if severity == "HIGH":
        act = "Immediate attention recommended. " + act

    return exp, act

# =========================
# MAIN INFERENCE LOOP
# =========================
print("✅ Cow Health Inference Started (UART MLX90640)")
print("Waiting for Streamlit control input...")

try:
    while True:

        # -------- Body part from Streamlit --------
        try:
            body_part = json.loads(CONTROL_FILE.read_text()).get("body_part", "udder")
        except:
            body_part = "udder"

        if body_part not in PARTS:
            body_part = "udder"

        # -------- Capture frames --------
        frames = []
        while len(frames) < CAPTURE_FRAMES:
            f = read_frame()
            if f is not None:
                frames.append(f)

        stack = np.stack(frames)
        mean_frame = np.mean(stack, axis=0)

        # -------- Ambient conditions --------
        amb, hum = read_ambient()
        if amb is None or hum is None:
            time.sleep(1)
            continue

        # -------- Feature extraction (MATCH TRAINING) --------
        hot = mean_frame[mean_frame >= np.percentile(mean_frame, 70)]

        mean_t = np.mean(hot)
        max_t = np.max(hot)
        std_t = np.std(mean_frame)

        delta_mean = mean_t - amb
        delta_max = max_t - amb

        x = np.array([[delta_mean, delta_max, std_t, hum]])
        x_scaled = scalers[body_part].transform(x)

        # -------- GMM scoring --------
        score = gmm_models[body_part].score_samples(x_scaled)[0]
        threshold = thresholds[body_part]

        abnormal = score < threshold

        if not abnormal:
            condition = "normal"
            confidence = 0.0
        else:
            confidence = min(1.0, (threshold - score) / abs(threshold))
            if body_part == "udder":
                condition = "mastitis_suspected"
            elif body_part in ["leg", "hoof"]:
                condition = "lameness_suspected"
            else:
                condition = "abnormal_unspecified"

        severity = (
            "HIGH" if confidence > 0.6
            else "MEDIUM" if confidence > 0.3
            else "LOW"
        )

        explanation, action = generate_advice(condition, severity)

        # -------- Save visualization --------
        Tmin, Tmax = PART_TEMP_RANGES[body_part]
        img8 = np.clip(
            (mean_frame - Tmin) * 255 / (Tmax - Tmin),
            0, 255
        ).astype(np.uint8)

        img_color = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(IMAGE_FILE), img_color)

        # -------- Save result JSON --------
        result = {
            "timestamp": datetime.now().isoformat(),
            "body_part": body_part,
            "condition": condition,
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "gmm_score": round(float(score), 2),
            "explanation": explanation,
            "action": action
        }

        RESULT_FILE.write_text(json.dumps(result, indent=2))

        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping inference...")

finally:
    ser.close()
    dht.exit()
