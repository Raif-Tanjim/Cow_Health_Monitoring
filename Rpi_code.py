import serial
import time
import json
import cv2
import numpy as np
from datetime import datetime
import joblib
import board
import adafruit_dht

# ================= PATHS =================
BASE = "/home/raif/shared"
CONTROL_FILE = f"{BASE}/control/body_part.json"
OUTPUT_JSON = f"{BASE}/output/latest_result.json"
OUTPUT_IMG = f"{BASE}/output/latest_image.png"

ARTIFACTS = "/home/raif/COW_Health/artifacts"

# ================= CONSTANTS =================
PORT = "/dev/ttyAMA0"
BAUD = 115200
FRAME_SIZE = 1544
PIXELS = 32 * 24
CAPTURE_FRAMES = 20

FEATURES = ["delta_mean", "delta_max", "frame_std", "humidity"]
DEFAULT_PART = "udder"

PART_TEMP_RANGES = {
    "eye":   (30.0, 38.0),
    "udder": (28.0, 40.0),
    "leg":   (26.0, 36.0),
    "hoof":  (24.0, 34.0),
    "etc":   (25.0, 40.0),
}

# ================= LOAD MODELS =================
PARTS = ["udder", "eye", "leg", "hoof", "etc"]
gmm_models = {}
scalers = {}

for p in PARTS:
    gmm_models[p] = joblib.load(f"{ARTIFACTS}/gmm_models/gmm_{p}.joblib")
    scalers[p] = joblib.load(f"{ARTIFACTS}/scalers/scaler_{p}.joblib")

with open(f"{ARTIFACTS}/config/thresholds.json") as f:
    thresholds = json.load(f)

# ================= BODY PART FROM STREAMLIT =================
def get_body_part():
    try:
        with open(CONTROL_FILE) as f:
            return json.load(f).get("body_part", DEFAULT_PART)
    except:
        return DEFAULT_PART

# ================= DHT22 =================
dht = adafruit_dht.DHT22(board.D4, use_pulseio=False)

def read_ambient():
    try:
        return dht.temperature, dht.humidity
    except:
        return None, None

# ================= SERIAL INIT =================
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(1)
ser.write(bytes([0xA5, 0x25, 0x01, 0xCB]))  # 4Hz
time.sleep(0.1)
ser.write(bytes([0xA5, 0x35, 0x02, 0xDC]))
time.sleep(0.2)
ser.reset_input_buffer()

def read_frame():
    data = ser.read(FRAME_SIZE)
    if len(data) != FRAME_SIZE or data[0] != 0x5A:
        return None
    raw = data[4:4 + PIXELS * 2]
    temps = np.frombuffer(raw, np.int16).astype(np.float32) / 100.0
    return temps.reshape(24, 32)

# ================= ADVICE =================
def generate_advice(condition, severity):
    if condition == "normal":
        return "No abnormal thermal pattern detected.", "Continue routine monitoring."

    if condition == "mastitis_suspected":
        exp = "Elevated udder temperature detected."
        act = "Inspect udder and milk quality."
    elif condition == "lameness_suspected":
        exp = "Localized heat detected in leg or hoof."
        act = "Inspect hooves and reduce movement."
    else:
        exp = "Abnormal thermal pattern detected."
        act = "Recheck and observe closely."

    if severity == "HIGH":
        act = "Immediate attention recommended. " + act

    return exp, act

# ================= MAIN LOOP =================
while True:

    body_part = get_body_part()

    frames = []
    while len(frames) < CAPTURE_FRAMES:
        f = read_frame()
        if f is not None:
            frames.append(f)

    stack = np.stack(frames)
    mean_frame = np.mean(stack, axis=0)

    amb, hum = read_ambient()
    if amb is None or hum is None:
        time.sleep(1)
        continue

    # ---- FEATURE EXTRACTION (MATCH TRAINING) ----
    hot = mean_frame[mean_frame >= np.percentile(mean_frame, 70)]

    mean_t = np.mean(hot)
    max_t = np.max(hot)
    std_t = np.std(mean_frame)

    delta_mean = mean_t - amb
    delta_max = max_t - amb

    x = np.array([[delta_mean, delta_max, std_t, hum]])
    x_scaled = scalers[body_part].transform(x)

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

    severity = "HIGH" if confidence > 0.6 else "MEDIUM" if confidence > 0.3 else "LOW"
    explanation, action = generate_advice(condition, severity)

    # ---- DISPLAY IMAGE ----
    Tmin, Tmax = PART_TEMP_RANGES[body_part]
    img8 = np.clip((mean_frame - Tmin) * 255 / (Tmax - Tmin), 0, 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(OUTPUT_IMG, img_color)

    # ---- SAVE RESULT ----
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

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    time.sleep(2)
