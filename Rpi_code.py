import serial
import time
import numpy as np
import cv2
import json
import os
from datetime import datetime

import board
import adafruit_dht

# ======================================================
# PATHS
# ======================================================
BASE_DIR = "/home/raif/COW_Health"
LIVE_JSON = os.path.join(BASE_DIR, "live_input.json")
FLAG_FILE = os.path.join(BASE_DIR, "live_request.flag")
IMAGE_DIR = os.path.join(BASE_DIR, "live_images")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ======================================================
# MLX90640 SERIAL
# ======================================================
PORT = "/dev/ttyAMA0"
BAUD = 115200
FRAME_SIZE = 1544
PIXELS = 32 * 24

CAPTURE_FRAMES = 40
OUT_W, OUT_H = 320, 240

CURRENT_PART = "udder"

PART_TEMP_RANGES = {
    "eye":   (30.0, 38.0),
    "udder": (28.0, 40.0),
    "leg":   (26.0, 36.0),
    "hoof":  (24.0, 34.0),
    "etc":   (25.0, 40.0),
}

# ======================================================
# DHT22
# ======================================================
dht = adafruit_dht.DHT22(board.D4, use_pulseio=False)
last_env = {"t": None, "h": None, "time": 0}

def read_ambient_safe():
    now = time.time()
    if now - last_env["time"] < 3:
        return last_env["t"], last_env["h"]
    try:
        t = dht.temperature
        h = dht.humidity
        if t is not None and h is not None:
            last_env.update({"t": round(t,2), "h": round(h,2), "time": now})
    except RuntimeError:
        pass
    return last_env["t"], last_env["h"]

# ======================================================
# SERIAL INIT
# ======================================================
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(1)
ser.write(bytes([0xA5, 0x25, 0x01, 0xCB]))
time.sleep(0.1)
ser.write(bytes([0xA5, 0x35, 0x02, 0xDC]))
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

def render_display(frame, part):
    fup = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
    tmin, tmax = PART_TEMP_RANGES[part]
    img8 = np.clip((fup - tmin) * 255 / (tmax - tmin), 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    return cv2.flip(img, 1)

def extract_features(frames, ambient):
    stack = np.stack(frames).astype(np.float32)
    mean_frame = np.mean(stack, axis=0)
    hot = mean_frame[mean_frame >= np.percentile(mean_frame, 70)]

    mean_temp = float(np.mean(hot))
    max_temp = float(np.max(hot))
    frame_std = float(np.std(mean_frame))

    return {
        "delta_mean": round(mean_temp - ambient, 2),
        "delta_max": round(max_temp - ambient, 2),
        "frame_std": round(frame_std, 2)
    }

# ======================================================
# LIVE LOOP
# ======================================================
cv2.namedWindow("Cow Thermal Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cow Thermal Live", 640, 480)

print("Pi live system ready — waiting for web request")

try:
    while True:
        frame = read_frame()
        if frame is None:
            continue

        cv2.imshow("Cow Thermal Live", render_display(frame, CURRENT_PART))
        cv2.waitKey(1)

        # ---- Trigger from Streamlit ----
        if os.path.exists(FLAG_FILE):
            amb, hum = read_ambient_safe()
            if amb is None:
                continue

            frames = []
            while len(frames) < CAPTURE_FRAMES:
                f = read_frame()
                if f is not None:
                    frames.append(f)

            feats = extract_features(frames, amb)

            # Save 3 images
            idxs = [0, CAPTURE_FRAMES//2, CAPTURE_FRAMES-1]
            for i, idx in enumerate(idxs, 1):
                cv2.imwrite(
                    os.path.join(IMAGE_DIR, f"live_{i}.png"),
                    render_display(frames[idx], CURRENT_PART)
                )

            live_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cow_part": CURRENT_PART,
                "ambient_temp": amb,
                "humidity": hum,
                **feats
            }

            with open(LIVE_JSON, "w") as f:
                json.dump(live_data, f, indent=2)

            os.remove(FLAG_FILE)
            print("Live capture completed")

finally:
    ser.close()
    dht.exit()
    cv2.destroyAllWindows()
