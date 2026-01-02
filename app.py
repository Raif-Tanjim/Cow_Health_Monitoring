import streamlit as st

# ================= MUST BE FIRST STREAMLIT COMMAND =================
st.set_page_config(
    page_title="Cow Health Monitoring",
    layout="centered"
)

# ================= IMPORTS =================
import json
import time
import numpy as np
import joblib
from pathlib import Path

# ================= CONFIG =================
ARTIFACTS = "artifacts"
PARTS = ["udder", "eye", "leg", "hoof", "etc"]
FEATURES = ["delta_mean", "delta_max", "frame_std", "humidity"]

CONTROL_DIR = Path("shared/control")
OUTPUT_DIR = Path("shared/output")
CONTROL_FILE = CONTROL_DIR / "body_part.json"
RESULT_FILE = OUTPUT_DIR / "latest_result.json"
IMAGE_FILE = OUTPUT_DIR / "latest_image.png"

# ================= SAFE DIRECTORY CREATION =================
CONTROL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= LOAD MODELS (MANUAL MODE) =================
@st.cache_resource
def load_models():
    gmm_models = {}
    scalers = {}

    for p in PARTS:
        gmm_models[p] = joblib.load(f"{ARTIFACTS}/gmm_models/gmm_{p}.joblib")
        scalers[p] = joblib.load(f"{ARTIFACTS}/scalers/scaler_{p}.joblib")

    with open(f"{ARTIFACTS}/config/thresholds.json") as f:
        thresholds = json.load(f)

    return gmm_models, scalers, thresholds


gmm_models, scalers, thresholds = load_models()

# ================= ADVICE LOGIC =================
def generate_advice(condition, severity):
    if condition == "normal":
        return (
            "No abnormal thermal pattern detected.",
            "Continue routine monitoring."
        )

    if condition == "mastitis_suspected":
        explanation = "Elevated udder temperature detected."
        action = "Inspect udder and milk quality."
    elif condition == "lameness_suspected":
        explanation = "Localized heat detected in leg or hoof."
        action = "Inspect hooves and reduce movement."
    else:
        explanation = "Abnormal thermal pattern detected."
        action = "Recheck and observe closely."

    if severity == "HIGH":
        action = "Immediate attention recommended. " + action

    return explanation, action

# ================= UI =================
st.title("🐄 Cow Health Monitoring Dashboard")
st.caption("Thermal-based early warning system (Edge AI)")

mode = st.radio(
    "Select operation mode",
    ["Manual Inspection", "Live Sensor Mode"],
    horizontal=True
)

# ==========================================================
# 🟢 MANUAL INSPECTION MODE
# ==========================================================
if mode == "Manual Inspection":

    st.subheader("Manual Thermal Feature Input")

    body_part = st.selectbox("Body Part", PARTS)

    delta_mean = st.number_input("Δ Mean Temperature (°C)", value=1.5)
    delta_max = st.number_input("Δ Max Temperature (°C)", value=2.5)
    frame_std = st.number_input("Thermal Variability (Std)", value=2.0)
    humidity = st.number_input("Humidity (%)", value=70.0)

    if st.button("Run Health Assessment"):

        x = np.array([[delta_mean, delta_max, frame_std, humidity]])
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

        severity = (
            "HIGH" if confidence > 0.6
            else "MEDIUM" if confidence > 0.3
            else "LOW"
        )

        explanation, action = generate_advice(condition, severity)

        st.divider()
        st.subheader("Assessment Result")

        st.metric("Condition", condition)
        st.metric("Severity", severity)
        st.metric("Confidence", f"{confidence*100:.1f}%")

        st.info(explanation)
        st.warning(action)

# ==========================================================
# 🔵 LIVE SENSOR MODE
# ==========================================================
else:

    st.subheader("Live Raspberry Pi Monitoring")

    body_part = st.selectbox(
        "Body Part Being Captured",
        PARTS,
        index=0
    )

    # Write control file ONLY in Live mode
    with open(CONTROL_FILE, "w") as f:
        json.dump({"body_part": body_part}, f)

    placeholder = st.empty()

    while True:
        try:
            with open(RESULT_FILE) as f:
                result = json.load(f)

            with placeholder.container():

                if IMAGE_FILE.exists():
                    st.image(
                        str(IMAGE_FILE),
                        caption="Live Thermal View",
                        use_column_width=True
                    )

                st.metric("Condition", result["condition"])
                st.metric("Severity", result["severity"])
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")

                st.info(result["explanation"])
                st.warning(result["action"])

        except FileNotFoundError:
            st.info("Waiting for Raspberry Pi data...")

        time.sleep(2)
