import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Cow Health Monitoring Dashboard",
    layout="centered"
)

st.title("🐄 Cow Health Monitoring Dashboard")
st.caption(
    "Thermal-based early warning system for cow health and milk production support"
)

st.divider()

# =========================================================
# Load model artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    gmm_models = {}
    scalers = {}

    # Load GMM models
    for fname in os.listdir("artifacts/gmm_models"):
        part = fname.replace("gmm_", "").replace(".joblib", "")
        gmm_models[part] = joblib.load(f"artifacts/gmm_models/{fname}")

    # Load scalers
    for fname in os.listdir("artifacts/scalers"):
        part = fname.replace("scaler_", "").replace(".joblib", "")
        scalers[part] = joblib.load(f"artifacts/scalers/{fname}")

    # Load thresholds
    with open("artifacts/config/thresholds.json", "r") as f:
        thresholds = json.load(f)

    return gmm_models, scalers, thresholds


gmm_models, scalers, thresholds = load_artifacts()

FEATURES = ["delta_mean", "delta_max", "frame_std", "humidity"]

# =========================================================
# Helper functions
# =========================================================
def compute_gmm_score(row):
    part = row["cow_part"]

    if part not in gmm_models:
        return None

    x = pd.DataFrame(
        [[row[f] for f in FEATURES]],
        columns=FEATURES
    )

    x_scaled = scalers[part].transform(x)
    return gmm_models[part].score_samples(x_scaled)[0]


def infer_condition(row):
    if row["abnormal"] == 0:
        return "Normal", "NONE", 0.0, "No action needed."

    confidence = round(min(1.0, abs(row["gmm_score"]) / 6), 2)

    if row["cow_part"] == "udder":
        return (
            "Mastitis Suspected",
            "HIGH" if confidence > 0.6 else "MEDIUM",
            confidence,
            "Inspect udder, increase milking frequency, consult veterinarian."
        )

    if row["cow_part"] in ["hoof", "leg"]:
        return (
            "Lameness Suspected",
            "HIGH" if confidence > 0.6 else "MEDIUM",
            confidence,
            "Inspect hooves and legs, reduce movement, consult veterinarian."
        )

    if row["cow_part"] in ["eye", "body"]:
        return (
            "Fever or Infection Suspected",
            "MEDIUM",
            confidence,
            "Monitor cow closely and consult veterinarian if temperature remains high."
        )

    return (
        "Abnormal (Unspecified)",
        "LOW",
        confidence,
        "Monitor cow and recheck later."
    )

# =========================================================
# Mode selection
# =========================================================
st.subheader("⚙️ Input Mode")

mode = st.radio(
    "Select data source:",
    ["Manual Input (Demo Mode)", "Live Sensor Input (Raspberry Pi)"]
)

st.divider()

# =========================================================
# MODE 1 — MANUAL INPUT
# =========================================================
if mode == "Manual Input (Demo Mode)":

    st.subheader("📥 Manual Thermal Feature Input")

    with st.form("manual_input"):
        cow_part = st.selectbox(
            "Body Part",
            ["udder", "hoof", "leg", "eye", "body"]
        )

        delta_mean = st.number_input("Δ Mean Temperature (°C)", value=1.5)
        delta_max = st.number_input("Δ Max Temperature (°C)", value=2.5)
        frame_std = st.number_input("Thermal Variability (Std)", value=2.0)
        humidity = st.number_input("Humidity (%)", value=70.0)

        submitted = st.form_submit_button("Analyze Cow Health")

    if submitted:
        row = {
            "cow_part": cow_part,
            "delta_mean": delta_mean,
            "delta_max": delta_max,
            "frame_std": frame_std,
            "humidity": humidity
        }

        gmm_score = compute_gmm_score(row)
        abnormal = int(gmm_score < thresholds[cow_part])

        row["gmm_score"] = gmm_score
        row["abnormal"] = abnormal

        condition, severity, confidence, advice = infer_condition(row)

        st.divider()
        st.subheader("🩺 Health Assessment Result")

        if severity == "NONE":
            st.success("✅ Cow Condition: NORMAL")
        elif severity == "HIGH":
            st.error("🚨 HIGH RISK DETECTED")
        else:
            st.warning("⚠️ Attention Required")

        st.markdown(f"""
        **Detected Condition:** {condition}  
        **Affected Body Part:** {cow_part.capitalize()}  
        **Severity Level:** {severity}
        """)

        st.markdown("### 🔎 Detection Confidence")
        st.progress(confidence)
        st.caption(f"Confidence: **{int(confidence * 100)}%**")

        st.info(f"**Recommended Action:** {advice}")

# =========================================================
# MODE 2 — LIVE SENSOR INPUT (PI → FILE)
# =========================================================
else:
    st.subheader("📡 Live Cow Health Status (From Raspberry Pi)")

    RESULT_FILE = Path("latest_result.json")

    if RESULT_FILE.exists():
        with open(RESULT_FILE, "r") as f:
            result = json.load(f)

        severity = result.get("severity", "NONE")
        confidence = result.get("confidence", 0.0)

        if severity == "NONE":
            st.success("✅ Cow Condition: NORMAL")
        elif severity == "HIGH":
            st.error("🚨 HIGH RISK DETECTED")
        else:
            st.warning("⚠️ Attention Required")

        st.markdown(f"""
        **Detected Condition:** {result.get("condition", "Unknown").replace('_',' ').title()}  
        **Body Part:** {result.get("cow_part", "Unknown").title()}  
        **Severity Level:** {severity}  
        **Timestamp:** {result.get("timestamp", "N/A")}
        """)

        st.markdown("### 🔎 Detection Confidence")
        st.progress(confidence)
        st.caption(f"Confidence: **{int(confidence * 100)}%**")

    else:
        st.info("Waiting for live sensor data from Raspberry Pi...")
