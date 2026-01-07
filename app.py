import streamlit as st
st.set_page_config(page_title="Cow Health Monitoring", layout="wide")

# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import joblib
import json
import os
import base64
from datetime import datetime
import time

# =========================
# BACKGROUND (STREAMLIT-SAFE)
# =========================
def set_background(image_path):
    if not os.path.exists(image_path):
        st.warning("Background image not found. Running without background.")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background: rgba(255, 255, 255, 0.92);
            padding: 2.5rem;
            border-radius: 18px;
            max-width: 1200px;
            margin-top: 2rem;
        }}

        /* YOUR ORIGINAL GLOBAL RULE (UNCHANGED) */
        .block-container * {{
            color: #000000 !important;
        }}

        h1, h2, h3, h4 {{
            font-weight: 800 !important;
        }}

        div.stButton > button {{
            background-color: #2ecc71 !important;
            color: #ffffff !important;
            font-weight: 700;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            border: none;
        }}

        /* =========================
           STYLE FIXES (ONLY)
        ========================= */

        /* Score box */
        .block-container .score-box {{
            background: #1f2937 !important;
            padding: 16px;
            border-radius: 14px;
            margin-top: 12px;
            font-size: 16px;
        }}

        .block-container .score-box * {{
            color: #ffffff !important;
        }}

        .block-container .score-box .label {{
            font-weight: 700;
            color: #93c5fd !important;
        }}

        /* Selectbox */
        .block-container div[data-testid="stSelectbox"] > div {{
            background-color: #1f2937 !important;
            border-radius: 10px;
            border: 1px solid #374151;
        }}

        .block-container div[data-testid="stSelectbox"] * {{
            color: #ffffff !important;
            font-weight: 700;
        }}

        ul[role="listbox"] {{
            background-color: #1f2937 !important;
        }}

        li[role="option"] {{
            color: #ffffff !important;
        }}

        li[role="option"]:hover {{
            background-color: #2ecc71 !important;
            color: #000000 !important;
        }}

        footer {{
            visibility: hidden;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# =========================
# CONFIG
# =========================
ART_DIR = "artifacts"
FEATURES = ["delta_mean", "delta_max", "frame_std"]
LIVE_FILE = "live_input.json"
FLAG_FILE = "capture.flag"
IMAGE_DIR = "images"

# =========================
# LOAD MODEL ARTIFACTS
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    gmm_models = {}
    scalers = {}

    for part in ["udder", "eye", "leg", "hoof", "etc"]:
        gmm_models[part] = joblib.load(f"{ART_DIR}/gmm_models/gmm_{part}.joblib")
        scalers[part] = joblib.load(f"{ART_DIR}/scalers/scaler_{part}.joblib")

    with open(f"{ART_DIR}/config/thresholds.json") as f:
        thresholds = json.load(f)

    return gmm_models, scalers, thresholds

gmm_models, scalers, thresholds = load_artifacts()

# =========================
# MODEL FUNCTIONS
# =========================
def compute_gmm_score(part, delta_mean, delta_max, frame_std):
    x = pd.DataFrame([[delta_mean, delta_max, frame_std]], columns=FEATURES)
    x_scaled = scalers[part].transform(x)
    return gmm_models[part].score_samples(x_scaled)[0]

def interpret_score(score, threshold):
    margin = score - threshold
    if score < threshold - 20:
        return "INVALID", "No valid cow thermal signal detected.", 0.0
    if score < threshold:
        return "ABNORMAL", "Abnormal thermal pattern detected.", min(1.0, abs(margin) / 10)
    return "NORMAL", "Thermal pattern consistent with a healthy cow.", min(1.0, margin / 5)

# =========================
# UI HELPERS (UNCHANGED)
# =========================
def result_card(status, message):
    colors = {
        "NORMAL": "#e6f4ea",
        "ABNORMAL": "#fdecea",
        "INVALID": "#fff4e5"
    }
    icons = {
        "NORMAL": "✅",
        "ABNORMAL": "⚠️",
        "INVALID": "❌"
    }

    st.markdown(
        f"""
        <div style="background:{colors[status]}; padding:22px;
                    border-radius:16px; margin-top:20px;">
            <h2>{icons[status]} {status}</h2>
            <p style="font-size:18px;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def confidence_bar(conf):
    st.markdown(
        f"""
        <div style="margin-top:15px;">
            <strong>Confidence</strong>
            <div style="background:#ddd; border-radius:10px; height:20px;">
                <div style="width:{int(conf*100)}%;
                            background:#2ecc71; height:100%;"></div>
            </div>
            <small>{int(conf*100)}%</small>
        </div>
        """,
        unsafe_allow_html=True
    )

def info_card(label, value, icon):
    st.markdown(
        f"""
        <div style="background:white; padding:15px;
                    border-radius:14px; text-align:center;">
            <h3>{icon}</h3>
            <h4>{label}</h4>
            <p style="font-size:20px;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# HEADER
# =========================
st.title("🐄 Cow Health Monitoring Dashboard")
st.caption("Thermal-based early warning system for cow health and milk production")
st.divider()

mode = st.radio("Select Mode", ["Manual Analysis", "Live Raspberry Pi Feed"], horizontal=True)

# ======================================================
# MANUAL MODE
# ======================================================
if mode == "Manual Analysis":

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📥 Manual Thermal Input")

        part = st.selectbox("Body Part", ["udder", "eye", "leg", "hoof", "etc"])
        delta_mean = st.slider("Δ Mean Temperature (°C)", 5.0, 20.0, 12.0)
        delta_max  = st.slider("Δ Max Temperature (°C)", 8.0, 25.0, 16.0)
        frame_std  = st.slider("Thermal Variability (Std)", 2.0, 10.0, 5.0)

        run = st.button("Run Analysis")

    with col2:
        st.subheader("📊 Detection Result")

        if run:
            score = compute_gmm_score(part, delta_mean, delta_max, frame_std)
            threshold = float(thresholds[part])
            status, msg, conf = interpret_score(score, threshold)

            result_card(status, msg)

            st.markdown(
                f"""
                <div class="score-box">
                    <div><span class="label">Body Part:</span> {part}</div>
                    <div><span class="label">GMM Score:</span> {score:.2f}</div>
                    <div><span class="label">Threshold:</span> {threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence_bar(conf)

            if status == "ABNORMAL":
                st.info("🩺 Inspect the affected body part. Consult a veterinarian.")
            elif status == "NORMAL":
                st.info("🐄 Cow appears healthy. No action required.")
            else:
                st.info("📷 Ensure a cow body part is clearly visible.")

# ======================================================
# LIVE MODE
# ======================================================
else:
    st.subheader("📡 Live Raspberry Pi Thermal Feed")

    if st.button("🔄 Check Again"):
        with open(FLAG_FILE, "w") as f:
            f.write("1")
        st.info("📡 Capture request sent…")
        time.sleep(3)
        st.rerun()

    if not os.path.exists(LIVE_FILE):
        st.warning("Waiting for live data from Raspberry Pi…")
        st.stop()

    with open(LIVE_FILE) as f:
        live = json.load(f)

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Ambient Temp (°C)", live["ambient_temp"], "🌡")
    with c2:
        info_card("Humidity (%)", live["humidity"], "💧")
    with c3:
        info_card("Timestamp", live["timestamp"], "⏱")

    st.divider()

    part = live["cow_part"]
    score = compute_gmm_score(part, live["delta_mean"], live["delta_max"], live["frame_std"])
    threshold = float(thresholds[part])
    status, msg, conf = interpret_score(score, threshold)

    result_card(status, msg)

    st.markdown(
        f"""
        <div class="score-box">
            <div><span class="label">Body Part:</span> {part}</div>
            <div><span class="label">GMM Score:</span> {score:.2f}</div>
            <div><span class="label">Threshold:</span> {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    confidence_bar(conf)

    if status == "ABNORMAL":
        st.info("🩺 Immediate inspection recommended.")
    elif status == "NORMAL":
        st.info("🐄 Cow appears healthy. No action required.")
    else:
        st.info("📷 No valid cow thermal signal detected.")
