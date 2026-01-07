🐄 Cow Health Monitoring (Thermal + Edge AI)

This project demonstrates a low-cost cow health monitoring system using thermal data, lightweight machine learning, and edge computing.

It uses a MLX90640 thermal sensor and a Raspberry Pi to detect abnormal temperature patterns related to cow health and milk production.

🔍 What This Project Does

Collects thermal data from different cow body parts

Extracts simple temperature features from thermal frames

Learns normal temperature patterns from healthy cows

Detects abnormal thermal behavior (early warning)

Identifies likely health issues such as:

mastitis (udder)

lameness (hoof / leg)

fever or infection (eye / body)

Displays results in a simple web dashboard

All inference is designed to run locally on edge devices.

🧠 How It Works (High Level)

Thermal frames are captured using the MLX90640 sensor

Statistical features are extracted (mean, max, variability)

An unsupervised model (Gaussian Mixture Model) detects anomalies

The system outputs:

health status

severity

confidence level

basic advice

🖥️ Dashboard

The Streamlit dashboard supports two modes:

Manual Input Mode – for testing and demonstration

Live Sensor Mode – reads real-time output from Raspberry Pi

🧰 Tech Stack

Python

MLX90640 thermal sensor

Raspberry Pi

Scikit-learn (GMM, OC-SVM)

Streamlit

📁 Repository Structure
├── app.py                 # Streamlit dashboard
├── Rpi_code.py            # Raspberry Pi inference script
├── artifacts/             # Trained models and thresholds
├── *.ipynb                # Data analysis and modeling
├── data_combined.csv
├── synthetic_cow_thermal_data.csv
├── requirements.txt
└── README.md

🚀 How to Run
pip install -r requirements.txt
streamlit run app.py


For Raspberry Pi inference:

python Rpi_code.py
