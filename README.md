🐄 Cow Health Monitoring (Thermal + Edge AI)

This project demonstrates a low-cost cow health monitoring system using thermal imaging, lightweight machine learning, and edge computing.

It uses an MLX90640 thermal sensor and a Raspberry Pi to detect abnormal temperature patterns related to cow health and milk production.

All inference is designed to run locally on edge devices.

🔍 What This Project Does

Collects thermal data from different cow body parts

Extracts temperature-based features from thermal frames

Learns normal temperature patterns from healthy cows

Detects abnormal thermal behavior as an early warning

Identifies likely health issues such as:

Mastitis (udder)

Lameness (hoof / leg)

Fever or infection (eye / body)

Displays results in a simple web dashboard

🧠 How It Works (High Level)

Thermal frames are captured using the MLX90640 sensor

Statistical features are extracted:

Mean temperature

Maximum temperature

Temperature variability

An unsupervised anomaly detection model (Gaussian Mixture Model) learns normal patterns

The system outputs:

Health status

Severity

Confidence level

Basic advice

🖥️ Dashboard

The Streamlit dashboard supports two modes:

Manual Input Mode – for testing and demonstration

Live Sensor Mode – reads real-time output from the Raspberry Pi

🧰 Tech Stack

Python

MLX90640 thermal sensor

Raspberry Pi

Scikit-learn (GMM, One-Class SVM)

Streamlit

📁 Repository Structure
├── app.py                  # Streamlit dashboard
├── Rpi_code.py             # Raspberry Pi inference script
├── artifacts/              # Trained models and thresholds
├── *.ipynb                 # Data analysis and modeling notebooks
├── data_combined.csv
├── synthetic_cow_thermal_data.csv
├── requirements.txt
└── README.md

🚀 How to Run
Install dependencies
pip install -r requirements.txt

Run the dashboard
streamlit run app.py

Run inference on Raspberry Pi
python Rpi_code.py

📌 Notes

Designed for low-cost deployment

Works fully offline

Suitable for early disease detection in dairy farms

📄 License

This project is for research and educational purposes.
