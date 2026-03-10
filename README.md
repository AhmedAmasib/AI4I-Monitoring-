# 🔧 AI4I Predictive Maintenance Monitor

> Real-time machine failure detection using vibration and temperature data — deployed on AWS SageMaker with a live Streamlit dashboard.

---

## 🚨 The Problem
Industrial machines fail unexpectedly, causing costly downtime and quality issues. Traditional maintenance is reactive — you fix it after it breaks. This project shifts that to **predictive maintenance** — detecting failures before they happen.

---

## 🎯 What We Built
A machine learning pipeline that monitors CNC machine sensor data in real time to predict two failure types:
- **HDF** — Heat Dissipation Failure
- **TWF** — Tool Wear Failure

---

## ✅ Results & KPIs
| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score (anomaly detection) | ≥ 0.85 | ✅ |
| P95 Inference Latency | ≤ 50ms | ✅ |

---

## 🛠️ Tech Stack
- **Python** — data processing & modelling
- **AWS SageMaker** — model training & deployment
- **Streamlit** — live monitoring dashboard
- **Scikit-learn / ML models** — failure classification

---

## 🚀 Live Demo
👉 [Click here to open the live dashboard](https://bck7wzaajw9jaemrqfdqcf.streamlit.app/)

---

## 📁 Project Structure
```
AI4I-Monitoring/
├── cnc_maintenance/   # Core ML pipeline & data processing
├── models/            # Trained model artifacts
├── requirements.txt   # Dependencies
└── IOT report.pdf     # Full project report
```

---

## 🏃 Run Locally
```bash
git clone https://github.com/AhmedAmasib/AI4I-Monitoring-
cd AI4I-Monitoring-
pip install -r requirements.txt
streamlit run app.py
```

---

## 👥 Team
Built as a group project — primary contributor: Ahmed Mohamed
