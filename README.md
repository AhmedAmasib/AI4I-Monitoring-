# AI4I Predictive Maintenance Monitor

A Streamlit dashboard that scores CNC milling machine sensor readings with an XGBoost classifier and flags machine states as **healthy** or **critical**. The model was trained on AWS SageMaker using the public AI4I 2020 Predictive Maintenance dataset.

**[Live demo](https://bck7wzaajw9jaemrqfdqcf.streamlit.app/)** · **[Project report (PDF)](IOT%20report.pdf)**

---

## Problem

Unplanned downtime on CNC machines is expensive, and scheduled maintenance is often either too early or too late. This project explores predictive maintenance: using sensor data to estimate failure risk so an operator can act before a machine stops.

## What it does

- Upload a CSV or Parquet file of machine sensor data.
- The model returns a failure-risk score (0–100%) for every row.
- Rows above the risk threshold (0.5) are marked 🔴 **CRITICAL**, the rest 🟢 **HEALTHY**.
- Explore results in an interactive Plotly scatter plot (risk score vs. temperature, with hover details for every sensor value).
- Review a colour-graded results table and download the full report as CSV.

## Data and model

| Item | Details |
|------|---------|
| Dataset | [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (synthetic, 10,000 rows) |
| Sensor inputs | Air temperature, process temperature, rotational speed, torque, tool wear |
| Model | XGBoost 1.7.6, `binary:logistic`, 50 boosting rounds, 12 input features |
| Training | AWS SageMaker built-in XGBoost (job `sagemaker-xgboost-2025-12-19-08-50-24-690`) |
| Inference | Runs locally inside the Streamlit app. No SageMaker endpoint is called |

## Evaluation

Model evaluation is being re-done on a stratified hold-out set. Failures are rare in this dataset (about 3.4% of rows), so accuracy is not a useful metric. The results below will be filled in from the evaluation script:

| Metric | Result |
|--------|--------|
| Precision (failure class) | _pending_ |
| Recall (failure class) | _pending_ |
| F1 (failure class) | _pending_ |
| PR-AUC | _pending_ |
| Inference latency (P95, single row) | _pending_ |

## Input format

The saved model expects the same 12 columns, in the same order, as the data it was trained on. The sensor columns were standardized before training, and the app does **not** scale or validate uploads. Uploading the raw AI4I CSV directly will not give meaningful results. Preprocessing is being moved into the model package so raw data can be uploaded safely (see Roadmap).

## Tech stack

- **Python 3.11**
- **XGBoost 1.7.6** for the classifier
- **Streamlit** for the dashboard
- **Plotly** and **pandas** for charts and data handling
- **AWS SageMaker** for model training

## Project structure

```
AI4I-Monitoring-/
├── cnc_maintenance/
│   ├── app.py              # Streamlit dashboard
│   └── extract_model.py    # Unpacks models/model.tar.gz (SageMaker output)
├── models/
│   ├── model.tar.gz        # Model archive downloaded from SageMaker
│   └── xgboost-model       # Extracted model used by the app
├── .devcontainer/          # Codespaces / VS Code dev container config
├── requirements.txt
└── IOT report.pdf          # Full project report
```

## Run locally

```bash
git clone https://github.com/AhmedAmasib/AI4I-Monitoring-.git
cd AI4I-Monitoring-
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run cnc_maintenance/app.py
```

The app loads `models/xgboost-model` directly. You only need `extract_model.py` if you download a new `model.tar.gz` from SageMaker:

```bash
python cnc_maintenance/extract_model.py
```

> **Note:** the saved model uses XGBoost's legacy binary format, which newer XGBoost releases (3.1+) can no longer read. Keep `xgboost==1.7.6` until the model is re-exported as JSON/UBJ.

## Roadmap

- [ ] Add the training and evaluation code, plus a reproducible train/test split
- [ ] Report precision, recall, F1 and PR-AUC, and measure inference latency
- [ ] Review the input features to make sure the model predicts from sensor readings only
- [ ] Bundle preprocessing (scaling) with the model and validate uploaded columns
- [ ] Re-export the model as JSON/UBJ
- [ ] Add a sample input file and dashboard screenshots
- [ ] Tune the risk threshold and show feature importance in the dashboard
- [ ] Add unit tests and a CI workflow

## Team

University group project. Primary contributor: [Ahmed Mohamed](https://github.com/AhmedAmasib). Supervised by Prof. Mehmet Akyol.
