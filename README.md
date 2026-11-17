# AI4I Predictive Maintenance Monitor

A Streamlit dashboard that scores CNC milling machine sensor readings with an XGBoost classifier and flags machine states as **healthy** or **critical**. It uses the public AI4I 2020 Predictive Maintenance dataset. The first model was trained on AWS SageMaker. It has been retrained to remove label leakage (see [Evaluation](#evaluation)).

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
| Target | `Machine failure` (339 failures, 3.4% of rows) |
| Inputs | Product type (L/M/H), air temperature, process temperature, rotational speed, torque, tool wear |
| Engineered features | Temperature difference (process − air), mechanical power (torque × angular speed), tool wear × torque |
| Excluded columns | `UDI`, `Product ID`, and the failure-mode labels `TWF`, `HDF`, `PWF`, `OSF`, `RNF` |
| Model | XGBoost classifier (`models/model.json`), class-weighted for imbalance, early stopping on a validation set |

## Evaluation

Training and evaluation live in [`cnc_maintenance/train.py`](cnc_maintenance/train.py). The data is split 60/20/20 into train, validation and test sets (stratified, seed 42). The validation set is used for early stopping and to choose the risk threshold. The test set is used once, for the final numbers. All results are saved to [`models/metrics.json`](models/metrics.json).

Failures are only 3.4% of rows, so accuracy is not a useful metric. A model that always says "healthy" would score 96.6%.

| Metric (failure class) | Test set (2,000 rows) | 5-fold cross-validation |
|--------|--------|--------|
| Precision | 0.96 | — |
| Recall | 0.74 | — |
| F1 | **0.83** | 0.87 ± 0.02 |
| PR-AUC | 0.89 | 0.89 ± 0.02 |
| ROC-AUC | 0.97 | — |

On the test set the model caught 50 of 68 failures and raised 2 false alarms out of 1,932 healthy rows (risk threshold 0.95, chosen on the validation set).

**Recall by failure mode (test set):**

| Failure mode | Caught |
|---|---|
| Power failure (PWF) | 13 / 13 |
| Overstrain failure (OSF) | 14 / 16 |
| Heat dissipation failure (HDF) | 25 / 29 |
| Tool wear failure (TWF) | 1 / 10 |

Tool wear failures are the weak spot. In AI4I, the tool is replaced or fails at a random point between 200 and 240 minutes of wear, so these failures are largely unpredictable from a single reading. Two further test failures have no failure mode recorded and were not caught.

**Latency:** feature engineering plus prediction for a single row takes about 2.4 ms at P50 and 3.3 ms at P95 on a CPU, measured over 950 calls.

Results were produced with Python 3.13, XGBoost 1.7.6 and scikit-learn 1.9.1. Other library versions can shift the numbers slightly. The versions used are recorded in `metrics.json`.

### Why the model was retrained

The first model (`models/xgboost-model`, trained on SageMaker) used the failure-mode columns `TWF`, `HDF`, `PWF` and `OSF` as inputs. `Machine failure` is defined from those columns, so the model was reading the answer rather than predicting it. Re-scoring it on the full dataset gives an F1 of 0.99 with those columns included, and an F1 of 0.01 (recall 0.3%) when they are set to zero. In other words, it learned almost nothing from the sensor readings. The new model uses sensor readings and product type only.

## Input format

The dashboard currently still loads the original model, which expects 12 pre-processed columns in a fixed order. Moving the dashboard to the new model, which accepts the raw AI4I columns, is the next step (see Roadmap). `cnc_maintenance/features.py` already does the conversion and checks for missing columns.

## Tech stack

- **Python 3.11**
- **XGBoost 1.7.6** for the classifier
- **Streamlit** for the dashboard
- **Plotly** and **pandas** for charts and data handling
- **scikit-learn** for data splitting and metrics
- **AWS SageMaker** for training the original model

## Project structure

```
AI4I-Monitoring-/
├── cnc_maintenance/
│   ├── app.py              # Streamlit dashboard
│   ├── features.py         # Feature engineering and input checks
│   ├── train.py            # Training and evaluation
│   └── extract_model.py    # Unpacks models/model.tar.gz (SageMaker output)
├── models/
│   ├── model.json          # Retrained, sensor-only model
│   ├── metrics.json        # Evaluation results, threshold and versions
│   ├── model.tar.gz        # Original model archive from SageMaker
│   └── xgboost-model       # Original model (still used by the app)
├── data/raw/               # Put ai4i2020.csv here (not committed)
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

To retrain and re-evaluate, download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), save it as `data/raw/ai4i2020.csv`, then run:

```bash
python cnc_maintenance/train.py
```

This rewrites `models/model.json` and `models/metrics.json` and prints the results. It takes a few seconds on a laptop.

The app loads `models/xgboost-model` directly. You only need `extract_model.py` if you download a new `model.tar.gz` from SageMaker:

```bash
python cnc_maintenance/extract_model.py
```

> **Note:** the original model uses XGBoost's legacy binary format, which XGBoost 3.1+ can no longer read. Keep `xgboost==1.7.6` until the app is switched to `model.json`.

## Roadmap

- [x] Add the training and evaluation code, plus a reproducible train/test split
- [x] Report precision, recall, F1 and PR-AUC, and measure inference latency
- [x] Retrain on sensor readings only (remove the failure-mode label leakage)
- [x] Save the model in JSON format
- [ ] Switch the dashboard to the new model and accept raw AI4I uploads
- [ ] Improve tool wear failure detection
- [ ] Add a sample input file and dashboard screenshots
- [ ] Tune the risk threshold and show feature importance in the dashboard
- [ ] Add unit tests and a CI workflow

## Team

University group project. Primary contributor: [Ahmed Mohamed](https://github.com/AhmedAmasib). Supervised by Prof. Mehmet Akyol.
