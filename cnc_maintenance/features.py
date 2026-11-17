"""Feature engineering shared by training and the dashboard.

Only sensor readings and the product type are used as inputs. The AI4I
failure-mode columns (TWF, HDF, PWF, OSF, RNF) are labels, not sensor data:
`Machine failure` is defined from them, so feeding them to the model leaks
the answer.
"""
import numpy as np
import pandas as pd

# Raw AI4I column -> safe name (XGBoost rejects "[", "]" and "<" in names)
RAW_COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

# Never used as model inputs
LABEL_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

TYPE_CODES = {"L": 0, "M": 1, "H": 2}  # low / medium / high quality variant

FEATURES = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",      # process - air temperature (heat dissipation)
    "power_w",          # torque x angular speed (power)
    "wear_x_torque",    # tool wear x torque (overstrain)
]


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Return the required raw columns that are absent from df."""
    return [c for c in RAW_COLUMNS if c not in df.columns]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw AI4I-format rows into the model's feature matrix."""
    missing = missing_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    x["type"] = x["type"].astype(str).str.strip().str.upper().map(TYPE_CODES)
    if x["type"].isna().any():
        raise ValueError("Column 'Type' must contain only L, M or H")

    x["temp_diff_k"] = x["process_temp_k"] - x["air_temp_k"]
    x["power_w"] = x["torque_nm"] * x["speed_rpm"] * 2 * np.pi / 60
    x["wear_x_torque"] = x["tool_wear_min"] * x["torque_nm"]
    return x[FEATURES].astype(float)
