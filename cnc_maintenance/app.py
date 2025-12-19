import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="AI4I | CNC Predictive Systems", layout="wide", page_icon="⚙️")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_industrial = load_lottieurl("https://lottie.host/7970d4c8-3796-419b-a010-09048a604297/4wO6m6Z7mG.json")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        background-color: #2563EB;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "xgboost-model")

def load_model():
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

with st.sidebar:
    st_lottie(lottie_industrial, height=150, key="sidebar_robot")
    st.markdown("## 👥 Students")
    st.info("**Ahmed Mohamed**\n231023208")
    st.info("**Arda Saygin**\n231023224")
    st.info("**Ahmed Salih**\n221023224")
    st.markdown("---")
    st.write("📍 **Project:** AI4I Monitoring")
    st.write("🚀 **Status:** Live Deployment")

head_col1, head_col2 = st.columns([2, 1])

with head_col1:
    st.title("Industrial CNC Predictive Maintenance")
    st.write("Welcome to the AI4I monitoring interface. This system uses real-time telemetry to predict machine downtime and technical failures before they occur.")

with head_col2:
    st_lottie(lottie_industrial, height=200, key="main_robot")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Machine Sensor Logs (CSV or Parquet)", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    st.success("File uploaded successfully! Ready for diagnostic.")
    
    if st.button("🚀 EXECUTE SYSTEM ANALYSIS"):
        with st.spinner("Analyzing machine patterns..."):
            model = load_model()
            
            clean_data = data.select_dtypes(include=['number'])
            clean_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in clean_data.columns]
            
            dmatrix = xgb.DMatrix(clean_data)
            preds = model.predict(dmatrix)
            
            data["Risk Score"] = preds
            data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]
            
            st.write("### 📊 Diagnostic Intelligence")
            st.dataframe(data.style.background_gradient(subset=["Risk Score"], cmap="Reds"))

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Detailed Maintenance Report", data=csv, file_name="cnc_report.csv")
