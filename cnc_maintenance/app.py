import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="AI4I | CNC Monitoring", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        color: white;
        border: none;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_url = "https://lottie.host/7970d4c8-3796-419b-a010-09048a604297/4wO6m6Z7mG.json"
lottie_json = load_lottieurl(lottie_url)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "xgboost-model")

def load_model():
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

with st.sidebar:
    if lottie_json:
        st_lottie(lottie_json, height=120, key="side_logo")
    else:
        st.title("⚙️")
        
    st.markdown("### 👥 Students")
    st.info("**Ahmed Mohamed**\n231023208")
    st.info("**Arda Saygin**\n231023224")
    st.info("**Ahmed Salih**\n221023224")
    
    st.markdown("---")
    st.write("📊 **Project:** AI4I Monitoring")
    st.write("🛠️ **Model:** XGBoost v1.7.6")

header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("Industrial CNC Predictive Maintenance")
    st.write("This dashboard utilizes Machine Learning to process sensor telemetry and predict potential equipment failures in real-time.")

with header_right:
    if lottie_json:
        st_lottie(lottie_json, height=180, key="main_anim")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload CNC Sensor Data (CSV or Parquet)", type=["csv", "parquet"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_parquet(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(data))
    col2.metric("Features Detected", len(data.columns))
    col3.metric("System Status", "Ready", "Online")

    if st.button("🚀 EXECUTE SYSTEM ANALYSIS"):
        with st.spinner("Analyzing patterns..."):
            try:
                model = load_model()
                
                predict_data = data.select_dtypes(include=['number'])
                predict_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in predict_data.columns]
                
                dmatrix = xgb.DMatrix(predict_data)
                predictions = model.predict(dmatrix)
                
                data["Risk Score"] = predictions
                data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in predictions]
                
                st.write("### 🔍 Failure Diagnostic Results")
                st.dataframe(
                    data.style.background_gradient(subset=["Risk Score"], cmap="Reds")
                              .format({"Risk Score": "{:.2%}"})
                )

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Final Report",
                    data=csv,
                    file_name="cnc_analysis_results.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during analysis: {e}")
