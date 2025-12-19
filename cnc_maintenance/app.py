import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import requests
import time
from streamlit_lottie import st_lottie

st.set_page_config(page_title="AI4I | CNC Monitoring", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    .main { background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_robot = load_lottieurl("https://lottie.host/7970d4c8-3796-419b-a010-09048a604297/4wO6m6Z7mG.json")
lottie_scan = load_lottieurl("https://lottie.host/809f6e3c-88e8-4682-84f9-25f0e137817d/X6y8yYm8yB.json")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "xgboost-model")

def load_model():
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

with st.sidebar:
    if lottie_robot:
        st_lottie(lottie_robot, height=120, key="side_logo")
    st.markdown("### 👥 Students")
    st.info("**Ahmed Mohamed**\n231023208")
    st.info("**Arda Saygin**\n231023224")
    st.info("**Ahmed Salih**\n221023224")

st.title("Industrial CNC Predictive Maintenance")
st.write("Real-time AI diagnostics for factory floor equipment.")

uploaded_file = st.file_uploader("📂 Upload Data", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    if st.button("🚀 EXECUTE SYSTEM SCAN"):
        # 1. Show scanning animation
        scan_placeholder = st.empty()
        with scan_placeholder.container():
            st.write("### 🔍 Scanning machine telemetry...")
            if lottie_scan:
                st_lottie(lottie_scan, height=300, key="scanning")
            time.sleep(2) # Artificial delay to let user see the animation
        
        scan_placeholder.empty() # Remove animation when done
        
        # 2. Run the actual analysis
        try:
            model = load_model()
            num_data = data.select_dtypes(include=['number'])
            num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
            
            preds = model.predict(xgb.DMatrix(num_data))
            data["Risk Score"] = preds
            data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]
            
            # 3. Show Results
            st.balloons() if all(p <= 0.5 for p in preds) else st.warning("Failures detected in sequence!")
            st.write("### 📊 Diagnostic Intelligence")
            st.dataframe(data.style.background_gradient(subset=["Risk Score"], cmap="Reds"))
        except Exception as e:
            st.error(f"Analysis failed: {e}")
