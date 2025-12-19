import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI4I | CNC Monitoring", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 32px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #333333 !important;
        font-weight: bold !important;
    }

    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 10px;
        font-weight: bold;
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
    st.markdown("# 🏭 SYSTEM")
    st.info("**Ahmed Mohamed** (231023208)\n\n**Arda Saygin** (231023224)\n\n**Ahmed Salih** (221023224)")
    st.markdown("---")
    st.write("📊 **Project:** AI4I Monitoring")

st.title("Industrial CNC Predictive Maintenance")
st.write("Real-time AI diagnostics for high-precision manufacturing.")

with st.expander("ℹ️ Click here to see how this AI works"):
    st.write("""
        This system uses an **XGBoost (Extreme Gradient Boosting)** model to analyze sensor data. 
        By monitoring variables like RPM, Torque, and Temperature, the AI can detect 
        patterns that lead to **Heat Dissipation Failures** or **Tool Wear**.
    """)

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Sensor Data", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Telemetry Rows", len(data))
    m2.metric("Machine Status", "Connected")
    m3.metric("Health Check", "Ready")

    if st.button("🚀 EXECUTE DIAGNOSTIC SCAN"):
        model = load_model()
        num_data = data.select_dtypes(include=['number'])
        num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
        
        preds = model.predict(xgb.DMatrix(num_data))
        data["Risk Score"] = preds
        data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]
        
        st.success("Analysis Complete!")

        # Dynamic Paragraph Interaction
        st.markdown("### 📝 Analysis Summary")
        critical_count = (data["Status"] == "🔴 CRITICAL").sum()
        
        if critical_count > 0:
            st.error(f"Attention: The AI has detected **{critical_count}** critical failure points in this dataset. Please review the highlighted rows below.")
        else:
            st.balloons()
            st.success("The system is operating within normal parameters. No failures detected.")

        st.dataframe(
            data.style.background_gradient(subset=["Risk Score"], cmap="Reds")
                      .format({"Risk Score": "{:.2%}"}),
            use_container_width=True
        )

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results", data=csv, file_name="cnc_report.csv")
