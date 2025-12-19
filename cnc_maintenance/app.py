import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI4I | CNC Monitoring", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #1E3A8A;
        color: white;
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
    st.markdown("---")
    st.markdown("### 👥 Students")
    st.info("**Ahmed Mohamed**\n231023208")
    st.info("**Arda Saygin**\n231023224")
    st.info("**Ahmed Salih**\n221023224")
    st.markdown("---")
    st.write("📊 **Project:** AI4I Monitoring")
    st.write("🛠️ **Model:** XGBoost Classifier")

st.title("Industrial CNC Predictive Maintenance")
st.write("Analyze machine telemetry to prevent thermal and mechanical failures.")

uploaded_file = st.file_uploader("📂 Upload Sensor Data (CSV/Parquet)", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Telemetry Rows", len(data))
    col2.metric("Machine Status", "Connected")
    col3.metric("Health Check", "Pending")

    if st.button("🚀 EXECUTE DIAGNOSTIC SCAN"):
        with st.status("Running ML Analysis...", expanded=True) as status:
            model = load_model()
            num_data = data.select_dtypes(include=['number'])
            num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
            
            dmatrix = xgb.DMatrix(num_data)
            preds = model.predict(dmatrix)
            
            data["Risk Score"] = preds
            data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        st.write("### 📈 Risk Correlation Analysis")
        
        # Check for Air temperature or Process temperature columns
        temp_col = next((c for c in data.columns if 'temperature' in c.lower()), None)
        
        if temp_col:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(data[temp_col], data["Risk Score"], alpha=0.5, c=data["Risk Score"], cmap='Reds')
            ax.set_xlabel(f"Temperature ({temp_col})")
            ax.set_ylabel("Failure Risk Score")
            ax.set_title("Correlation: Temperature vs. Failure Risk")
            st.pyplot(fig)
        
        

        st.write("### 📊 Diagnostic Intelligence Report")
        st.dataframe(
            data.style.background_gradient(subset=["Risk Score"], cmap="Reds")
                      .format({"Risk Score": "{:.2%}"}),
            use_container_width=True
        )

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results", data=csv, file_name="cnc_report.csv")
