import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import plotly.express as px

st.set_page_config(page_title="AI4I | CNC Monitoring", layout="wide", page_icon="⚙️")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { color: #000000 !important; font-weight: bold !important; }
    [data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 10px; font-weight: bold; width: 100%; height: 3.5em; }
    .logic-box { background-color: #f0f7ff; padding: 20px; border-left: 5px solid #1E3A8A; border-radius: 8px; color: #1e1e1e; }
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
    
    # PROFESSOR SECTION
    st.markdown("### 👨‍🏫 Supervised By")
    st.success("**Prof. Mehmet Akyol**")
    
    st.markdown("---")
    
    # STUDENT SECTION
    st.markdown("### 👥 Students")
    st.info("**Ahmed Mohamed** (231023208)\n\n**Arda Saygin** (231023224)\n\n**Ahmed Salih** (221023224)")
    
    st.markdown("---")
    st.write("🛠️ **Model:** XGBoost v1.7.6")

st.title("Industrial CNC Predictive Maintenance")

with st.expander("🧠 How the AI Works: Decision Tree Logic"):
    st.markdown('<div class="logic-box">', unsafe_allow_html=True)
    st.write("""
    The system uses **Gradient Boosted Decision Trees**. 
    Think of it as a flowchart where the AI asks specific questions about your machine:
    
    1. **Splitting:** It asks "Is the Torque higher than 50Nm?" If yes, it follows one path; if no, another.
    2. **Ensemble:** It doesn't rely on one tree. It uses hundreds of trees that "vote" on whether a failure is likely.
    3. **Optimization:** Every new tree learns from the mistakes of the previous one.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("### 🔑 Predictive Indicators:")
    c1, c2, c3 = st.columns(3)
    c1.write("**Thermal Stress:** High Air/Process temperature delta.")
    c2.write("**Mechanical Strain:** Unexpected torque spikes.")
    c3.write("**Usage Lifecycle:** Total tool wear minutes.")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Sensor Data", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    if st.button("🚀 EXECUTE DIAGNOSTIC SCAN"):
        model = load_model()
        num_data = data.select_dtypes(include=['number'])
        num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
        
        preds = model.predict(xgb.DMatrix(num_data))
        data["Risk Score"] = preds
        data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]

        st.markdown("### 🕵️‍♂️ A fun way to discover the failure!")
        st.write("Point your mouse at the dots to see the hidden telemetry data.")
        
        temp_col = next((c for c in data.columns if 'temperature' in c.lower()), data.columns[0])
        fig = px.scatter(data, x=temp_col, y="Risk Score", color="Status", 
                         color_discrete_map={"🔴 CRITICAL": "red", "🟢 HEALTHY": "green"},
                         hover_data=data.columns, template="plotly_white")
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Analysis Summary")
        critical_cases = (data["Status"] == "🔴 CRITICAL").sum()
        st.info(f"The model detected {critical_cases} critical machine states. Review the table below for details.")

        st.write("### 📊 Detailed Telemetry Report")
        st.dataframe(data.style.background_gradient(subset=["Risk Score"], cmap="Reds").format({"Risk Score": "{:.2%}"}))

        st.markdown("---")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Full Diagnostic Report",
            data=csv,
            file_name="cnc_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )
