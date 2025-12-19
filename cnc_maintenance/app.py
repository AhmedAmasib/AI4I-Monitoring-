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
    .logic-box { background-color: #eef2ff; padding: 20px; border-left: 5px solid #1E3A8A; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "xgboost-model")

def load_model():
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

with st.sidebar:
    st.markdown("# CNC SYSTEM")
    st.info("**Ahmed Mohamed** (231023208)\n\n**Arda Saygin** (231023224)\n\n**Ahmed Salih** (221023224)")

st.title("Industrial CNC Predictive Maintenance")

# --- NEW SECTION: HOW THE AI WORKS ---
with st.expander(" How does this AI work? (Technical Explanation)"):
    st.markdown('<div class="logic-box">', unsafe_allow_html=True)
    st.write("""
    This platform uses an **XGBoost (eXtreme Gradient Boosting)** algorithm. 
    Unlike a simple "if/then" rule, this AI works by building hundreds of small **Decision Trees**.
    
    1. **Learning:** Each tree looks at a small piece of the data (like Torque vs. RPM).
    2. **Correcting:** If the first tree makes a mistake, the second tree focuses specifically on that mistake.
    3. **Summation:** When you upload data, the AI runs your sensors through all these trees and calculates a **Probability Score**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("###  The 3 Key Variables the AI Monitors:")
    col_a, col_b, col_c = st.columns(3)
    col_a.write("**1. Air Temperature:** Detects if the cooling system is failing (HDF).")
    col_b.write("**2. Torque:** High torque spikes often mean the tool is stuck or broken.")
    col_c.write("**3. Tool Wear:** Monitors the 'minutes of use' to predict blade failure (TWF).")






st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Sensor Data", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    if st.button("EXECUTE DIAGNOSTIC SCAN"):
        model = load_model()
        num_data = data.select_dtypes(include=['number'])
        num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
        
        preds = model.predict(xgb.DMatrix(num_data))
        data["Risk Score"] = preds
        data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]

        
        st.markdown("### 🕵️‍♂️ A fun way to discover the failure!")
        temp_col = next((c for c in data.columns if 'temperature' in c.lower()), data.columns[0])
        fig = px.scatter(data, x=temp_col, y="Risk Score", color="Status", 
                         color_discrete_map={"🔴 CRITICAL": "red", "🟢 HEALTHY": "green"},
                         hover_data=data.columns, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        
        st.markdown("###  Professional Analysis Summary")
        critical_cases = (data["Status"] == "🔴 CRITICAL").sum()
        st.info(f"The XGBoost model identified {critical_cases} critical risks. Hover over the chart above to see the sensor values for each specific incident.")

        
        st.write("### 📊 Detailed Telemetry Report")
        st.dataframe(data.style.background_gradient(subset=["Risk Score"], cmap="Reds").format({"Risk Score": "{:.2%}"}))

