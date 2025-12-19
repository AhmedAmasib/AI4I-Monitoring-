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
    [data-testid="stMetricLabel"] { color: #333333 !important; }
    [data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 10px; }
    .stButton>button { background-color: #1E3A8A; color: white; border-radius: 10px; font-weight: bold; width: 100%; height: 3.5em; }
    .fun-header { color: #2563EB; font-family: 'Trebuchet MS', sans-serif; font-weight: bold; font-size: 24px; }
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
    st.write("🛠️ **Model:** XGBoost v1.7.6")

st.title("Industrial CNC Predictive Maintenance")
st.write("Upload your factory data below to begin the diagnostic process.")

uploaded_file = st.file_uploader("📂 Upload Sensor Data (CSV/Parquet)", type=["csv", "parquet"])

if uploaded_file:
    data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_parquet(uploaded_file)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Telemetry Rows", len(data))
    m2.metric("Machine Status", "Connected")
    m3.metric("Analysis", "Ready")

    if st.button("🚀 EXECUTE SYSTEM SCAN"):
        model = load_model()
        num_data = data.select_dtypes(include=['number'])
        num_data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in num_data.columns]
        
        preds = model.predict(xgb.DMatrix(num_data))
        data["Risk Score"] = preds
        data["Status"] = ["🔴 CRITICAL" if p > 0.5 else "🟢 HEALTHY" for p in preds]

        # 1. FUN INTERACTIVE SECTION AT THE TOP
        st.markdown('<p class="fun-header">🕵️‍♂️ A fun way to discover the failure!</p>', unsafe_allow_html=True)
        st.write("Hover your mouse over the points below. The red dots represent machines in danger—see if you can find the sensor reading that caused the risk!")
        
        temp_col = next((c for c in data.columns if 'temperature' in c.lower()), data.columns[0])
        
        fig = px.scatter(
            data, 
            x=temp_col, 
            y="Risk Score",
            color="Status",
            color_discrete_map={"🔴 CRITICAL": "red", "🟢 HEALTHY": "green"},
            hover_data=data.columns,
            template="plotly_white"
        )
        fig.update_traces(marker=dict(size=12, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

        

        # 2. THE STATIC PROFESSIONAL PARAGRAPH
        st.markdown("### 📝 Professional Analysis Summary")
        critical_cases = (data["Status"] == "🔴 CRITICAL").sum()
        
        analysis_text = f"""
        Based on the XGBoost model analysis of **{len(data)}** telemetry points, the system has identified 
        **{critical_cases}** instances of high-risk operational behavior. The chart above illustrates the 
        correlation between **{temp_col}** and the probability of failure. Points marked in **RED** exceed the 50% risk threshold and require immediate mechanical inspection to prevent downtime.
        """
        st.info(analysis_text)

        # 3. DATA TABLE AT THE BOTTOM
        st.write("### 📊 Detailed Telemetry Report")
        st.dataframe(
            data.style.background_gradient(subset=["Risk Score"], cmap="Reds")
                      .format({"Risk Score": "{:.2%}"}),
            use_container_width=True
        )

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results", data=csv, file_name="cnc_report.csv")
