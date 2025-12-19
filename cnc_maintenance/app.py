import streamlit as st
import pandas as pd
import xgboost as xgb
import os

st.set_page_config(page_title="CNC Health Monitor", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "xgboost-model")

def load_model():
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

st.sidebar.markdown("### By The Students")
st.sidebar.write("**Ahmed Mohamed** - 231023208")
st.sidebar.write("**Arda Saygin** - 231023224")
st.sidebar.write("**Ahmed Salih** - 221023224")

st.sidebar.markdown("---")
st.sidebar.write("Project: AI4I Monitoring System")
st.sidebar.write("Model: XGBoost Classifier")

st.title("Industrial CNC Predictive Maintenance")
st.write("Upload machine sensor data to predict failure risks in real-time.")

uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_parquet(uploaded_file)
    
    st.write("### Raw Sensor Data")
    st.dataframe(data.head())

    if st.button("Analyze Machine Health"):
        model = load_model()
        
        data.columns = [str(c).replace("[", "").replace("]", "").replace("<", "") for c in data.columns]
        
        predict_data = data.select_dtypes(include=['number'])
        
        dmatrix = xgb.DMatrix(predict_data)
        predictions = model.predict(dmatrix)
        
        data["Failure_Risk"] = predictions
        data["Status"] = ["⚠️ CRITICAL" if p > 0.5 else "✅ HEALTHY" for p in predictions]
        
        st.write("### Prediction Results")
        st.dataframe(data.head(10))

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analysis Report",
            data=csv,
            file_name="cnc_analysis_results.csv",
            mime="text/csv",
        )
