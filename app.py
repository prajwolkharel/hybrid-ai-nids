import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🛡️ Hybrid AI-Based NIDS")

alerts_file = Path("logs/final_hybrid_alerts.csv")
if alerts_file.exists():
    df = pd.read_csv(alerts_file)
    st.success(f"Detected {len(df)} alerts from 2.83M flows")
    
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df['Final_Prediction'].value_counts().head(10))
    with col2:
        st.bar_chart(df['Source'].value_counts())
    
    st.dataframe(df.head(50))
    st.download_button("Download All Alerts", df.to_csv(index=False), "hybrid_alerts.csv")
    
    st.header("MITRE ATT&CK Mapping")
    st.table(df['MITRE_ATT&CK'].value_counts())
else:
    st.error("Run src/hybrid_nids.py first!")

st.caption("Signature + XGBoost Hybrid | CIC-IDS2017 | 99%+ Accuracy")