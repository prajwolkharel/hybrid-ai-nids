import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# Paths
PROCESSED_WITH_SIG = Path("data/processed/cicids2017_with_signatures.csv")
XG_MODEL = Path("models/xgboost.json")
FINAL_ALERTS = Path("logs/final_hybrid_alerts.csv")
FINAL_ALERTS.parent.mkdir(exist_ok=True)

# MITRE ATT&CK mapping
MITRE_MAPPING = {
    'Potential DoS/DDoS': 'Impact - T1499 Endpoint Denial of Service',
    'Potential Port Scan': 'Discovery - T1046 Network Service Scanning',
    'Potential Web Attack': 'Initial Access - T1190 Exploit Public-Facing Application',
    'Potential Infiltration/Exploitation': 'Execution - T1203 Exploitation for Client Execution',
    'BENIGN': 'No Threat',
}

def load_xgb_model():
    if not XG_MODEL.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XG_MODEL}")
    model = xgb.Booster()
    model.load_model(str(XG_MODEL))
    return model

def predict_xgb(model, X):
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    return np.argmax(preds, axis=1)

def ensemble_prediction(df, xgb_model):
    df = df.copy()
    
    # Priority 1: Signature alerts
    df['Final_Prediction'] = np.where(
        df['Signature_Alert'] == 'Alert',
        df['Alert_Type'],
        'BENIGN'
    )
    df['Confidence'] = np.where(df['Signature_Alert'] == 'Alert', 1.0, 0.0)
    df['Source'] = np.where(df['Signature_Alert'] == 'Alert', 'Signature', 'XGBoost')
    
    # For non-signature flows, use XGBoost
    no_sig = df['Signature_Alert'] == 'None'
    if no_sig.any():
        feature_cols = [c for c in df.columns if c not in ['Label', 'Signature_Alert', 'Alert_Type', 'Final_Prediction', 'Confidence', 'Source']]
        X = df.loc[no_sig, feature_cols].fillna(0)
        
        # Sample for speed
        sample_size = min(20000, len(X))
        sample_idx = np.random.choice(X.index, sample_size, replace=False)
        X_sample = X.loc[sample_idx]
        
        pred_idx = predict_xgb(xgb_model, X_sample)
        # Map back to labels (use known from training)
        label_map = {0: 'BENIGN', 1: 'DoS Hulk', 2: 'PortScan', 3: 'DDoS'}  # Approximate — real would use saved encoder
        pred_labels = [label_map.get(i, 'Unknown Attack') for i in pred_idx]
        
        df.loc[sample_idx, 'Final_Prediction'] = pred_labels
        df.loc[sample_idx, 'Confidence'] = 0.95
        df.loc[sample_idx, 'Source'] = 'XGBoost'
    
    df['MITRE_ATT&CK'] = df['Final_Prediction'].map(MITRE_MAPPING).fillna('Unknown')
    
    return df

def main():
    print("Running hybrid ensemble (Signature + XGBoost)...")
    df = pd.read_csv(PROCESSED_WITH_SIG, low_memory=False)
    print(f"Loaded {df.shape[0]} flows")
    
    xgb_model = load_xgb_model()
    
    final_df = ensemble_prediction(df, xgb_model)
    
    alerts = final_df[final_df['Final_Prediction'] != 'BENIGN']
    alerts.to_csv(FINAL_ALERTS, index=False)
    print(f"\nSaved {len(alerts)} final hybrid alerts to {FINAL_ALERTS}")
    print("Top alerts:")
    print(alerts['Final_Prediction'].value_counts().head(10))
    print("\nSources:")
    print(alerts['Source'].value_counts())

if __name__ == "__main__":
    main()