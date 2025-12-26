import pandas as pd
from pathlib import Path

# Configuration
PROCESSED_FILE = Path("data/processed/cicids2017_processed.csv")
SIGNATURE_REPORT = Path("logs/signature_alerts.csv")

# Create logs directory
SIGNATURE_REPORT.parent.mkdir(exist_ok=True)

def apply_signature_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Initialize alert columns
    df['Signature_Alert'] = 'None'
    df['Alert_Type'] = ''
    
    print("Applying signature rules...")
    
    # Rule 1: DoS / DDoS - High packet rate or long flows
    dos_condition = (
        (df['Total Fwd Packets'] > 1000) |
        (df['Flow Duration'] > 10000000) |  # >10 seconds
        (df['Flow Packets/s'] > 500)
    )
    df.loc[dos_condition, 'Signature_Alert'] = 'Alert'
    df.loc[dos_condition, 'Alert_Type'] = 'Potential DoS/DDoS'
    
    # Rule 2: Port Scan / Probe - Many flows to common ports with small packets
    probe_condition = (
        (df['Destination Port'].isin([80, 443, 22, 21, 3389, 445])) &
        (df['Total Length of Fwd Packets'] < 200) &
        (df['Fwd Packet Length Mean'] < 60)
    )
    df.loc[probe_condition, 'Signature_Alert'] = 'Alert'
    df.loc[probe_condition, 'Alert_Type'] = 'Potential Port Scan'
    
    # Rule 3: Web Attack - Bursts to port 80/443
    web_attack_condition = (
        (df['Destination Port'].isin([80, 443])) &
        (df['Total Fwd Packets'] > 40) &
        (df['Flow Duration'] < 5000000)  # Short aggressive flows
    )
    df.loc[web_attack_condition, 'Signature_Alert'] = 'Alert'
    df.loc[web_attack_condition, 'Alert_Type'] = 'Potential Web Attack'
    
    # Rule 4: Infiltration / Suspicious - High init window or low backward traffic
    # Replaced Source Port with safer features
    infiltration_condition = (
        (df['Init_Win_bytes_forward'] > 60000) &  # Large window (common in exploits)
        (df['Total Backward Packets'] < 5) &
        (df['Flow Duration'] > 1000000)
    )
    df.loc[infiltration_condition, 'Signature_Alert'] = 'Alert'
    df.loc[infiltration_condition, 'Alert_Type'] = 'Potential Infiltration/Exploitation'
    
    # Summary
    alert_count = (df['Signature_Alert'] == 'Alert').sum()
    print(f"Total signature alerts generated: {alert_count}")
    if alert_count > 0:
        print("\nAlert types breakdown:")
        print(df[df['Signature_Alert'] == 'Alert']['Alert_Type'].value_counts())
    
    return df

def main():
    print(f"Loading processed dataset from {PROCESSED_FILE}...")
    if not PROCESSED_FILE.exists():
        print(f"Error: {PROCESSED_FILE} not found. Run preprocessing first.")
        return
    
    df = pd.read_csv(PROCESSED_FILE, low_memory=False)
    print(f"Loaded shape: {df.shape}")
    
    df_with_alerts = apply_signature_rules(df)
    
    # Save only alerts
    alerts_df = df_with_alerts[df_with_alerts['Signature_Alert'] == 'Alert']
    if not alerts_df.empty:
        alerts_df.to_csv(SIGNATURE_REPORT, index=False)
        print(f"\nSaved {len(alerts_df)} alerts to {SIGNATURE_REPORT}")
    else:
        print("\nNo alerts triggered.")
    
    # Save full enriched dataset
    enriched_file = Path("data/processed/cicids2017_with_signatures.csv")
    df_with_alerts.to_csv(enriched_file, index=False)
    print(f"Saved full dataset with signature labels to {enriched_file}")

if __name__ == "__main__":
    main()