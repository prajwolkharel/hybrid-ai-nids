from nfstream import NFStreamer
import pandas as pd
import xgboost as xgb
from pathlib import Path
import numpy as np

# Paths
XG_MODEL = Path("models/xgboost.json")
LIVE_ALERTS = Path("logs/live_alerts.csv")
LIVE_ALERTS.parent.mkdir(exist_ok=True)

def load_xgb_model():
    print("Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(str(XG_MODEL))
    return model

def apply_live_signature(flow):
    alert = "None"
    alert_type = ""
    
    # DoS: High packets or bytes
    if flow.bidirectional_packets > 800 or flow.bidirectional_bytes > 500000:
        alert = "Alert"
        alert_type = "Potential DoS/DDoS"
    
    # Port Scan: Many dst packets, few src
    if flow.dst2src_packets > 30 and flow.src2dst_packets < 10:
        alert = "Alert"
        alert_type = "Potential Port Scan"
    
    # Web Attack: HTTP port with burst
    if flow.dst_port == 80 or flow.dst_port == 443:
        if flow.src2dst_packets > 40:
            alert = "Alert"
            alert_type = "Potential Web Attack"
    
    return alert, alert_type

def predict_xgb_single(model, flow_df):
    dmatrix = xgb.DMatrix(flow_df)
    pred_prob = model.predict(dmatrix)[0]
    pred_class = np.argmax(pred_prob)
    return pred_class, pred_prob[pred_class]

def main():
    print("Starting real-time hybrid NIDS...")
    xgb_model = load_xgb_model()
    
    live_alerts = []
    
    # Your WiFi interface
    interface = "wlp0s20f3"  # Change to "lo" for safe test
    #interface = "lo"

    print(f"Monitoring interface: {interface}")
    print("Generating traffic (browse, ping, download) to see alerts!")
    print("Press Ctrl+C to stop\n")
    
    streamer = NFStreamer(
        source=interface,
        statistical_analysis=True,
        performance_report=0,  # No report
        idle_timeout=120,
        active_timeout=1800
    )
    
    for flow in streamer:
        if flow.bidirectional_packets < 5:  # Skip tiny flows
            continue
        
        sig_alert, sig_type = apply_live_signature(flow)
        
        final_alert = sig_alert
        final_type = sig_type
        confidence = 1.0
        source = "Signature"
        
        if sig_alert == "None":
            # XGBoost fallback
            flow_dict = flow.to_dict()
            flow_df = pd.DataFrame([flow_dict])
            # Select numeric features (avoid strings)
            numeric_cols = flow_df.select_dtypes(include=[np.number]).columns
            X = flow_df[numeric_cols]
            
            try:
                pred_class, pred_conf = predict_xgb_single(xgb_model, X)
                if pred_class != 0:  # Not BENIGN (adjust if label map known)
                    final_alert = "Alert"
                    final_type = "ML Detected Anomaly"
                    confidence = pred_conf
                    source = "XGBoost"
            except:
                pass
        
        if final_alert == "Alert":
            print(f"🚨 [{source}] {final_type} | Confidence: {confidence:.2f}")
            print(f"   Src: {flow.src_ip}:{flow.src_port} → Dst: {flow.dst_ip}:{flow.dst_port}")
            print(f"   Packets: {flow.bidirectional_packets} | Bytes: {flow.bidirectional_bytes}\n")
            
            live_alerts.append({
                "Time": pd.Timestamp.now(),
                "Source_IP": flow.src_ip,
                "Dest_IP": flow.dst_ip,
                "Source_Port": flow.src_port,
                "Dest_Port": flow.dst_port,
                "Alert_Type": final_type,
                "Confidence": confidence,
                "Source": source
            })
    
    # Save on exit
    if live_alerts:
        pd.DataFrame(live_alerts).to_csv(LIVE_ALERTS, index=False)
        print(f"Saved {len(live_alerts)} live alerts to {LIVE_ALERTS}")

if __name__ == "__main__":
    main()