import os
import pandas as pd
import numpy as np
from datetime import datetime

# Standard 80 columns of CSE-CIC-IDS2018 from CICFlowMeter
CIC_IDS2018_COLUMNS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", 
    "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", 
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", 
    "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", 
    "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", 
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", 
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", 
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", 
    "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", 
    "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", 
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", 
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", 
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", 
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", 
    "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", 
    "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts", 
    "Init Bwd Win Byts", "Fwd Act Pkts", "Fwd Seg Size Min", "Active Mean", 
    "Active Std", "Active Max", "Active Min", "Idle Mean", 
    "Idle Std", "Idle Max", "Idle Min", "Label"
]

def generate_synthetic_ids2018(output_path, num_rows=1000):
    """
    Generate synthetic CSE-CIC-IDS2018 CSV data for testing.
    """
    print(f"[WARNING] Real CSE-CIC-IDS2018 file not found at: {output_path}")
    print(f"[*] Generating synthetic dataset ({num_rows} rows) for compatibility...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    data = {}
    
    for col in CIC_IDS2018_COLUMNS:
        if col == "Dst Port":
            data[col] = np.random.choice([80, 443, 22, 8080, 53], size=num_rows)
        elif col == "Protocol":
            data[col] = np.random.choice([6, 17, 0], size=num_rows)
        elif col == "Timestamp":
            data[col] = [datetime.now().strftime("%d/%m/%Y %H:%M:%S") for _ in range(num_rows)]
        elif col == "Label":
            data[col] = np.random.choice(["Benign", "DDoS attacks-LOIC-HTTP", "Bot", "Brute Force -Web"], 
                                         p=[0.75, 0.15, 0.08, 0.02], size=num_rows)
        elif "Flags" in col or "Cnt" in col or "Count" in col:
            data[col] = np.random.choice([0, 1], p=[0.9, 0.1], size=num_rows)
        else:
            data[col] = np.random.exponential(scale=100.0, size=num_rows).astype(np.float32)
            
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Synthetic dataset saved successfully at: {output_path}\n")

def load_and_preprocess_ids2018(data_path, sample_rows=None):
    """
    Load CSE-CIC-IDS2018 data and clean it.
    """
    if not os.path.exists(data_path):
        generate_synthetic_ids2018(data_path, num_rows=2000)
        
    print(f"[*] Reading CSE-CIC-IDS2018 data from {data_path}...")
    
    df = pd.read_csv(data_path, nrows=sample_rows)
    
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
        
    label_col = "Label"
    if label_col not in df.columns:
        for c in df.columns:
            if c.lower() in ["label", "target", "class"]:
                label_col = c
                break
                
    y = df[label_col].values
    X_df = df.drop(columns=[label_col])
    
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = X_df[col].astype('category').cat.codes
        
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0)
    
    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()
    
    print(f"[OK] Loaded {X.shape[0]} samples, {X.shape[1]} original features.")
    return X, y, feature_names
