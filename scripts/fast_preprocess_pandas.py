import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import time

def fast_preprocess_and_select(raw_csv_path, output_dir_processed, output_dir_models):
    print("="*50)
    print("[INFO] BẮT ĐẦU TIỀN XỬ LÝ NHANH BẰNG PANDAS (BYPASS PYSPARK)")
    print("="*50)
    
    start_time = time.time()
    
    print(f"[*] Đang tải dữ liệu từ {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    print("[*] Xử lý missing values & duplicates...")
    df = df.drop_duplicates()
    if 'type' in df.columns:
        df = df.drop(columns=['type'])
        
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [c for c in num_cols if c != 'label']
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [c for c in cat_cols if c != 'label']
    
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c].fillna(df[c].mean(), inplace=True)
            
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c].fillna(df[c].mode()[0], inplace=True)
            
    print("[*] Encoding & Scaling...")
    le = LabelEncoder()
    for c in cat_cols:
        df[c] = le.fit_transform(df[c].astype(str))
        
    scaler = MinMaxScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
    print("[*] Tính toán Mutual Information (Top 30)...")
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # Handle any remaining nan/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sample down if dataset is extremely large to speed up MI
    if len(X) > 100000:
        print("    -> Dataset lớn, lấy mẫu 100k dòng để tính MI nhanh...")
        idx = np.random.choice(len(X), 100000, replace=False)
        X_sample, y_sample = X[idx], y[idx]
        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
    features = [c for c in df.columns if c != 'label']
    feature_mi = list(zip(features, mi_scores))
    feature_mi.sort(key=lambda x: x[1], reverse=True)
    
    top_k = 30
    top_features = [f[0] for f in feature_mi[:top_k]]
    
    # Save selected features json
    os.makedirs(output_dir_models, exist_ok=True)
    json_path = os.path.join(output_dir_models, "selected_features.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_features, f, indent=4)
    print(f"[OK] Đã lưu {json_path}")
    
    # Select only top features + label
    df_selected = df[top_features + ['label']]
    
    print("[*] Split Train/Test và lưu file...")
    train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42, stratify=df_selected['label'])
    
    os.makedirs(output_dir_processed, exist_ok=True)
    train_path = os.path.join(output_dir_processed, "train_processed.csv")
    train_df.to_csv(train_path, index=False)
    
    print(f"[OK] Đã lưu {train_path}")
    print(f"[INFO] Hoàn tất sau {time.time() - start_time:.2f} giây!")

if __name__ == "__main__":
    raw_path = r"c:\Users\Nguyen Trong Khoi\Downloads\DLL\ton_iot.csv"
    out_proc = r"c:\Users\Nguyen Trong Khoi\Downloads\DLL\Main-P\CyberDetect-Phase2\data\processed"
    out_mod = r"c:\Users\Nguyen Trong Khoi\Downloads\DLL\Main-P\CyberDetect-Phase2\saved_models"
    fast_preprocess_and_select(raw_path, out_proc, out_mod)
