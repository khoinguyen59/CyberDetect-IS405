"""
Retrain MLP trên các dataset mới (UNSW-NB15, BoT-IoT) với SMOTE-ENN.
Thực hiện full pipeline: Load raw → Encode → Select features → SMOTE-ENN → Train → Evaluate.
"""
import os
import sys
import pandas as pd
import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Sua loi ModuleNotFoundError: them src/ vao path de import model + modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model.cyberdetect_mlp import build_model
from modules.resampling.smote_enn import apply_smote_enn


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Cấu hình dataset
# ============================================================
DATASET_CONFIG = {
    "unsw": {
        "name": "UNSW-NB15",
        "path": "data/raw/UNSW-NB15/UNSW_NB15_training-set.csv",
        "label_candidates": ["label", "Label", "attack_cat"],
        "drop_cols": ["id", "attack_cat"],  # metadata không phải feature
    },
    "botiot": {
        "name": "BoT-IoT",
        "path": "data/raw/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_Best.csv",
        "label_candidates": ["attack", "label", "Label"],
        "drop_cols": ["pkSeqID", "category", "subcategory", "saddr", "daddr"],
    }
}

TOP_K_FEATURES = 30  # Theo paper: top-30 MI features


def detect_label(df, candidates):
    """Tìm cột label từ danh sách ưu tiên."""
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: cột cuối cùng
    return df.columns[-1]


def preprocess_dataset(df, label_col, drop_cols, top_k=30):
    """
    Preprocessing pipeline cho raw dataset:
    1. Drop metadata
    2. Tách label
    3. Encode categoricals (LabelEncoder)
    4. MI feature selection (top-k)
    5. Return X, y (numpy arrays)
    """
    # Drop metadata
    cols_to_drop = [c for c in drop_cols if c in df.columns and c != label_col]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Tách label
    y = df[label_col].values
    X_df = df.drop(columns=[label_col])
    
    # Encode categorical columns
    le_dict = {}
    for col in X_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        le_dict[col] = le
    
    # Convert to float
    X_df = X_df.astype(float)
    
    # Handle NaN/Inf
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0)
    
    # Mutual Information feature selection
    n_features = min(top_k, X_df.shape[1])
    print(f"[*] Chọn top-{n_features} features bằng Mutual Information...")
    
    # Sample nếu dataset quá lớn (MI tính lâu trên >100K rows)
    if len(X_df) > 100000:
        sample_idx = np.random.RandomState(42).choice(len(X_df), 100000, replace=False)
        mi_scores = mutual_info_classif(X_df.iloc[sample_idx], y[sample_idx], random_state=42)
    else:
        mi_scores = mutual_info_classif(X_df, y, random_state=42)
    
    # Chọn top-k
    top_features_idx = np.argsort(mi_scores)[::-1][:n_features]
    selected_features = X_df.columns[top_features_idx].tolist()
    print(f"[*] Top-5 features: {selected_features[:5]}")
    
    X = X_df[selected_features].values
    
    # Encode label thành 0/1 nếu chưa phải numeric
    if y.dtype == object:
        le_label = LabelEncoder()
        y = le_label.fit_transform(y)
    
    # Đảm bảo binary
    unique_labels = np.unique(y)
    if len(unique_labels) > 2:
        print(f"[WARNING] {len(unique_labels)} classes detected, converting to binary (0=Normal, 1=Attack)")
        y = (y > 0).astype(int)
    
    return X, y, selected_features


def retrain_on_dataset(dataset_key):
    """Retrain MLP trên 1 dataset với SMOTE-ENN."""
    config = DATASET_CONFIG[dataset_key]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Du lieu nam o GOC project (../data/raw/...), khong phai trong scripts/
    data_path = os.path.join(_project_root(), config["path"])
    
    print("\n" + "="*60)
    print(f"  RETRAIN TRÊN {config['name']} VỚI SMOTE-ENN")
    print("="*60)
    
    # 1. Load
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy: {data_path}")
        print("[SKIP] Bỏ qua dataset này.")
        return None
    
    print(f"[*] Đang tải {config['name']} từ {data_path}...")
    # Tu dong nhan dau phan cach (',' hoac ';' nhu BoT-IoT 10-best)
    df = pd.read_csv(data_path, sep=None, engine="python")
    # Strip BOM + khoang trang o ten cot (UNSW-NB15 co BOM tren cot 'id')
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    # Bo cot index/khong ten (BoT-IoT co cot dau rong gay ro ri qua thu tu dong)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed|^$")]
    print(f"[*] Shape: {df.shape} | cols: {list(df.columns)[:6]}...")
    
    # 2. Detect label
    label_col = detect_label(df, config["label_candidates"])
    print(f"[*] Cột label: '{label_col}'")

    # 2b. Cap so dong cho dataset RAT LON (BoT-IoT ~3.6M) -> tranh OOM/timeout SMOTE-ENN tren Colab.
    #     Lay mau PHAN TANG de giu nguyen ty le lop (thuc hanh pho bien voi BoT-IoT 5% sample).
    MAX_ROWS = 200000
    if len(df) > MAX_ROWS:
        try:
            df, _ = train_test_split(df, train_size=MAX_ROWS, stratify=df[label_col], random_state=42)
        except Exception:
            df = df.sample(n=MAX_ROWS, random_state=42)
        df = df.reset_index(drop=True)
        print(f"[*] Dataset lon -> lay mau phan tang {MAX_ROWS} dong (tranh OOM SMOTE-ENN tren Colab).")

    # 3. Preprocess
    X, y, selected_features = preprocess_dataset(
        df, label_col, config["drop_cols"], top_k=TOP_K_FEATURES
    )
    print(f"[*] Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"[*] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 4. Split — handle extreme imbalance where stratify might fail
    unique_counts = dict(zip(*np.unique(y, return_counts=True)))
    min_class_count = min(unique_counts.values())
    
    if min_class_count < 5:
        print(f"[WARNING] Minority class chỉ có {min_class_count} samples — dùng random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # 5. SMOTE-ENN — safety check cho extreme imbalance (BoT-IoT: 107 Normal vs 733K Attack)
    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    min_train = min(train_counts.values())
    max_train = max(train_counts.values())
    ratio = min_train / max_train if max_train > 0 else 0
    
    if ratio < 0.01:
        # Extreme imbalance → SMOTE 1:1 sẽ OOM. Dùng sampling_strategy=0.3
        print(f"[WARNING] Extreme imbalance (ratio={ratio:.6f}). Dùng sampling_strategy=0.3")
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        k = min(5, min_train - 1) if min_train > 1 else 1
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.3, k_neighbors=k, random_state=42),
            random_state=42
        )
        X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
        new_counts = dict(zip(*np.unique(y_train_res, return_counts=True)))
        print(f"[OK] Sau SMOTE-ENN (0.3): {new_counts}")
    else:
        X_train_res, y_train_res = apply_smote_enn(X_train, y_train)
    
    # 6. Normalize
    print("[*] Normalizing...")
    scaler = MinMaxScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
    
    # 7. Build & Train
    input_dim = X_train_res.shape[1]
    num_classes = 2
    print(f"[*] Kiến trúc MLP (Input: {input_dim}, Classes: {num_classes})")
    
    model = build_model(input_dim=input_dim, num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    save_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{dataset_key}.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    ]
    
    print("[*] Bắt đầu training...")
    start = time.time()
    history = model.fit(
        X_train_res, y_train_res,
        epochs=50, batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start
    print(f"[OK] Training hoàn tất trong {elapsed:.2f} giây.")
    
    # 8. Evaluate
    print(f"\n[*] Đánh giá trên tập TEST ({config['name']}):")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    
    # 9. Save
    model_path = os.path.join(save_dir, f"cyberdetect_mlp_{dataset_key}_smote_enn.h5")
    model.save(model_path)
    print(f"[INFO] Đã lưu: {model_path}")
    
    # Save selected features
    feat_path = os.path.join(save_dir, f"selected_features_{dataset_key}.json")
    with open(feat_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"[INFO] Đã lưu features: {feat_path}")
    
    return {
        "dataset": config["name"],
        "accuracy": acc,
        "training_time": elapsed,
        "features_used": len(selected_features),
    }


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    results = []
    
    if dataset_arg in ("all", "unsw"):
        r = retrain_on_dataset("unsw")
        if r: results.append(r)
    
    if dataset_arg in ("all", "botiot"):
        r = retrain_on_dataset("botiot")
        if r: results.append(r)
    
    # Ghi ket qua THAT ra CSV de notebook 08 doc (thay cho viec in so cung)
    results_dir = os.path.join(_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "cross_dataset_results.csv")
    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("\n" + "="*60)
        print("  TONG KET RETRAIN CROSS-DATASET (DO THAT)")
        print("="*60)
        for r in results:
            print(f"  {r['dataset']}: Accuracy = {r['accuracy']*100:.2f}% | "
                  f"Time = {r['training_time']:.1f}s | Features = {r['features_used']}")
        print(f"\n[OK] Da luu ket qua -> {out_csv}")
    else:
        print("\n[CANH BAO] Khong co dataset nao chay duoc (thieu file du lieu).")
        print("  Hay dat UNSW-NB15 / BoT-IoT vao:")
        for k, c in DATASET_CONFIG.items():
            print(f"    - {c['name']}: {c['path']}")
        print("  Day la ket qua TRUNG THUC: chua co du lieu thi khong co so lieu (khong bia).")
