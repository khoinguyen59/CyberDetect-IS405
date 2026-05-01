import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import json

from modules.cross_dataset.feature_aligner import FeatureAligner

def evaluate_zero_shot(dataset_path, dataset_name, model_path, feature_list_path):
    """
    Load saved model and evaluate on new unseen dataset (UNSW-NB15/BoT-IoT).
    """
    print("="*50)
    print(f"[INFO] Bắt đầu đánh giá Zero-Shot Transfer Learning: {dataset_name}")
    print("="*50)
    
    # 1. Load model and features list
    print("[*] Đang tải mô hình đã huấn luyện...")
    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy mô hình tại {model_path}")
        return
        
    model = load_model(model_path)
    
    if not os.path.exists(feature_list_path):
        # Fallback to hardcoded list if the JSON was not saved in Phase 1
        print("[WARNING] Không tìm thấy file JSON lưu 30 features, sử dụng danh sách mẫu...")
        # Lấy 30 tính năng mặc định đã phân tích từ TON_IoT (chỉ là ví dụ giả định)
        target_features = [f"Feature_{i}" for i in range(30)] 
    else:
        with open(feature_list_path, 'r') as f:
            target_features = json.load(f)
            
    # 2. Load New Dataset
    print(f"[*] Đang tải dữ liệu {dataset_name} từ {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Tập dữ liệu {dataset_path} không tồn tại. Vui lòng chạy download_datasets.py")
        return
        
    df_new = pd.read_csv(dataset_path)
    
    # Phát hiện cột nhãn tự động cho nhiều dataset khác nhau
    # UNSW-NB15: có cột 'label' (binary 0/1) 
    # BoT-IoT: có cột 'attack' (binary 0/1), KHÔNG có cột 'label'
    # TON_IoT: có cột 'label'
    label_candidates = ['label', 'attack', 'Label', 'attack_cat']
    label_col = None
    for candidate in label_candidates:
        if candidate in df_new.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df_new.columns[-1]
        print(f"[WARNING] Không tìm thấy cột label quen thuộc, dùng cột cuối: {label_col}")
    
    print(f"[*] Sử dụng cột nhãn: '{label_col}'")
    y_true = df_new[label_col].values
    
    # Convert label về dạng Binary (0: Normal, 1: Attack) nếu nó đang ở dạng text
    if df_new[label_col].dtype == object:
        y_true = np.where(df_new[label_col].str.lower().isin(['normal', 'benign', '0', 'none']), 0, 1)
    else:
        # Nếu numeric, đảm bảo giá trị nằm trong {0, 1}
        y_true = np.where(y_true == 0, 0, 1)
    
    # Loại bỏ cột nhãn + các cột meta không phải feature
    meta_cols = [label_col]
    for mc in ['category', 'subcategory', 'attack_cat', 'pkSeqID', 'id']:
        if mc in df_new.columns and mc != label_col:
            meta_cols.append(mc)
    X_raw = df_new.drop(columns=meta_cols)
    
    # 3. Align Features
    aligner = FeatureAligner(target_features)
    X_aligned = aligner.align(X_raw, dataset_name)
    
    # 4. Encode categorical columns rồi Normalize
    print("[*] Encoding và Normalizing data...")
    from sklearn.preprocessing import LabelEncoder
    for col in X_aligned.columns:
        if X_aligned[col].dtype == object:
            le = LabelEncoder()
            X_aligned[col] = le.fit_transform(X_aligned[col].astype(str))
    
    X_aligned = X_aligned.astype(float)
    X_aligned = (X_aligned - X_aligned.min()) / (X_aligned.max() - X_aligned.min() + 1e-8)
    
    # 5. Predict
    print("[*] Bắt đầu dự đoán (Predicting)...")
    y_pred_prob = model.predict(X_aligned)
    
    if y_pred_prob.shape[1] == 1:
        # Mô hình là dạng Binary (1 node output sigmoid)
        y_pred_binary = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        # Mô hình là dạng Multiclass
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        y_pred_binary = np.where(y_pred_class == 0, 0, 1)
    
    # 6. Report
    acc = accuracy_score(y_true, y_pred_binary)
    print(f"\n[KẾT QUẢ ZERO-SHOT TRÊN {dataset_name}]")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["Normal", "Attack"]))

if __name__ == "__main__":
    # Cấu hình đường dẫn cho các file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, "saved_models", "cyberdetect_mlp.h5")
    features_path = os.path.join(base_dir, "saved_models", "selected_features.json")
    
    # === Dataset 1: UNSW-NB15 (tải tự động từ HuggingFace) ===
    unsw_test_path = os.path.join(base_dir, "data", "raw", "UNSW-NB15", "UNSW_NB15_training-set.csv")
    if os.path.exists(unsw_test_path):
        evaluate_zero_shot(
            dataset_path=unsw_test_path, 
            dataset_name="UNSW-NB15", 
            model_path=model_path, 
            feature_list_path=features_path
        )
    else:
        print(f"[SKIP] UNSW-NB15 chưa được tải. Chạy download_datasets.py trước.")
    
    # === Dataset 2: BoT-IoT (đã có sẵn trong data/raw/BoT-IoT/) ===
    bot_test_path = os.path.join(base_dir, "data", "raw", "BoT-IoT", "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv")
    if os.path.exists(bot_test_path):
        evaluate_zero_shot(
            dataset_path=bot_test_path,
            dataset_name="BoT-IoT",
            model_path=model_path,
            feature_list_path=features_path
        )
    else:
        print(f"[SKIP] BoT-IoT chưa được tải. Vui lòng tải thủ công vào data/raw/BoT-IoT/")
