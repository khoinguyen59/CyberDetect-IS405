import os
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # Tránh lỗi GUI trên Colab (Lỗi Phase 1)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model.cyberdetect_mlp import build_model
from modules.resampling.smote_enn import apply_smote_enn
from modules.resampling.adasyn import apply_adasyn

def retrain_with_resampling(data_path, method="smote_enn"):
    """
    Huấn luyện lại mô hình MLP với các thuật toán Resampling nâng cao 
    (SMOTE-ENN hoặc ADASYN) thay vì SMOTE thuần của bài báo gốc.
    """
    print("="*50)
    print(f"[INFO] BẮT ĐẦU RETRAIN VỚI PHƯƠNG PHÁP: {method.upper()}")
    print("="*50)
    
    # 1. Load Data
    print(f"[*] Load dữ liệu từ {data_path}...")
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy dữ liệu tại {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Giả định dữ liệu đã qua tiền xử lý (đã có 30 features)
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    
    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Áp dụng Resampling
    if method == "smote_enn":
        X_train_res, y_train_res = apply_smote_enn(X_train, y_train)
    elif method == "adasyn":
        X_train_res, y_train_res = apply_adasyn(X_train, y_train)
    else:
        print("[WARNING] Không dùng resampling hoặc method không hợp lệ, dùng dữ liệu gốc.")
        X_train_res, y_train_res = X_train, y_train
        
    # Normalize data (MinMax fallback)
    print("[*] Normalizing data...")
    x_min, x_max = X_train_res.min(axis=0), X_train_res.max(axis=0)
    X_train_res = (X_train_res - x_min) / (x_max - x_min + 1e-8)
    X_test = (X_test - x_min) / (x_max - x_min + 1e-8)
    
    # 3. Tạo và Huấn luyện Mô hình
    input_dim = X_train_res.shape[1]
    num_classes = len(np.unique(y))
    
    # Giữ nguyên kiến trúc 2-node softmax như Phase 1 để so sánh công bằng
    # Phase 1 model: Dense(2, softmax) + sparse_categorical_crossentropy
    loss_fn = 'sparse_categorical_crossentropy'
        
    print(f"[*] Khởi tạo kiến trúc MLP (Input: {input_dim}, Classes: {num_classes})")
    model = build_model(input_dim=input_dim, num_classes=num_classes)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # [FIX PHASE 1] Lưu checkpoint phòng hờ đứt kết nối Colab
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{method}.h5")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    print("[*] Bắt đầu quá trình huấn luyện (Training)...")
    start_train = time.time()
    history = model.fit(
        X_train_res, y_train_res,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    print(f"[OK] Training hoàn tất trong {time.time() - start_train:.2f} giây.")
    
    # 4. Đánh giá
    print("\n[*] Đánh giá trên tập TEST:")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Dùng sklearn trực tiếp thay vì gọi evaluate_per_class (khác signature)
    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
    
    # 5. Lưu mô hình
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"cyberdetect_mlp_{method}.h5")
    model.save(model_path)
    print(f"[INFO] Đã lưu mô hình mới tại: {model_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "processed", "train_processed.csv")
    
    import sys
    # Cho phép chọn method qua command line: python retrain_resampling.py [method]
    # method: all, none, smote_enn, adasyn
    method_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if method_arg == "all":
        print("\n" + "="*60)
        print("  CHẠY TẤT CẢ 3 THÍ NGHIỆM RESAMPLING")
        print("="*60)
        retrain_with_resampling(data_path, method="none")
        retrain_with_resampling(data_path, method="smote_enn")
        retrain_with_resampling(data_path, method="adasyn")
    else:
        retrain_with_resampling(data_path, method=method_arg)
