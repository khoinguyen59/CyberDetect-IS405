import numpy as np
from imblearn.combine import SMOTEENN
import time

def apply_smote_enn(X_train, y_train, random_state=42):
    """
    Applies SMOTE-ENN to handle class imbalance.
    SMOTE oversamples minority classes, and ENN (Edited Nearest Neighbours) 
    cleans up the noisy samples that overlap the classes.
    """
    print("\n" + "="*50)
    print("[INFO] Bắt đầu xử lý mất cân bằng dữ liệu: SMOTE-ENN")
    print("="*50)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[*] Phân phối class TRƯỚC khi xử lý:")
    for cls, count in zip(unique, counts):
        print(f"    - Class {cls}: {count} samples")
        
    start_time = time.time()
    
    # Initialize SMOTE-ENN
    # sampling_strategy='auto' resamples all classes but the majority class
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=random_state)
    
    # Apply resampling
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    unique_res, counts_res = np.unique(y_resampled, return_counts=True)
    print(f"\n[*] Phân phối class SAU khi xử lý (SMOTE-ENN):")
    for cls, count in zip(unique_res, counts_res):
        print(f"    - Class {cls}: {count} samples")
        
    print(f"[OK] Hoàn thành SMOTE-ENN trong {elapsed:.2f} giây.")
    return X_resampled, y_resampled
