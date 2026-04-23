import numpy as np
from imblearn.over_sampling import ADASYN
import time

def apply_adasyn(X_train, y_train, random_state=42):
    """
    Applies ADASYN (Adaptive Synthetic) to handle class imbalance.
    ADASYN uses a weighted distribution for different minority class examples 
    according to their level of difficulty in learning.
    """
    print("\n" + "="*50)
    print("[INFO] Bắt đầu xử lý mất cân bằng dữ liệu: ADASYN")
    print("="*50)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[*] Phân phối class TRƯỚC khi xử lý:")
    for cls, count in zip(unique, counts):
        print(f"    - Class {cls}: {count} samples")
        
    start_time = time.time()
    
    # Initialize ADASYN
    # sampling_strategy='auto' resamples all minority classes
    adasyn = ADASYN(sampling_strategy='auto', random_state=random_state)
    
    # Apply resampling
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    unique_res, counts_res = np.unique(y_resampled, return_counts=True)
    print(f"\n[*] Phân phối class SAU khi xử lý (ADASYN):")
    for cls, count in zip(unique_res, counts_res):
        print(f"    - Class {cls}: {count} samples")
        
    print(f"[OK] Hoàn thành ADASYN trong {elapsed:.2f} giây.")
    return X_resampled, y_resampled
