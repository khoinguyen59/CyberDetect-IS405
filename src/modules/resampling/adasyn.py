import numpy as np
from imblearn.over_sampling import ADASYN
import time

def apply_adasyn(X_train, y_train, random_state=42):
    """
    Applies ADASYN (Adaptive Synthetic) to handle class imbalance.
    """
    print("\n" + "="*50)
    print("[INFO] Starting class imbalance handling: ADASYN")
    print("="*50)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[*] Class distribution BEFORE resampling:")
    for cls, count in zip(unique, counts):
        print(f"    - Class {cls}: {count} samples")
        
    start_time = time.time()
    
    # Initialize ADASYN
    adasyn = ADASYN(sampling_strategy='auto', random_state=random_state)
    
    # Apply resampling
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    unique_res, counts_res = np.unique(y_resampled, return_counts=True)
    print(f"\n[*] Class distribution AFTER resampling (ADASYN):")
    for cls, count in zip(unique_res, counts_res):
        print(f"    - Class {cls}: {count} samples")
        
    print(f"[OK] Completed ADASYN in {elapsed:.2f} seconds.")
    return X_resampled, y_resampled
