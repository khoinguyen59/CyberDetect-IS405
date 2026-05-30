import numpy as np
from imblearn.combine import SMOTEENN
import time

def apply_smote_enn(X_train, y_train, random_state=42):
    """
    Applies SMOTE-ENN to handle class imbalance.
    """
    print("\n" + "="*50)
    print("[INFO] Starting class imbalance handling: SMOTE-ENN")
    print("="*50)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"[*] Class distribution BEFORE resampling:")
    for cls, count in zip(unique, counts):
        print(f"    - Class {cls}: {count} samples")
        
    start_time = time.time()
    
    # Initialize SMOTE-ENN
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=random_state)
    
    # Apply resampling
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    unique_res, counts_res = np.unique(y_resampled, return_counts=True)
    print(f"\n[*] Class distribution AFTER resampling (SMOTE-ENN):")
    for cls, count in zip(unique_res, counts_res):
        print(f"    - Class {cls}: {count} samples")
        
    print(f"[OK] Completed SMOTE-ENN in {elapsed:.2f} seconds.")
    return X_resampled, y_resampled
