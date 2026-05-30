import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Add 'src' folder to sys.path so we can import modules
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(base_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.cyberdetect_mlp import build_model
from modules.cross_dataset.cic_ids2018_loader import load_and_preprocess_ids2018
from modules.resampling.smote_enn import apply_smote_enn
from modules.resampling.adasyn import apply_adasyn

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain CyberDetect-MLP on CSE-CIC-IDS2018 dataset.")
    parser.add_argument("--data", default="data/raw/CSE-CIC-IDS2018/ids2018_sample.csv",
                        help="Path to CSE-CIC-IDS2018 CSV file.")
    parser.add_argument("--sample-rows", type=int, default=200000,
                        help="Number of rows to sample for training (default: 200,000). Use -1 for all.")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of top features to select using Mutual Information.")
    parser.add_argument("--resampling", choices=["smote_enn", "adasyn", "none"], default="smote_enn",
                        help="Resampling method to apply on training data.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

def main():
    args = parse_args()
    set_seeds(args.seed)
    
    print("\n" + "="*70)
    print("  STARTING PIPELINE RETRAIN CYBERDETECT-MLP ON CSE-CIC-IDS2018")
    print("="*70)
    print(f"[*] Configuration:")
    print(f"    - Data file: {args.data}")
    print(f"    - Rows: {'All' if args.sample_rows == -1 else args.sample_rows}")
    print(f"    - Top K features selected: {args.top_k}")
    print(f"    - Resampling: {args.resampling.upper()}")
    print(f"    - Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"    - Seed: {args.seed}")
    print("="*70)
    
    # 1. Load data
    sample_size = None if args.sample_rows == -1 else args.sample_rows
    X_raw, y_raw, feature_names = load_and_preprocess_ids2018(args.data, sample_size)
    
    # Map labels to binary (0: Normal/Benign, 1: Attack)
    unique_labels = np.unique(y_raw)
    print(f"[*] Detected raw labels: {unique_labels}")
    
    y = np.where(pd.Series(y_raw).astype(str).str.lower() == 'benign', 0, 1)
    print(f"[*] Binary label distribution: Normal (0) = {np.sum(y == 0)}, Attack (1) = {np.sum(y == 1)}")
    
    # 2. Mutual Information Feature Selection
    print(f"\n[*] Calculating Mutual Information scores for {len(feature_names)} features...")
    t_start_mi = time.time()
    
    # Subsample 50,000 rows for MI calculation if dataset is large
    if len(X_raw) > 50000:
        print("[*] Dataset large, subsampling 50,000 rows for MI computation...")
        idx_mi = np.random.choice(len(X_raw), 50000, replace=False)
        mi_scores = mutual_info_classif(X_raw[idx_mi], y[idx_mi], random_state=args.seed)
    else:
        mi_scores = mutual_info_classif(X_raw, y, random_state=args.seed)
        
    t_elapsed_mi = time.time() - t_start_mi
    print(f"[OK] MI calculation completed in {t_elapsed_mi:.2f} seconds.")
    
    # Get Top K features
    top_indices = np.argsort(mi_scores)[::-1][:args.top_k]
    selected_features = [feature_names[i] for i in top_indices]
    
    print(f"\n[*] Top {args.top_k} features selected by Mutual Information:")
    for rank, (idx, name) in enumerate(zip(top_indices, selected_features)):
        print(f"    {rank+1:2d}. {name:<30} (MI score: {mi_scores[idx]:.4f})")
        
    # Save selected features list
    models_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    features_json_path = os.path.join(models_dir, "selected_features_ids2018.json")
    with open(features_json_path, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, indent=4)
    print(f"\n[INFO] Selected features saved to: {features_json_path}")
    
    # Filter features
    X_selected = X_raw[:, top_indices]
    
    # 3. Train/Test Split (80/20 Stratified)
    print(f"\n[*] Splitting dataset 80/20 (Stratified Split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"    - Train Set: {X_train.shape[0]} samples")
    print(f"    - Test Set:  {X_test.shape[0]} samples")
    
    # 4. Resampling on train data
    if args.resampling == "smote_enn":
        X_train_res, y_train_res = apply_smote_enn(X_train, y_train, random_state=args.seed)
    elif args.resampling == "adasyn":
        X_train_res, y_train_res = apply_adasyn(X_train, y_train, random_state=args.seed)
    else:
        print("\n[INFO] Skipping resampling, using original train data.")
        X_train_res, y_train_res = X_train, y_train
        
    # 5. MinMaxScaler Normalization
    print(f"\n[*] Normalizing features with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
    
    # 6. Model initialization and training
    input_dim = X_train_res.shape[1]
    num_classes = 2
    
    print(f"\n[*] Initializing CyberDetect-MLP model...")
    model = build_model(input_dim=input_dim, num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    checkpoint_path = os.path.join(models_dir, "checkpoint_ids2018.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    ]
    
    print(f"\n[*] Starting training...")
    start_train_time = time.time()
    history = model.fit(
        X_train_res, y_train_res,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    elapsed_train_time = time.time() - start_train_time
    print(f"[OK] Training completed in {elapsed_train_time:.2f} seconds.")
    
    # 7. Evaluate on Test Set
    print(f"\n[*] Evaluating on Test Set...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    except Exception:
        roc_auc = np.nan
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    print("\n" + "="*50)
    print("  EVALUATION RESULTS ON CSE-CIC-IDS2018")
    print("="*50)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc*100:.2f}%" if not np.isnan(roc_auc) else "  ROC-AUC:   N/A")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0))
    
    # 8. Save model and results
    model_save_path = os.path.join(models_dir, "cyberdetect_mlp_ids2018.h5")
    model.save(model_save_path)
    print(f"\n[INFO] Optimal model saved at: {model_save_path}")
    
    results_dir = os.path.join(base_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = os.path.join(results_dir, "ids2018_retrain_results.csv")
    
    results_data = {
        "Dataset": ["CSE-CIC-IDS2018"],
        "SampleRows": [args.sample_rows],
        "TopKFeatures": [args.top_k],
        "Resampling": [args.resampling],
        "Accuracy": [acc],
        "Precision": [precision],
        "Recall": [recall],
        "F1-Score": [f1],
        "ROC-AUC": [roc_auc],
        "TrainingTimeSec": [elapsed_train_time]
    }
    
    pd.DataFrame(results_data).to_csv(results_csv_path, index=False)
    print(f"[INFO] Results CSV report saved at: {results_csv_path}")
    print("\n[FINISH] Pipeline ran successfully!")

if __name__ == "__main__":
    main()
