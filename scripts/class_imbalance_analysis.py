# -*- coding: utf-8 -*-
"""
So sanh chien luoc xu ly mat can bang lop (MULTICLASS) tren TON_IoT.

Nguon du lieu: artifacts da lop do notebook 01 sinh ra (data/colab_processed/).
Khong co bo gia lap, khong co duong dan cung. Yeu cau TensorFlow + imbalanced-learn.

4 cau hinh: baseline | class_weight | smote_enn | hybrid (smote_enn + class_weight).
Xuat F1-score TUNG LOP (that su do tu mo hinh) -> results/class_imbalance_metrics.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf  # bat buoc - khong fallback gia lap
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

EPOCHS = 100
BATCH = 64


def project_root():
    """Tim thu muc goc chua data/colab_processed (uu tien CWD, roi parent cua scripts/)."""
    here = os.getcwd()
    cand = [here, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    for base in cand:
        if os.path.exists(os.path.join(base, "data", "colab_processed", "X_train_top30.parquet")):
            return base
    return cand[0]


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(input_dim, n_classes):
    m = Sequential([Input(shape=(input_dim,))])
    for u in (512, 256, 128):
        m.add(Dense(u, activation="relu"))
        m.add(BatchNormalization())
        m.add(Dropout(0.3))
    m.add(Dense(n_classes, activation="softmax"))
    m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train_and_perclass_f1(X_train, y_train, X_test, y_test, n_classes, method):
    """Huan luyen MLP da lop voi 1 chien luoc, tra ve F1 tung lop (do that tu test set)."""
    Xtr, ytr = X_train, y_train

    if method in ("smote_enn", "hybrid"):
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(ytr)
        k = max(1, min(5, counts[counts > 0].min() - 1))
        print(f"    [*] Ap dung SMOTE-ENN (k_neighbors={k}) tren tap TRAIN...")
        smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=42), random_state=42)
        Xtr, ytr = smote_enn.fit_resample(X_train, y_train)

    class_weight = None
    if method in ("class_weight", "hybrid"):
        w = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        class_weight = dict(enumerate(w))

    set_seeds(42)
    model = build_mlp(Xtr.shape[1], n_classes)
    model.fit(
        Xtr, ytr,
        validation_split=0.2,
        epochs=EPOCHS, batch_size=BATCH,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=0,
    )
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    # F1 tung lop, tinh tren toan bo nhan 0..n_classes-1 de thang hang voi class_names
    return f1_score(y_test, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)


def main():
    set_seeds(42)
    root = project_root()
    proc = os.path.join(root, "data", "colab_processed")
    results_dir = os.path.join(root, "results")
    xai_dir = os.path.join(results_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)

    meta_path = os.path.join(root, "models", "colab_preprocessing_metadata.json")
    if not os.path.exists(meta_path):
        print(f"[ERROR] Khong tim thay {meta_path}. Hay chay notebook 01 truoc.")
        sys.exit(1)
    meta = json.load(open(meta_path, encoding="utf-8"))
    class_names = meta["class_names"]
    n_classes = meta["n_classes"]
    if n_classes <= 2:
        print("[ERROR] Metadata khong phai da lop. Chay NB01 voi LABEL_COL='type'.")
        sys.exit(1)

    req = ["X_train_top30.parquet", "X_test_top30.parquet", "y_train.parquet", "y_test.parquet"]
    for f in req:
        if not os.path.exists(os.path.join(proc, f)):
            print(f"[ERROR] Thieu artifact {f} trong {proc}. Hay chay notebook 01 truoc.")
            sys.exit(1)

    X_train = pd.read_parquet(os.path.join(proc, "X_train_top30.parquet")).values.astype("float32")
    X_test = pd.read_parquet(os.path.join(proc, "X_test_top30.parquet")).values.astype("float32")
    y_train = pd.read_parquet(os.path.join(proc, "y_train.parquet"))["label"].values.astype(int)
    y_test = pd.read_parquet(os.path.join(proc, "y_test.parquet"))["label"].values.astype(int)

    print("=" * 70)
    print("  SO SANH XU LY MAT CAN BANG (MULTICLASS) - F1 TUNG LOP (DO THAT)")
    print("=" * 70)
    print(f"[*] {n_classes} lop: {class_names}")
    print(f"[*] Train {X_train.shape}, Test {X_test.shape}")

    methods = ["baseline", "class_weight", "smote_enn", "hybrid"]
    results = {}
    for method in methods:
        print(f"\n[*] Huan luyen cau hinh: {method.upper()} ({EPOCHS} epochs)...")
        results[method] = train_and_perclass_f1(X_train, y_train, X_test, y_test, n_classes, method)

    rows = []
    for idx, name in enumerate(class_names):
        rows.append({
            "Class": name,
            "Baseline_F1": results["baseline"][idx],
            "ClassWeight_F1": results["class_weight"][idx],
            "SMOTE_ENN_F1": results["smote_enn"][idx],
            "Hybrid_F1": results["hybrid"][idx],
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "class_imbalance_metrics.csv")
    df.to_csv(out_csv, index=False)
    print("\n[OK] Da luu F1 tung lop (do that) ->", out_csv)
    print(df.to_string(index=False))

    # Bieu do cho cac lop thieu so thuc su trong dataset (it mau nhat)
    counts = np.bincount(y_train, minlength=n_classes)
    minority_idx = list(np.argsort(counts)[:4])
    labels = [class_names[i] for i in minority_idx]
    x = np.arange(len(labels)); width = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - 1.5 * width, [results["baseline"][i] for i in minority_idx], width, label="Baseline", color="#7f7f7f")
    ax.bar(x - 0.5 * width, [results["class_weight"][i] for i in minority_idx], width, label="Class Weight", color="#1f77b4")
    ax.bar(x + 0.5 * width, [results["smote_enn"][i] for i in minority_idx], width, label="SMOTE-ENN", color="#ff7f0e")
    ax.bar(x + 1.5 * width, [results["hybrid"][i] for i in minority_idx], width, label="Hybrid", color="#2ca02c")
    ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1.1)
    ax.set_title("F1 tung lop cho cac lop THIEU SO nhat (do that)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in labels]); ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(xai_dir, "class_imbalance_comparison.png")
    plt.savefig(out_png, dpi=150); plt.close()
    print("[OK] Bieu do ->", out_png)
    print("\n[FINISH] Hoan tat (khong dung so lieu gia lap).")


if __name__ == "__main__":
    main()
