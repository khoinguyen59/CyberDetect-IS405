# -*- coding: utf-8 -*-
"""
Kiem dinh thong ke - 10 lan chay doc lap (Table 4 / Fig 9).

Self-contained & MULTICLASS:
  - Doc artifacts da lop tu NB01 (data/colab_processed/).
  - Huan luyen CyberDetect-MLP (full: BN + Dropout + cosine annealing + class_weight balanced)
    10 lan voi 10 seed khac nhau -> Mean +/- Std + boxplot.
  - KHONG dung model.trainer (von ap SMOTE sampling_strategy='auto' qua tay).
  - Sua loi ModuleNotFoundError: khong con import 'model.*'.
Chay: `python scripts/statistical_test.py` (khong can tham so).
"""
import os
import sys
import math
import random
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight


def project_root():
    here = os.getcwd()
    cand = [here, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    for base in cand:
        if os.path.exists(os.path.join(base, "data", "colab_processed", "X_train_top30.parquet")):
            return base
    return cand[0]


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def cosine(e, T=100):
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * e / T))


def build(dim, n_classes):
    m = Sequential([Input(shape=(dim,))])
    for u in (512, 256, 128):
        m.add(Dense(u, activation="relu")); m.add(BatchNormalization()); m.add(Dropout(0.3))
    m.add(Dense(n_classes, activation="softmax"))
    m.compile(Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    root = project_root()
    proc = os.path.join(root, "data", "colab_processed")
    parser = argparse.ArgumentParser(description="10-run statistical analysis (Table 4, Fig 9)")
    parser.add_argument("--output_dir", default=os.path.join(root, "results", "statistical"))
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    req = ["X_train_top30.parquet", "X_test_top30.parquet", "y_train.parquet", "y_test.parquet"]
    for f in req:
        if not os.path.exists(os.path.join(proc, f)):
            print(f"[ERROR] Thieu {f} trong {proc}. Chay notebook 01 truoc.")
            sys.exit(1)

    X_train = pd.read_parquet(os.path.join(proc, "X_train_top30.parquet")).values.astype("float32")
    X_test = pd.read_parquet(os.path.join(proc, "X_test_top30.parquet")).values.astype("float32")
    y_train = pd.read_parquet(os.path.join(proc, "y_train.parquet"))["label"].values.astype(int)
    y_test = pd.read_parquet(os.path.join(proc, "y_test.parquet"))["label"].values.astype(int)
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    print(f"[INFO] Train {X_train.shape} | classes={n_classes} | {args.runs} runs")

    metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]}
    for run in range(args.runs):
        seed = 42 + run
        print(f"\n{'-'*50}\n[INFO] Run {run+1}/{args.runs} (seed={seed})\n{'-'*50}")
        set_seeds(seed)
        model = build(X_train.shape[1], n_classes)
        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
                  class_weight=dict(enumerate(w)),
                  callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                  verbose=0)
        prob = model.predict(X_test, verbose=0); yp = prob.argmax(axis=1)
        metrics["accuracy"].append(accuracy_score(y_test, yp))
        metrics["precision"].append(precision_score(y_test, yp, average="macro", zero_division=0))
        metrics["recall"].append(recall_score(y_test, yp, average="macro", zero_division=0))
        metrics["f1_score"].append(f1_score(y_test, yp, average="macro", zero_division=0))
        try:
            metrics["roc_auc"].append(roc_auc_score(y_test, prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc"].append(float("nan"))
        tf.keras.backend.clear_session()

    pd.DataFrame(metrics).to_csv(os.path.join(args.output_dir, "raw_10runs.csv"), index_label="run")
    summary = []
    for k, v in metrics.items():
        v = np.array(v, dtype=float)
        summary.append({"Metric": k, "Mean": round(np.nanmean(v), 4), "Std": round(np.nanstd(v), 4),
                        "Mean±Std": f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}",
                        "Min": round(np.nanmin(v), 4), "Max": round(np.nanmax(v), 4)})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(args.output_dir, "table4_statistics.csv"), index=False)
    print("\n[OK] Table 4:\n", sdf.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=pd.DataFrame({"F1-Score": metrics["f1_score"]}), ax=axes[0], palette="Set2")
    axes[0].set_title("F1-Score qua 10 runs", fontweight="bold")
    allm = pd.DataFrame(metrics); allm.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    sns.boxplot(data=allm, ax=axes[1], palette="Set3"); axes[1].set_title("Tat ca metrics (10 runs)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fig9_boxplot.png"), dpi=150); plt.close()
    print("[OK] Fig 9 ->", os.path.join(args.output_dir, "fig9_boxplot.png"))
    print("\n[FINISH] Kiem dinh thong ke hoan tat (da lop, class_weight, khong SMOTE qua tay).")


if __name__ == "__main__":
    main()
