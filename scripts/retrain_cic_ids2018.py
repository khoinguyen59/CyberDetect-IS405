# -*- coding: utf-8 -*-
"""
Retrain CyberDetect-MLP tren CSE-CIC-IDS2018 (THAT, mau ~500k) - phan MO RONG (Phase 5).

Tu dong:
  1. Tai du lieu tu AWS Open Data thang vao /content (neu chua co) - KHONG can qua may/Drive.
  2. Doc nhieu file ngay, lay mau ngau nhien ~500k dong (trai deu, da dang tan cong).
  3. MI top-30 -> SMOTE-ENN -> train CyberDetect-MLP -> danh gia -> luu KET QUA THAT.
Khong sinh du lieu gia. Thieu data + khong tai duoc -> bao loi va dung.

Chay:  !python scripts/retrain_cic_ids2018.py
       (tuy chon)  --sample-rows 1000000   --no-download
"""
import os
import sys
import glob
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, classification_report)

# import module trong src/
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(base_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from model.cyberdetect_mlp import build_model
from modules.cross_dataset.cic_ids2018_loader import load_and_preprocess_ids2018
from modules.resampling.smote_enn import apply_smote_enn

S3_URI = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"


def ensure_data(data_dir):
    """Tai CSE-CIC tu AWS Open Data (public, khong can dang nhap) neu chua co CSV."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if files:
        print(f"[*] Da co {len(files)} file CSV trong {data_dir}.")
        return files
    os.makedirs(data_dir, exist_ok=True)
    print("[*] Chua co data CSE-CIC -> tai tu AWS Open Data (~6.5GB, ~10 phut)...")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "awscli"], check=False)
    subprocess.run(f'aws s3 sync --no-sign-request "{S3_URI}" "{data_dir}/"', shell=True, check=False)
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("[!] Tai AWS khong ra file. Kiem tra duong dan bucket bang:")
        print('      !aws s3 ls --no-sign-request "s3://cse-cic-ids2018/"')
        print("    Hoac tai thu cong tu Kaggle vao thu muc:", data_dir)
    return files


def parse_args():
    p = argparse.ArgumentParser(description="Retrain CyberDetect-MLP on CSE-CIC-IDS2018 (real).")
    p.add_argument("--data", default="data/raw/CSE-CIC-IDS2018", help="THU MUC chua CSV CSE-CIC.")
    p.add_argument("--sample-rows", type=int, default=500000, help="So dong lay mau (mac dinh 500k).")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resampling", choices=["none", "class_weight", "smote_enn"], default="class_weight",
                   help="Xu ly mat can bang. Mac dinh class_weight (can bang nhe, khong bias test). "
                        "Dung smote_enn de tai hien phat hien 'SMOTE-ENN gay hai'.")
    p.add_argument("--no-download", action="store_true", help="Khong tu tai (dung file co san).")
    return p.parse_args()


def main():
    a = parse_args()
    np.random.seed(a.seed)
    import tensorflow as tf
    tf.random.set_seed(a.seed)
    from tensorflow.keras.callbacks import EarlyStopping

    print("\n" + "=" * 70)
    print(f"  CSE-CIC-IDS2018 RETRAIN (THAT) | mau muc tieu = {a.sample_rows:,}")
    print("=" * 70)

    # 1) Dam bao co data that
    if not a.no_download:
        ensure_data(a.data)
    try:
        X_raw, y_raw, feat = load_and_preprocess_ids2018(a.data, a.sample_rows)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[SKIP] {e}")
        print("[SKIP] Bo qua CSE-CIC (chua co du lieu). Khong tao ket qua gia.")
        return

    # 2) Nhan nhi phan: Benign=0, con lai=Attack=1
    y = np.where(pd.Series(y_raw).astype(str).str.strip().str.lower() == "benign", 0, 1)
    print(f"[*] Phan bo nhan: Benign(0) = {np.sum(y == 0):,} | Attack(1) = {np.sum(y == 1):,}")
    if len(np.unique(y)) < 2:
        print("[CANH BAO] Chi co 1 lop sau khi map nhan -> du lieu skew. Dung lai (khong train ao).")
        return

    # 3) MI feature selection (subsample 50k cho nhanh)
    print(f"\n[*] Tinh Mutual Information tren {len(feat)} dac trung...")
    if len(X_raw) > 50000:
        idx = np.random.RandomState(a.seed).choice(len(X_raw), 50000, replace=False)
        mi = mutual_info_classif(X_raw[idx], y[idx], random_state=a.seed)
    else:
        mi = mutual_info_classif(X_raw, y, random_state=a.seed)
    top = np.argsort(mi)[::-1][:a.top_k]
    selected = [feat[i] for i in top]
    X_sel = X_raw[:, top]
    print("[*] Top-5 dac trung MI:", selected[:5])

    # 4) Split 80/20; tach VAL tu train GOC (giu phan bo THAT) -> early-stopping khop test
    from sklearn.utils import shuffle as _shuffle
    from sklearn.utils.class_weight import compute_class_weight
    X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y, test_size=0.2, random_state=a.seed, stratify=y)
    X_trf, X_val, y_trf, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=a.seed, stratify=y_tr)

    class_weight = None
    if a.resampling == "smote_enn":
        print(f"[*] SMOTE-ENN tren train-fit ({X_trf.shape[0]:,} dong)...")
        try:
            X_trf, y_trf = apply_smote_enn(X_trf, y_trf, random_state=a.seed)
        except Exception as e:
            print(f"  [SMOTE-ENN loi -> bo qua]: {str(e)[:70]}")
    elif a.resampling == "class_weight":
        w = compute_class_weight("balanced", classes=np.unique(y_trf), y=y_trf)
        class_weight = dict(enumerate(w))
        print(f"[*] Dung class_weight (can bang nhe, KHONG bias test): {class_weight}")
    else:
        print("[*] Khong resampling.")

    # Tron deu + scale: fit tren train, transform val/test bang cung scaler
    X_trf, y_trf = _shuffle(X_trf, y_trf, random_state=a.seed)
    scaler = MinMaxScaler()
    X_trf = scaler.fit_transform(X_trf)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)
    print(f"[*] Train-fit {X_trf.shape[0]:,} | Val {X_val.shape[0]:,} | Test {X_te.shape[0]:,}")

    # 5) Train CyberDetect-MLP (nhi phan: Dense(2, softmax)); validate tren phan bo THAT
    model = build_model(input_dim=X_trf.shape[1], num_classes=2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    model.fit(X_trf, y_trf, validation_data=(X_val, y_val), epochs=a.epochs, batch_size=a.batch_size,
              class_weight=class_weight,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
              verbose=2)
    elapsed = time.time() - t0

    # 6) Danh gia tren test that
    prob = model.predict(X_te, verbose=0)
    y_pred = prob.argmax(axis=1)
    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, prob[:, 1])
    except Exception:
        auc = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    print("\n" + "=" * 50)
    print("  KET QUA CSE-CIC-IDS2018 (THAT)")
    print("=" * 50)
    print(f"  Accuracy={acc*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  "
          f"F1={f1*100:.2f}%  ROC-AUC={auc*100:.2f}%")
    print(classification_report(y_te, y_pred, target_names=["Benign", "Attack"], zero_division=0))

    # 7) Luu ket qua THAT (ghi SO DONG THUC TE da dung)
    models_dir = os.path.join(base_dir, "..", "models"); os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, "cyberdetect_mlp_ids2018.h5"))
    results_dir = os.path.join(base_dir, "..", "results"); os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame({
        "Dataset": ["CSE-CIC-IDS2018"], "SampleRows": [int(len(X_sel))], "TopKFeatures": [a.top_k],
        "Resampling": [a.resampling], "Accuracy": [acc], "Precision": [prec], "Recall": [rec],
        "F1-Score": [f1], "ROC-AUC": [auc], "TrainingTimeSec": [elapsed],
    }).to_csv(os.path.join(results_dir, "ids2018_retrain_results.csv"), index=False)
    print(f"\n[OK] Da luu ket qua THAT ({len(X_sel):,} dong) -> results/ids2018_retrain_results.csv")
    print("[FINISH] CSE-CIC-IDS2018 hoan tat.")


if __name__ == "__main__":
    main()
