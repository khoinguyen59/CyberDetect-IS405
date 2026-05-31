# -*- coding: utf-8 -*-
"""
Phan tich Concept Drift cho CyberDetect-MLP (DO THAT, khong gia lap).

Thiet ke thuc nghiem (minh bach):
  - Tao mot luong streaming TONG HOP co chu dich tu chinh TON_IoT:
      * Pha 1 (on dinh): mau theo phan bo tu nhien cua dataset.
      * Pha 2 (drift):   tang ty le tan cong + COVARIATE SHIFT (nhan cac dac trung
        byte/packet/duration) -> phan bo dau vao lech khoi luc train.
  - Tien xu ly tung dong bang DUNG preprocessor + scaler + top-features da luu o NB01
    (models/colab_preprocessor.joblib) -> dac trung khop voi mo hinh.
  - Rolling accuracy = do THAT tu du doan cua mo hinh so voi nhan that.
  - KS statistic do tren mot dac trung so thuc (src_bytes) giua cua so tham chieu va hien tai.

Yeu cau: mo hinh da train (models/colab_cyberdetect_mlp.h5) + preprocessor (NB01).
Khong co duong dan cung, khong co nhanh sinh so ngau nhien.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import tensorflow as tf  # bat buoc
import joblib


def project_root():
    here = os.getcwd()
    cand = [here, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    for base in cand:
        if os.path.exists(os.path.join(base, "models", "colab_cyberdetect_mlp.h5")):
            return base
    return cand[0]


def find_raw_csv(root):
    for c in ["data/ton_iot.csv", "data/raw/ton_iot.csv", "ton_iot.csv", "../ton_iot.csv"]:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    root = project_root()

    model_path = os.path.join(root, "models", "colab_cyberdetect_mlp.h5")
    prep_path = os.path.join(root, "models", "colab_preprocessor.joblib")
    meta_path = os.path.join(root, "models", "colab_preprocessing_metadata.json")
    for p in (model_path, prep_path, meta_path):
        if not os.path.exists(p):
            print(f"[ERROR] Thieu {p}. Hay chay notebook 01 + 02 truoc (de co preprocessor + model).")
            sys.exit(1)

    raw_csv = find_raw_csv(root)
    if raw_csv is None:
        print("[ERROR] Khong tim thay ton_iot.csv (dat o data/ton_iot.csv).")
        sys.exit(1)

    print("=" * 70)
    print("  PHAN TICH CONCEPT DRIFT (DO THAT TU MO HINH DA TRAIN)")
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)
    bundle = joblib.load(prep_path)
    pre = bundle["preprocessor"]; scaler_top = bundle["scaler_top"]
    top_idx = bundle["top_idx"]; le = bundle["label_encoder"]
    drop_always = bundle["drop_always"]; label_col = bundle["label_col"]
    meta = json.load(open(meta_path, encoding="utf-8"))
    n_classes = meta["n_classes"]
    print(f"[*] Model da train: {n_classes} lop | Du lieu: {raw_csv}")

    df = pd.read_csv(raw_csv, low_memory=False)

    def to_model_X(frame):
        """Raw rows -> dac trung khop mo hinh (dung preprocessor + top_idx + scaler da fit)."""
        Xr = frame.drop(columns=[c for c in drop_always if c in frame.columns], errors="ignore")
        all_feat = np.nan_to_num(pre.transform(Xr).astype(np.float32))
        return scaler_top.transform(all_feat[:, top_idx]).astype(np.float32)

    def true_y(frame):
        return le.transform(frame[label_col].astype(str))

    # --- Xay dung luong streaming tong hop ---
    print("\n[*] Tao luong streaming: Pha 1 (on dinh) + Pha 2 (drift + covariate shift)...")
    n1, n2 = 10000, 10000
    rng = np.random.RandomState(42)
    p1 = df.sample(n=min(n1, len(df)), random_state=101).reset_index(drop=True)

    p2 = df.sample(n=min(n2, len(df)), random_state=202).reset_index(drop=True)
    shift_cols = [c for c in ["src_bytes", "dst_bytes", "duration", "src_pkts", "dst_pkts",
                              "src_ip_bytes", "dst_ip_bytes"] if c in p2.columns]
    for c in shift_cols:
        p2[c] = pd.to_numeric(p2[c], errors="coerce").fillna(0) * 50.0  # covariate shift co chu dich
    print(f"    - Pha 1: {len(p1)} goi | Pha 2: {len(p2)} goi (shift x50 tren {len(shift_cols)} dac trung byte/pkt)")

    stream = pd.concat([p1, p2], ignore_index=True)
    X_stream = to_model_X(stream)
    y_stream = true_y(stream)
    # src_bytes raw (de do KS phan bo)
    raw_feature = pd.to_numeric(stream["src_bytes"], errors="coerce").fillna(0).values if "src_bytes" in stream.columns else X_stream[:, 0]

    # --- Sliding window: rolling accuracy THAT + KS ---
    W, STEP, DRIFT_AT = 1000, 100, len(p1)
    ref = raw_feature[:W]
    idxs, accs, ks_stats, pvals = [], [], [], []
    drift_points = []
    print("\n[*] Quet cua so truot, do rolling accuracy THAT...")
    for s in range(0, len(X_stream) - W, STEP):
        e = s + W
        prob = model.predict(X_stream[s:e], verbose=0)
        pred = prob.argmax(axis=1) if prob.shape[-1] > 1 else (prob.ravel() >= 0.5).astype(int)
        acc = float(np.mean(pred == y_stream[s:e]))
        ks, pv = ks_2samp(ref, raw_feature[s:e])
        idxs.append(e); accs.append(acc); ks_stats.append(float(ks)); pvals.append(float(pv))
        if pv < 0.001 and e > DRIFT_AT:
            drift_points.append(e)
        if s % 2000 == 0:
            print(f"    - idx {e:5d}: acc(THAT)={acc:.2%} | KS={ks:.3f} p={pv:.1e}")

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame({"Index": idxs, "RollingAccuracy": accs, "KSStatistic": ks_stats, "PValue": pvals}).to_csv(
        os.path.join(results_dir, "concept_drift_metrics.csv"), index=False)

    # Bieu do
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(idxs, accs, color="#1f77b4", linewidth=2.5, label="Rolling Accuracy (THAT)")
    ax1.set_xlabel("Streaming Event Index"); ax1.set_ylabel("Rolling Accuracy", color="#1f77b4")
    ax1.set_ylim(-0.05, 1.05); ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(idxs, ks_stats, color="#ff7f0e", linewidth=2, linestyle="--", label="KS Statistic")
    ax2.set_ylabel("KS Statistic", color="#ff7f0e"); ax2.set_ylim(-0.05, 1.05)
    ax1.axvline(x=DRIFT_AT, color="red", linestyle=":", linewidth=2, label=f"Drift injected (t={DRIFT_AT})")
    if drift_points:
        print(f"[OK] Phat hien drift dau tien tai t={drift_points[0]} (do tre {drift_points[0]-DRIFT_AT} su kien).")
    ax1.set_title("CONCEPT DRIFT - ROLLING ACCURACY (DO THAT) & KS", fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="lower left")
    plt.tight_layout()
    out_png = os.path.join(results_dir, "concept_drift_analysis.png")
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"[OK] Bieu do -> {out_png}")
    print(f"[OK] Metrics -> {os.path.join(results_dir, 'concept_drift_metrics.csv')}")
    print("\n[FINISH] Hoan tat (accuracy do that tu mo hinh, khong gia lap).")


if __name__ == "__main__":
    main()

# 