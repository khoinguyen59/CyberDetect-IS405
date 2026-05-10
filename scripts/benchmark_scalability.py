# -*- coding: utf-8 -*-
"""
Do kha nang mo rong & su dung tai nguyen (Table 9 / Fig 14) - DO THAT.

- Throughput DO THAT bang time.perf_counter quanh model.predict.
- CPU/Memory DO THAT bang psutil. Neu thieu psutil -> ghi NaN (KHONG bia so).
- GPU DO THAT bang GPUtil/nvidia-smi neu co; neu khong -> NaN (KHONG uoc luong).
- model_path/test co default -> chay duoc bang `!python scripts/benchmark_scalability.py`.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def project_root():
    here = os.getcwd()
    cand = [here, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    for base in cand:
        if os.path.exists(os.path.join(base, "models", "colab_cyberdetect_mlp.h5")):
            return base
    return cand[0]


def default_paths():
    root = project_root()
    return (os.path.join(root, "models", "colab_cyberdetect_mlp.h5"),
            os.path.join(root, "data", "colab_processed", "X_test_top30.parquet"),
            os.path.join(root, "results", "benchmark"))


def load_X(path, input_dim):
    if path.endswith(".parquet"):
        X = pd.read_parquet(path).values.astype(np.float32)
    else:
        df = pd.read_csv(path)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category").cat.codes
        df.fillna(0, inplace=True)
        X = df.drop("label", axis=1, errors="ignore").values.astype(np.float32)
    if X.shape[1] > input_dim:
        X = X[:, :input_dim]
    elif X.shape[1] < input_dim:
        X = np.hstack([X, np.zeros((X.shape[0], input_dim - X.shape[1]), dtype=np.float32)])
    return X


def read_gpu_util():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return round(gpus[0].load * 100, 1) if gpus else np.nan
    except Exception:
        return np.nan


def benchmark_scalability(model_path, test_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)
    X = load_X(test_path, model.input_shape[-1])

    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("[WARNING] Thieu psutil -> CPU/Mem se ghi NaN (KHONG bia so). Cai: pip install psutil")

    rows = []
    for rate in [10000, 50000, 100000]:
        n = min(rate, len(X))
        Xb = X[:n]
        _ = model.predict(Xb[:100], verbose=0)  # warm-up
        if has_psutil:
            psutil.cpu_percent(interval=None)  # reset
        start = time.perf_counter()
        _ = model.predict(Xb, verbose=0)
        elapsed = time.perf_counter() - start
        cpu = psutil.cpu_percent(interval=None) if has_psutil else np.nan
        mem = psutil.virtual_memory().percent if has_psutil else np.nan
        gpu = read_gpu_util()
        rows.append({
            "Streaming Rate (events/sec)": rate,
            "Samples Tested": n,
            "Throughput (events/sec)": int(n / elapsed),
            "CPU Utilization (%)": cpu,
            "GPU Utilization (%)": gpu,
            "Memory Usage (%)": mem,
        })
        print(f"  [OK] rate={rate:,}: throughput={int(n/elapsed):,}/s | CPU={cpu} GPU={gpu} Mem={mem} (DO THAT/NaN)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "table9_scalability.csv"), index=False)
    print("\n[OK] Table 9 ->", os.path.join(output_dir, "table9_scalability.csv"))
    print(df.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(df)); labels = [f"{r//1000}K" for r in df["Streaming Rate (events/sec)"]]
    ax1.bar(x, df["Throughput (events/sec)"], color="#2ecc71")
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_title("(a) Throughput (DO THAT)", fontweight="bold")
    ax1.set_ylabel("events/sec")
    w = 0.25
    ax2.bar(x - w, df["CPU Utilization (%)"], w, label="CPU %", color="#3498db")
    ax2.bar(x, df["GPU Utilization (%)"], w, label="GPU %", color="#e74c3c")
    ax2.bar(x + w, df["Memory Usage (%)"], w, label="Mem %", color="#f39c12")
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylim(0, 100)
    ax2.set_title("(b) Resource Utilization (psutil that / NaN)", fontweight="bold"); ax2.legend()
    plt.suptitle("Scalability & Resource Utilization (DO THAT)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig14_scalability.png"), dpi=150); plt.close()
    print("[OK] Fig 14 ->", os.path.join(output_dir, "fig14_scalability.png"))
    return df


if __name__ == "__main__":
    d_model, d_test, d_out = default_paths()
    parser = argparse.ArgumentParser(description="Scalability benchmark (Table 9, Fig 14)")
    parser.add_argument("model_path", nargs="?", default=d_model)
    parser.add_argument("test_csv", nargs="?", default=d_test)
    parser.add_argument("--output_dir", default=d_out)
    args = parser.parse_args()
    if not os.path.exists(args.model_path) or not os.path.exists(args.test_csv):
        print(f"[ERROR] Thieu model/test.\n  model={args.model_path}\n  test ={args.test_csv}\n  Chay NB01+02 truoc.")
        sys.exit(1)
    benchmark_scalability(args.model_path, args.test_csv, args.output_dir)
