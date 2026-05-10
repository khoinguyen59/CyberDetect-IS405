# -*- coding: utf-8 -*-
"""
Do do tre suy luan & throughput (Table 8 / Fig 13) - DO THAT.

- model_path / test_csv co GIA TRI MAC DINH (artifacts tu NB01/02) -> chay duoc bang
  `!python scripts/benchmark_latency.py` khong can tham so (sua loi argparse cu).
- Cac so deu DO THAT bang time.perf_counter quanh model.predict. Khong con he so overhead bia.
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


def benchmark_inference_latency(model_path, test_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path)
    X = load_X(test_path, model.input_shape[-1])
    print(f"[INFO] Test samples: {X.shape}")

    streaming_rates = [10000, 50000, 100000]
    results = []
    for rate in streaming_rates:
        n = min(rate, len(X))
        Xb = X[:n]
        _ = model.predict(Xb[:100], verbose=0)  # warm-up
        start = time.perf_counter()
        _ = model.predict(Xb, verbose=0)
        elapsed = time.perf_counter() - start
        inf_ms = (elapsed / n) * 1000.0
        results.append({
            "Streaming Rate (events/sec)": rate,
            "Samples Tested": n,
            "Inference Time per Sample (ms)": round(inf_ms, 4),
            "Total Batch Latency (ms)": round(elapsed * 1000.0, 1),
            "Throughput (events/sec)": int(n / elapsed),
        })
        print(f"  [OK] rate={rate:,}: {inf_ms:.4f} ms/sample, throughput={int(n/elapsed):,}/s (DO THAT)")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "table8_latency.csv"), index=False)
    print("\n[OK] Table 8 ->", os.path.join(output_dir, "table8_latency.csv"))
    print(df.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df)); w = 0.35
    ax1.bar(x - w / 2, df["Inference Time per Sample (ms)"], w, label="Inference/sample (ms)", color="#3498db")
    ax1.set_ylabel("Inference Time per Sample (ms)", color="#3498db"); ax1.tick_params(axis="y", labelcolor="#3498db")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, df["Throughput (events/sec)"], w, label="Throughput (evt/s)", color="#e74c3c")
    ax2.set_ylabel("Throughput (events/sec)", color="#e74c3c"); ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{r:,}" for r in df["Streaming Rate (events/sec)"]])
    ax1.set_title("Inference Latency & Throughput (DO THAT)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig13_latency.png"), dpi=150); plt.close()
    print("[OK] Fig 13 ->", os.path.join(output_dir, "fig13_latency.png"))
    return df


if __name__ == "__main__":
    d_model, d_test, d_out = default_paths()
    parser = argparse.ArgumentParser(description="Inference latency benchmark (Table 8, Fig 13)")
    parser.add_argument("model_path", nargs="?", default=d_model, help="Trained model (.h5)")
    parser.add_argument("test_csv", nargs="?", default=d_test, help="Test data (.parquet or .csv)")
    parser.add_argument("--output_dir", default=d_out)
    args = parser.parse_args()
    if not os.path.exists(args.model_path) or not os.path.exists(args.test_csv):
        print(f"[ERROR] Thieu model hoac test data.\n  model={args.model_path}\n  test ={args.test_csv}\n"
              "  Hay chay notebook 01 + 02 truoc.")
        sys.exit(1)
    benchmark_inference_latency(args.model_path, args.test_csv, args.output_dir)
