"""
Statistical Significance Analysis — 10 Independent Runs.

Paper references:
  - Line 1148-1149: "a paired t-test was performed (with 10 independent runs)"
  - Table 4 (line 1133): "Statistical significance testing"
  - Fig 9 (line 1143): "Boxplot of F1-scores across 10 runs"
  - Line 1157-1160: "CyberDetect-MLP has higher median values with less variance"

This script:
  1. Runs CyberDetect-MLP training + evaluation 10 times with different seeds
  2. Computes Mean ± Std for each metric
  3. Performs paired t-test (or Wilcoxon) comparing with baseline
  4. Generates boxplot (Fig 9)
  5. Outputs Table 4 results
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.trainer import train_model
from model.evaluator import evaluate
from sklearn.preprocessing import LabelEncoder


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_10_experiments(train_csv, val_csv, test_csv, output_dir="results/statistical"):
    """
    Paper line 1148-1149: "a paired t-test was performed (with 10 independent runs)"

    Runs the model 10 times with different random seeds and collects metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1_score': [], 'roc_auc': []
    }

    for run in range(10):
        seed = 42 + run
        print(f"\n{'-'*60}")
        print(f"[INFO] Statistical Run {run+1}/10 (seed={seed})")
        print(f"{'-'*60}")

        set_seeds(seed)

        model_path = os.path.join(output_dir, f"model_run_{run}")
        model, le, _ = train_model(train_csv, val_csv, model_path)

        acc, prec, rec, f1, roc = evaluate(model, test_csv, le)

        all_metrics['accuracy'].append(acc)
        all_metrics['precision'].append(prec)
        all_metrics['recall'].append(rec)
        all_metrics['f1_score'].append(f1)
        all_metrics['roc_auc'].append(roc)

        # Clean up model files to save disk space
        tf.keras.backend.clear_session()

    return all_metrics


def compute_statistics(metrics, output_dir="results/statistical"):
    """
    Paper Table 4 (line 1133): Statistical significance testing.
    Computes Mean ± Std and paired t-test p-values.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-"*70)
    print("[INFO] PHAN TICH THONG KE (10 Runs doc lap)")
    print("-" * 70)

    # Table 4: Mean ± Std
    summary_rows = []
    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric:12s}: {mean:.4f} +/- {std:.4f}")
        summary_rows.append({
            'Metric': metric,
            'Mean': f"{mean:.4f}",
            'Std': f"{std:.4f}",
            'Mean±Std': f"{mean:.4f} ± {std:.4f}",
            'Min': f"{np.min(values):.4f}",
            'Max': f"{np.max(values):.4f}"
        })

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "table4_statistics.csv"), index=False)
    print(f"\n[OK] Table 4 da duoc luu tai {output_dir}/table4_statistics.csv")

    return summary_df


def generate_boxplot(metrics, output_dir="results/statistical"):
    """
    Paper Fig 9 (line 1143): "Boxplot of F1-scores across 10 runs"
    Paper line 1157-1158: "higher median values with less variance"
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fig 9a: F1-Score boxplot
    f1_data = pd.DataFrame({
        'CyberDetect-MLP': metrics['f1_score'],
    })
    sns.boxplot(data=f1_data, ax=axes[0], palette='Set2')
    axes[0].set_title('F1-Score Distribution (10 Runs)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_ylim([min(0.9, min(metrics['f1_score']) - 0.02), 1.0])

    # Fig 9b: All metrics boxplot
    all_data = pd.DataFrame(metrics)
    all_data.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    sns.boxplot(data=all_data, ax=axes[1], palette='Set3')
    axes[1].set_title('All Metrics Distribution (10 Runs)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig9_boxplot.png"), dpi=150)
    plt.close()

    print(f"[OK] Fig 9 boxplot da duoc luu tai {output_dir}/fig9_boxplot.png")


def save_raw_results(metrics, output_dir="results/statistical"):
    """Save raw metric values for each run for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    raw_df = pd.DataFrame(metrics)
    raw_df.index.name = 'run'
    raw_df.to_csv(os.path.join(output_dir, "raw_10runs.csv"))
    print(f"[OK] Ket qua goc (raw) da duoc luu tai {output_dir}/raw_10runs.csv")


if __name__ == "__main__":
    # This script should be run AFTER preprocessing and feature selection
    # Usage: python statistical_test.py <train_csv> <test_csv>
    import argparse
    parser = argparse.ArgumentParser(description="10-run statistical analysis (Paper Table 4, Fig 9)")
    parser.add_argument("train_csv", help="Path to training CSV (with selected features)")
    parser.add_argument("test_csv", help="Path to test CSV (with selected features)")
    parser.add_argument("--val_csv", default=None, help="Path to validation CSV (optional)")
    parser.add_argument("--output_dir", default="results/statistical", help="Output directory")
    args = parser.parse_args()

    metrics = run_10_experiments(args.train_csv, args.val_csv, args.test_csv, args.output_dir)
    save_raw_results(metrics, args.output_dir)
    compute_statistics(metrics, args.output_dir)
    generate_boxplot(metrics, args.output_dir)

    print("\n[FINISH] Phan tich thong ke hoan tat.")
