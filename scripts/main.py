"""
CyberDetect-MLP — Main Pipeline.

Paper references:
  - Algorithm 1 (line 799): Big Data Preprocessing
  - Algorithm 2 (line 803): Feature Selection (MI, Spark-based)
  - Algorithm 3 (line 835): Model Training
  - Algorithm 4 (line 839): Real-time Detection
  - Line 990: "Each experiment is repeated three times"
  - Line 870: "XAI-based interpretation (Grad-CAM and SHAP)"
  - Fig 5 (line 1071): Training accuracy curve
  - Fig 6 (line 1076): Training loss curve
  - Line 1475: "We report per-class performance metrics"
"""

from preprocessing.spark_preprocessor import preprocess_data
from preprocessing.feature_selector import select_top_features
from preprocessing.dataset_splitter import stratified_split
from model.trainer import train_model, plot_training_history
from model.evaluator import evaluate, evaluate_per_class
from model.explainability import explain
from pipeline.realtime_detector import simulate_detection
from config import settings
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf


def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    # =====================================================
    # Step 1: Spark preprocessing (Algorithm 1, line 799)
    # =====================================================
    preprocessed = preprocess_data("data/raw/ton_iot.csv", "data/processed")

    # =====================================================
    # Step 2: Spark-based MI feature selection (Algorithm 2, line 803)
    # Paper line 868: "feature engineering based on Spark"
    # Paper line 982: "top-30 features selected"
    # =====================================================
    selected = select_top_features(preprocessed, "label", settings.TOP_K_FEATURES, "data/features")

    # =====================================================
    # Step 3: 80/20 stratified split (Paper line 1021)
    # =====================================================
    train_csv, val_csv, test_csv = stratified_split(selected, "data/features")

    # =====================================================
    # Step 4-5: Train + Evaluate — 3 repetitions
    # Paper line 990: "Each experiment is repeated three times
    # to account for stochastic variations"
    # =====================================================
    all_results = []

    for run in range(3):
        print(f"\n{'-'*60}")
        print(f"--- Bat dau Experiment Run {run+1}/3 (seed={42+run}) ---")
        print(f"{'-'*60}")
        set_seeds(42 + run)

        # Algorithm 3: Huan luyen model voi SMOTE + class weighting
        model, le, history = train_model(train_csv, val_csv, f"saved_models_run_{run}")

        # Tao Fig 5 & Fig 6 o run cuoi cung
        if run == 2:
            plot_training_history(history, "results/training")

        # Evaluate on held-out test set
        acc, prec, rec, f1, roc = evaluate(model, test_csv, le)

        all_results.append({
            'Run': run + 1,
            'Seed': 42 + run,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc,
        })

    # =====================================================
    # Save results to CSV (GAP-10)
    # Paper line 990-991: "final model parameters are selected
    # based on the best validation performance"
    # =====================================================
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/training_results_3runs.csv", index=False)

    # In ket qua trung binh (Mean ± Std)
    print("\n" + "-"*60)
    print("[OK] KET QUA CUOI CUNG (TRUNG BINH 3 RUNS):")
    print(f"  Accuracy:  {results_df['Accuracy'].mean():.4f} +/- {results_df['Accuracy'].std():.4f}")
    print(f"  Precision: {results_df['Precision'].mean():.4f} +/- {results_df['Precision'].std():.4f}")
    print(f"  Recall:    {results_df['Recall'].mean():.4f} +/- {results_df['Recall'].std():.4f}")
    print(f"  F1-Score:  {results_df['F1-Score'].mean():.4f} +/- {results_df['F1-Score'].std():.4f}")
    print(f"  ROC-AUC:   {results_df['ROC-AUC'].mean():.4f} +/- {results_df['ROC-AUC'].std():.4f}")
    print("-"*60)
    print(f"[INFO] Ket qua da duoc luu tai results/training_results_3runs.csv")

    # =====================================================
    # Step 5.5: Per-class evaluation (Paper line 1475)
    # "We report per-class performance metrics"
    # =====================================================
    evaluate_per_class(model, test_csv, le, "results")

    # =====================================================
    # Step 6: Explainability — SHAP + IG + Grad-CAM
    # Paper line 870, 1219, 1251
    # =====================================================
    explain(model, test_csv, "results/xai")

    # =====================================================
    # Step 7: Real-time detection simulation (Algorithm 4)
    # Paper line 947: logs to MongoDB (with JSON fallback)
    # =====================================================
    simulate_detection(
        f"saved_models_run_2/cyberdetect_mlp.h5",
        test_csv,
        "results/alerts.json"
    )

    print("\n[FINISH] Pipeline CyberDetect-MLP hoan tat!")
    print("[NOTE] De chay phan tich thong ke 10-run (Table 4, Fig 9),")
    print("   chay: python statistical_test.py <train_csv> <test_csv>")
    print("[NOTE] De chay so sanh voi baseline (Table 3, Fig 7-8),")
    print("   chay: python baseline_comparison.py <train_csv> <test_csv>")


if __name__ == "__main__":
    main()
