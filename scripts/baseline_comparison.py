"""
Baseline Model Comparison — Table 3 (line 1114-1118).

Paper reference:
  - Table 3 (line 1120): "Performance comparison of proposed model with baseline classifiers"
  - Fig 7 (line 1112): "Confusion matrix comparison across models"
  - Fig 8 (line 1124): "Comparative performance of CyberDetect-MLP and baseline models"
  - Models: Random Forest, XGBoost, Vanilla MLP (no BN/Dropout/Scheduler)

Baseline targets from Table 3:
  | Model               | Acc   | Prec  | Rec   | F1    | AUC   |
  |---------------------|-------|-------|-------|-------|-------|
  | Random Forest       | 94.21 | 93.80 | 92.95 | 93.37 | 95.12 |
  | XGBoost             | 96.35 | 95.92 | 95.48 | 95.70 | 96.81 |
  | Vanilla MLP         | 97.12 | 96.85 | 96.30 | 96.57 | 97.44 |
  | CyberDetect-MLP     | 98.87 | 98.74 | 98.62 | 98.68 | 99.10 |
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.cyberdetect_mlp import build_model
from config import settings


def train_random_forest(X_train, y_train, X_test, y_test, n_classes):
    """Paper Table 3: Random Forest baseline."""
    print("\n[WAIT] Dang huan luyen Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    result = _compute_metrics(y_test, y_pred, y_prob, n_classes, "Random Forest")
    result['y_pred'] = y_pred
    return result


def train_xgboost(X_train, y_train, X_test, y_test, n_classes):
    """Paper Table 3: XGBoost baseline."""
    print("\n[WAIT] Dang huan luyen XGBoost...")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("⚠️ xgboost not installed. Run: pip install xgboost")
        return None
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)
    result = _compute_metrics(y_test, y_pred, y_prob, n_classes, "XGBoost")
    result['y_pred'] = y_pred
    return result


def train_vanilla_mlp(X_train, y_train, X_test, y_test, n_classes):
    """
    Paper Table 3: Vanilla MLP (no BatchNorm, no Dropout, no LR scheduler).
    Same layer sizes but without the optimizations of CyberDetect-MLP.
    """
    print("\n[WAIT] Dang huan luyen Vanilla MLP (no BN/Dropout/Scheduler)...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0,
              validation_split=0.2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    result = _compute_metrics(y_test, y_pred, y_prob, n_classes, "Vanilla MLP")
    result['y_pred'] = y_pred
    return result


def _compute_metrics(y_true, y_pred, y_prob, n_classes, model_name):
    """Compute all 5 metrics matching paper Table 3."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ROC-AUC requires one-hot for multiclass
    y_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    if y_prob.shape[1] == n_classes:
        roc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    else:
        roc = 0.0

    result = {
        'Model': model_name,
        'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc
    }
    print(f"  [OK] {model_name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={roc:.4f}")
    return result


def generate_comparison_plot(results_df, output_dir):
    """Paper Fig 8: grouped bar chart comparing all models."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=row['Model'], color=colors[i % len(colors)])

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: CyberDetect-MLP vs Baselines (Table 3)', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0.88, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig8_baseline_comparison.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 8 da duoc luu tai {output_dir}/fig8_baseline_comparison.png")


def generate_confusion_matrices(results, y_test, class_names, output_dir):
    """
    Paper Fig 7 (line 1079-1082): Confusion matrix comparison across models.

    "CyberDetect-MLP (d) demonstrates superior classification performance
    with minimal misclassifications across all attack classes."
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    labels = ['(a)', '(b)', '(c)', '(d)']

    for i, r in enumerate(results):
        if i >= 4:
            break
        y_pred = r.get('y_pred', None)
        if y_pred is None:
            continue
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar_kws={'shrink': 0.8})
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].set_title(f"{labels[i]} {r['Model']}", fontweight='bold')

    # Hide unused axes
    for i in range(n_models, 4):
        axes[i].set_visible(False)

    plt.suptitle('Confusion Matrix Comparison Across Models (Fig 7)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_confusion_matrices.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 7 da duoc luu tai {output_dir}/fig7_confusion_matrices.png")


def run_baseline_comparison(train_csv, test_csv, output_dir="results/baselines"):
    """Full baseline comparison pipeline — Paper Table 3, Fig 7-8."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    le = LabelEncoder()

    X_train = train_df.drop('label', axis=1).values
    y_train = le.fit_transform(train_df['label'].values)
    X_test = test_df.drop('label', axis=1).values
    y_test = le.transform(test_df['label'].values)
    n_classes = len(le.classes_)

    results = []

    # 1. Random Forest
    r = train_random_forest(X_train, y_train, X_test, y_test, n_classes)
    if r: results.append(r)

    # 2. XGBoost
    r = train_xgboost(X_train, y_train, X_test, y_test, n_classes)
    if r: results.append(r)

    # 3. Vanilla MLP
    r = train_vanilla_mlp(X_train, y_train, X_test, y_test, n_classes)
    if r: results.append(r)

    # 4. CyberDetect-MLP (train with full pipeline)
    print("\n[INFO] Dang huan luyen CyberDetect-MLP (proposed)...")
    from model.trainer import train_model
    model, le2, history = train_model(train_csv, None, os.path.join(output_dir, "cyberdetect_model"))
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    r = _compute_metrics(y_test, y_pred, y_prob, n_classes, "CyberDetect-MLP (proposed)")
    r['y_pred'] = y_pred
    results.append(r)

    # Save Table 3
    results_for_csv = [{k: v for k, v in r.items() if k != 'y_pred'} for r in results]
    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(os.path.join(output_dir, "table3_comparison.csv"), index=False)
    print(f"\n[OK] Table 3 da duoc luu tai {output_dir}/table3_comparison.csv")
    print(results_df.to_string(index=False))

    # Generate Fig 7 — Confusion Matrices
    generate_confusion_matrices(results, y_test, [str(c) for c in le.classes_], output_dir)

    # Generate Fig 8 — Bar Chart
    generate_comparison_plot(results_df, output_dir)

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline comparison (Paper Table 3, Fig 8)")
    parser.add_argument("train_csv", help="Training CSV")
    parser.add_argument("test_csv", help="Test CSV")
    parser.add_argument("--output_dir", default="results/baselines")
    args = parser.parse_args()
    run_baseline_comparison(args.train_csv, args.test_csv, args.output_dir)
