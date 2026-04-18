"""
CyberDetect-MLP Model Evaluator.

Paper references:
  - Eq. 7-10 (line 896-914): Accuracy, Precision, Recall, F1-Score
  - Line 891: "accuracy, precision, recall, F1-score, and ROC curve"
  - Line 1475: "We report per-class performance metrics"
  - Fig 7 (line 1079-1082): Confusion matrices for 4 models
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import pandas as pd
import numpy as np
import tensorflow as tf
import os


def evaluate(model, test_csv, label_encoder):
    """
    Evaluate model on test set — Paper Eq. 7-10.

    Returns:
        Tuple of (accuracy, precision, recall, f1, roc_auc)
    """
    df = pd.read_csv(test_csv)
    X = df.drop('label', axis=1).values
    y_true = label_encoder.transform(df['label'])

    y_pred_prob = model.predict(X)
    y_pred = y_pred_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    roc = roc_auc_score(
        tf.keras.utils.to_categorical(y_true),
        y_pred_prob, average='macro', multi_class='ovr'
    )

    print(f"[OK] Accuracy={acc:.4f} Precision={prec:.4f} "
          f"Recall={rec:.4f} F1={f1:.4f} ROC-AUC={roc:.4f}")

    return acc, prec, rec, f1, roc


def evaluate_per_class(model, test_csv, label_encoder, output_dir="results"):
    """
    Per-class evaluation — Paper line 1475.

    Paper: "We report per-class performance metrics to instill confidence
    that class imbalance handling is effective."

    Returns:
        classification_report dict, confusion_matrix array
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(test_csv)
    X = df.drop('label', axis=1).values
    y_true = label_encoder.transform(df['label'])

    y_pred_prob = model.predict(X)
    y_pred = y_pred_prob.argmax(axis=1)

    class_names = [str(c) for c in label_encoder.classes_]

    # Per-class classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\n[INFO] Per-class Classification Report:")
    print(report_text)

    # Save to CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(output_dir, "per_class_report.csv")
    report_df.to_csv(report_csv)
    print(f"[OK] Bao cao chi tiet tung lop da duoc luu tai {report_csv}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    print(f"[OK] Ma tran nham lan da duoc luu tai {cm_csv}")

    return report, cm


def plot_confusion_matrix(cm, class_names, model_name, ax=None, output_path=None):
    """
    Plot a single confusion matrix heatmap — used for Fig 7 subfigures.

    Paper Fig 7 (line 1079-1082): "confusion matrices for four models"
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{model_name}', fontweight='bold')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[OK] Ma tran nham lan da duoc luu tai {output_path}")
