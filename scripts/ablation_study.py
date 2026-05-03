"""
Ablation Study — Table 5 (line 1180-1186), Fig 10.

Paper reference:
  - Line 1162-1166: "ablation study by incrementally disabling or replacing critical modules"
  - Table 5 (line 1188): "Ablation study on CyberDetect-MLP components"
  - Fig 10 (line 1192): "Performance analysis of CyberDetect-MLP ablated variants"

6 Variants:
  1. Full Model (all components)
  2. MLP-NoFeatureSelect (skip MI feature selection, use all features)
  3. MLP-NoBatchNorm (remove BatchNormalization layers)
  4. MLP-NoDropout (remove Dropout layers)
  5. MLP-NoScheduler (constant LR, no cosine annealing)
  6. MLP-RawFeatures (no preprocessing or normalization)
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import settings


def cosine_annealing(epoch, lr_min=1e-5, lr_max=1e-3, T=100):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / T))


def _build_variant(input_dim, num_classes, variant):
    """Build model variant based on ablation type."""
    if variant == "full":
        # Full CyberDetect-MLP
        return Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.3),
            Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif variant == "no_batchnorm":
        # MLP-NoBatchNorm (Table 5 line 1183)
        return Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(256, activation='relu'), Dropout(0.3),
            Dense(128, activation='relu'), Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif variant == "no_dropout":
        # MLP-NoDropout (Table 5 line 1184)
        return Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dense(256, activation='relu'), BatchNormalization(),
            Dense(128, activation='relu'), BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
    elif variant == "no_feature_select":
        # Same architecture, but trained on ALL features (no MI selection)
        return Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.3),
            Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif variant == "raw_features":
        # Same as no_feature_select but expects unnormalized input
        return Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), Dropout(0.3),
            Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _train_and_evaluate(X_train, y_train, X_test, y_test, n_classes,
                        variant, use_scheduler=True):
    """Train a single variant and return metrics."""
    model = _build_variant(X_train.shape[1], n_classes, variant)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(weights))

    callbacks = [EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)]
    if use_scheduler:
        callbacks.append(LearningRateScheduler(lambda e: cosine_annealing(e)))

    model.fit(X_train, y_train, epochs=100, batch_size=64,
              validation_split=0.2, callbacks=callbacks,
              class_weight=cw, verbose=0)

    y_pred = model.predict(X_test).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    tf.keras.backend.clear_session()
    return acc, prec, rec, f1


def run_ablation_study(train_csv, test_csv, all_features_train_csv=None,
                       raw_train_csv=None, output_dir="results/ablation"):
    """
    Full ablation study — Paper Table 5, Fig 10.

    Args:
        train_csv: Training CSV with selected features (top-30)
        test_csv: Test CSV with selected features
        all_features_train_csv: Training CSV with ALL features (for NoFeatureSelect)
        raw_train_csv: Raw unnormalized training CSV (for RawFeatures)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    le = LabelEncoder()
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop('label', axis=1).values
    y_train = le.fit_transform(train_df['label'].values)
    X_test = test_df.drop('label', axis=1).values
    y_test = le.transform(test_df['label'].values)
    n_classes = len(le.classes_)

    variants = [
        ("CyberDetect-MLP (Full Model)", "full", X_train, X_test, True),
        ("MLP-NoBatchNorm", "no_batchnorm", X_train, X_test, True),
        ("MLP-NoDropout", "no_dropout", X_train, X_test, True),
        ("MLP-NoScheduler", "full", X_train, X_test, False),  # Same arch, no scheduler
    ]

    # Add NoFeatureSelect variant if all-features data provided
    if all_features_train_csv and os.path.exists(all_features_train_csv):
        all_train = pd.read_csv(all_features_train_csv)
        all_test_path = all_features_train_csv.replace("train", "test")
        if os.path.exists(all_test_path):
            all_test = pd.read_csv(all_test_path)
            X_all_train = all_train.drop('label', axis=1).values
            X_all_test = all_test.drop('label', axis=1).values
            variants.insert(1, ("MLP-NoFeatureSelect", "no_feature_select",
                                X_all_train, X_all_test, True))

    # Add RawFeatures variant if raw data provided
    if raw_train_csv and os.path.exists(raw_train_csv):
        raw_train = pd.read_csv(raw_train_csv)
        raw_test_path = raw_train_csv.replace("train", "test")
        if os.path.exists(raw_test_path):
            raw_test = pd.read_csv(raw_test_path)
            X_raw_train = raw_train.drop('label', axis=1).values
            X_raw_test = raw_test.drop('label', axis=1).values
            # Handle NaN/Inf in raw data
            X_raw_train = np.nan_to_num(X_raw_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_raw_test = np.nan_to_num(X_raw_test, nan=0.0, posinf=0.0, neginf=0.0)
            variants.append(("MLP-RawFeatures", "raw_features",
                              X_raw_train, X_raw_test, True))

    results = []
    for name, variant, X_tr, X_te, use_sched in variants:
        print(f"\n{'-'*50}")
        print(f"[INFO] Ablation: {name}")
        print(f"{'-'*50}")
        acc, prec, rec, f1 = _train_and_evaluate(
            X_tr, y_train, X_te, y_test, n_classes, variant, use_sched
        )
        results.append({
            'Model Variant': name,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
        })
        print(f"  [OK] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    # Save Table 5
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "table5_ablation.csv"), index=False)
    print(f"\n[OK] Table 5 da duoc luu tai {output_dir}/table5_ablation.csv")
    print(results_df.to_string(index=False))

    # Generate Fig 10 — grouped bar chart
    _generate_fig10(results_df, output_dir)

    return results_df


def _generate_fig10(df, output_dir):
    """Paper Fig 10: 4-panel ablation performance chart."""
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
    titles = ['(a) Accuracy', '(b) Precision', '(c) Recall', '(d) F1-Score']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = sns.color_palette("Set2", len(df))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2][idx % 2]
        bars = ax.bar(range(len(df)), df[metric].values, color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Model Variant'].values, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(90, 100)

    plt.suptitle('Ablation Study — CyberDetect-MLP Components (Table 5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig10_ablation.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 10 da duoc luu tai {output_dir}/fig10_ablation.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study (Paper Table 5, Fig 10)")
    parser.add_argument("train_csv", help="Training CSV (selected features)")
    parser.add_argument("test_csv", help="Test CSV (selected features)")
    parser.add_argument("--all_features_train", default=None, help="Training CSV with ALL features")
    parser.add_argument("--raw_train", default=None, help="Raw unnormalized training CSV")
    parser.add_argument("--output_dir", default="results/ablation")
    args = parser.parse_args()
    run_ablation_study(args.train_csv, args.test_csv,
                       args.all_features_train, args.raw_train, args.output_dir)
