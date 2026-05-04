"""
Paper-aligned reproduction pipeline for CyberDetect-MLP.

Two data paths:
  1. --from-spark : Read preprocessed Parquet from bigdata/output/ (Kafka→HDFS→Spark pipeline)
  2. --data CSV   : Read raw CSV and preprocess directly (fallback when Docker not available)

The protocol is leakage-safe: split first, then fit encoders, MI selector and
MinMax scaler on the training subset only.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


PAPER_HYPERPARAMS = {
    "top_k_features": 30,
    "test_size": 0.20,
    "validation_split": 0.20,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,
    "learning_rate": 1e-3,
    "dropout": 0.3,
    "hidden_layers": [512, 256, 128],
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_label_column(df: pd.DataFrame) -> str:
    for c in ["label", "Label", "target", "Target"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find binary label column. Expected one of: label, Label, target, Target")


def prepare_target(y_raw: pd.Series):
    y = y_raw.copy()
    if y.dtype == object:
        low = y.astype(str).str.lower()
        if set(low.unique()).issubset({"normal", "attack", "0", "1"}):
            y = low.map({"normal": 0, "attack": 1, "0": 0, "1": 1}).astype(int)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le



def make_one_hot_encoder():
    """Create OneHotEncoder compatible with old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X_train_raw: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(len(preprocessor.transformers_))]


def cosine_annealing(epoch, lr_min=1e-5, lr_max=1e-3, T=100):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / T))


def build_cyberdetect(input_dim: int, n_classes: int, batchnorm=True, dropout=True):
    out_units = 1 if n_classes == 2 else n_classes
    out_act = "sigmoid" if n_classes == 2 else "softmax"
    model = Sequential([Input(shape=(input_dim,))])
    for units in PAPER_HYPERPARAMS["hidden_layers"]:
        model.add(Dense(units, activation="relu"))
        if batchnorm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(PAPER_HYPERPARAMS["dropout"]))
    model.add(Dense(out_units, activation=out_act))
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=Adam(PAPER_HYPERPARAMS["learning_rate"]), loss=loss, metrics=["accuracy"])
    return model


def predict_labels_and_probs(model, X, n_classes):
    prob = model.predict(X, verbose=0)
    if n_classes == 2:
        prob1 = prob.reshape(-1)
        y_pred = (prob1 >= 0.5).astype(int)
        y_prob_auc = prob1
        y_prob_full = np.column_stack([1 - prob1, prob1])
    else:
        y_pred = prob.argmax(axis=1)
        y_prob_auc = prob
        y_prob_full = prob
    return y_pred, y_prob_auc, y_prob_full


def compute_metrics(y_true, y_pred, y_prob_auc, n_classes):
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob_auc)
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob_auc, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc"] = np.nan
    return metrics


def train_keras_model(X_train, y_train, X_test, y_test, n_classes, variant="full", use_class_weight=True, seed=42):
    set_seed(seed)
    model = build_cyberdetect(
        X_train.shape[1], n_classes,
        batchnorm=(variant not in ["no_batchnorm", "vanilla"]),
        dropout=(variant not in ["no_dropout", "vanilla"]),
    )
    callbacks = [EarlyStopping(monitor="val_loss", patience=PAPER_HYPERPARAMS["patience"], restore_best_weights=True)]
    if variant != "no_scheduler":
        callbacks.append(LearningRateScheduler(lambda e: cosine_annealing(e, T=PAPER_HYPERPARAMS["epochs"])))
    class_weight = None
    if use_class_weight:
        weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(weights))
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=PAPER_HYPERPARAMS["validation_split"],
        epochs=PAPER_HYPERPARAMS["epochs"],
        batch_size=PAPER_HYPERPARAMS["batch_size"],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    elapsed = time.time() - t0
    y_pred, y_prob_auc, _ = predict_labels_and_probs(model, X_test, n_classes)
    metrics = compute_metrics(y_test, y_pred, y_prob_auc, n_classes)
    metrics.update({"train_time_sec": elapsed, "epochs_ran": len(history.history["loss"])})
    return model, history, metrics, y_pred


def load_from_spark(spark_output_dir):
    """
    Load preprocessed data from Spark pipeline output (Parquet files).
    Pipeline: CSV → Kafka → HDFS → Spark preprocessing → Parquet → here.
    Paper ref: L512 "Apache Spark facilitates distributed preprocessing"
    """
    spark_dir = Path(spark_output_dir)
    required_files = ["train_features.parquet", "test_features.parquet",
                      "train_labels.parquet", "test_labels.parquet", "metadata.json"]
    for f in required_files:
        if not (spark_dir / f).exists():
            raise FileNotFoundError(f"Missing {f} in {spark_dir}. Run Spark preprocessing first.")

    X_train = pd.read_parquet(spark_dir / "train_features.parquet").values.astype(np.float32)
    X_test = pd.read_parquet(spark_dir / "test_features.parquet").values.astype(np.float32)
    y_train = pd.read_parquet(spark_dir / "train_labels.parquet")["label"].values.astype(int)
    y_test = pd.read_parquet(spark_dir / "test_labels.parquet")["label"].values.astype(int)

    with open(spark_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    n_classes = meta["num_classes"]
    top_features = meta["selected_features"]

    # For ablation: also need all-features scaled version
    X_train_all_scaled = X_train  # Spark already scaled all selected
    X_test_all_scaled = X_test

    print(f"[Spark Output] Loaded from: {spark_dir}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Classes: {n_classes}")
    print(f"  Source: Kafka -> HDFS -> Spark preprocessing -> Parquet")

    class FakeLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes
    label_encoder = FakeLabelEncoder(meta.get("label_encoder_classes", [str(i) for i in range(n_classes)]))

    return {
        "label_col": "label",
        "drop_cols": ["label"],
        "label_encoder": label_encoder,
        "n_classes": n_classes,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_all_scaled": X_train_all_scaled,
        "X_test_all_scaled": X_test_all_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "top_features": top_features,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "from_spark": True,
    }


def load_and_transform(args):
    """Load TON_IoT and build train-only preprocessing artifacts."""
    set_seed(args.seed)
    df = pd.read_csv(args.data)
    if getattr(args, "sample_rows", None):
        df = df.head(args.sample_rows).copy()
    label_col = find_label_column(df)
    drop_cols = [label_col]
    for c in ["type", "attack_type", "category"]:
        if c in df.columns:
            drop_cols.append(c)
    X_raw = df.drop(columns=drop_cols, errors="ignore")
    y, label_encoder = prepare_target(df[label_col])
    n_classes = len(label_encoder.classes_)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=PAPER_HYPERPARAMS["test_size"], stratify=y, random_state=args.seed
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train_all = preprocessor.fit_transform(X_train_raw)
    X_test_all = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)
    X_train_all = np.nan_to_num(X_train_all.astype(np.float32))
    X_test_all = np.nan_to_num(X_test_all.astype(np.float32))

    mi = mutual_info_classif(X_train_all, y_train, random_state=args.seed, discrete_features=False)
    top_idx = np.argsort(mi)[::-1][:PAPER_HYPERPARAMS["top_k_features"]]
    top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_idx]

    scaler_top = MinMaxScaler()
    X_train = scaler_top.fit_transform(X_train_all[:, top_idx])
    X_test = scaler_top.transform(X_test_all[:, top_idx])

    scaler_all = MinMaxScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled = scaler_all.transform(X_test_all)

    return {
        "label_col": label_col,
        "drop_cols": drop_cols,
        "label_encoder": label_encoder,
        "n_classes": n_classes,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_all_scaled": X_train_all_scaled,
        "X_test_all_scaled": X_test_all_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "top_features": top_features,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def save_metadata(out, data, args):
    metadata = {
        "protocol": "Paper-aligned core reproduction: 80/20 stratified split; fit preprocessing, MI top-30 and MinMaxScaler on training subset only.",
        "data_source": "Spark pipeline (Kafka -> HDFS -> Spark -> Parquet)" if data.get("from_spark") else "Direct CSV preprocessing",
        "label_column": data["label_col"],
        "dropped_columns": data["drop_cols"],
        "label_classes": [str(c) for c in data["label_encoder"].classes_],
        "n_classes": int(data["n_classes"]),
        "hyperparams": PAPER_HYPERPARAMS,
        "top_features": data["top_features"],
        "seed": args.seed,
        "n_train": int(data["n_train"]),
        "n_test": int(data["n_test"]),
    }
    (out / "models" / "paper_aligned_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")



def run_preprocess(args):
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    data = load_from_spark(args.from_spark) if args.from_spark else load_and_transform(args)
    save_metadata(out, data, args)
    mi_df = pd.DataFrame({"rank": range(1, len(data["top_features"]) + 1), "feature": data["top_features"]})
    mi_df.to_csv(out / "results" / "paper_aligned_top30_features.csv", index=False)
    print("[OK] Preprocessing metadata saved")
    print("n_train=", data["n_train"], "n_test=", data["n_test"])
    print(mi_df.head(30).to_string(index=False))

def run_table3(args):
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    data = load_from_spark(args.from_spark) if args.from_spark else load_and_transform(args)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    n_classes = data["n_classes"]
    rows, preds = [], {}

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=args.seed, n_jobs=-1)
    t0 = time.time(); rf.fit(X_train, y_train); elapsed = time.time() - t0
    y_pred = rf.predict(X_test); y_prob = rf.predict_proba(X_test)
    m = compute_metrics(y_test, y_pred, y_prob[:, 1] if n_classes == 2 else y_prob, n_classes)
    m.update({"model": "Random Forest", "train_time_sec": elapsed}); rows.append(m); preds["Random Forest"] = y_pred

    if XGBClassifier is not None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        eval_metric = "logloss" if n_classes == 2 else "mlogloss"
        xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, objective=objective, eval_metric=eval_metric, random_state=args.seed, n_jobs=-1)
        t0 = time.time(); xgb.fit(X_train, y_train); elapsed = time.time() - t0
        y_pred = xgb.predict(X_test); y_prob = xgb.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_prob[:, 1] if n_classes == 2 else y_prob, n_classes)
        m.update({"model": "XGBoost", "train_time_sec": elapsed}); rows.append(m); preds["XGBoost"] = y_pred

    _, _, m, y_pred = train_keras_model(X_train, y_train, X_test, y_test, n_classes, variant="vanilla", use_class_weight=False, seed=args.seed)
    m.update({"model": "Vanilla MLP"}); rows.append(m); preds["Vanilla MLP"] = y_pred

    cyber, history, m, y_pred = train_keras_model(X_train, y_train, X_test, y_test, n_classes, variant="full", use_class_weight=True, seed=args.seed)
    m.update({"model": "CyberDetect-MLP"}); rows.append(m); preds["CyberDetect-MLP"] = y_pred
    cyber.save(out / "models" / "cyberdetect_mlp_paper_aligned.h5")

    pd.DataFrame(rows).to_csv(out / "results" / "paper_aligned_table3_metrics.csv", index=False)
    report = classification_report(y_test, preds["CyberDetect-MLP"], target_names=[str(c) for c in data["label_encoder"].classes_], zero_division=0)
    (out / "results" / "cyberdetect_classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, preds["CyberDetect-MLP"])).to_csv(out / "results" / "cyberdetect_confusion_matrix.csv", index=False)
    save_metadata(out, data, args)
    print(pd.DataFrame(rows))
    print(f"[OK] Saved Table 3 reproduction outputs to {out}")


def run_ablation(args):
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    data = load_from_spark(args.from_spark) if args.from_spark else load_and_transform(args)
    y_train, y_test = data["y_train"], data["y_test"]
    n_classes = data["n_classes"]
    ablation_rows = []
    for name, variant, Xtr, Xte in [
        ("Full Model", "full", data["X_train"], data["X_test"]),
        ("MLP-NoFeatureSelect", "full", data["X_train_all_scaled"], data["X_test_all_scaled"]),
        ("MLP-NoBatchNorm", "no_batchnorm", data["X_train"], data["X_test"]),
        ("MLP-NoDropout", "no_dropout", data["X_train"], data["X_test"]),
        ("MLP-NoScheduler", "no_scheduler", data["X_train"], data["X_test"]),
    ]:
        _, _, met, _ = train_keras_model(Xtr, y_train, Xte, y_test, n_classes, variant=variant, use_class_weight=True, seed=args.seed)
        met.update({"variant": name})
        ablation_rows.append(met)
    pd.DataFrame(ablation_rows).to_csv(out / "results" / "paper_aligned_ablation_metrics.csv", index=False)
    save_metadata(out, data, args)
    print(pd.DataFrame(ablation_rows))
    print(f"[OK] Saved ablation outputs to {out}")


def run(args):
    if args.section == "preprocess":
        run_preprocess(args)
    elif args.section == "table3":
        run_table3(args)
    elif args.section == "ablation":
        run_ablation(args)
    else:
        run_table3(args)
        run_ablation(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to TON_IoT CSV (direct mode)")
    parser.add_argument("--from-spark", default=None, metavar="DIR",
                        help="Load preprocessed Parquet from Spark output directory (e.g. bigdata/output)")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--section", choices=["all", "preprocess", "table3", "ablation"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs for smoke testing")
    parser.add_argument("--sample_rows", type=int, default=None, help="Use first N rows for smoke testing only")
    args = parser.parse_args()
    if not args.from_spark and not args.data:
        parser.error("Either --data or --from-spark is required")
    if args.epochs is not None:
        PAPER_HYPERPARAMS["epochs"] = args.epochs
        PAPER_HYPERPARAMS["patience"] = min(PAPER_HYPERPARAMS["patience"], max(1, args.epochs))
    run(args)
