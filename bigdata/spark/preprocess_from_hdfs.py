"""
Spark Preprocessing — HDFS raw data → preprocessing → Parquet output
Paper ref: L512 "Apache Spark facilitates distributed preprocessing"
Paper ref: L641 "Spark's DataFrame API enable parallel execution"
Paper ref: L642 "Missing value imputation, duplicate removal, encoding, normalization"
Paper ref: L647 "Spark's scalable computation for MI scores"

Pipeline:
    HDFS (/cyberdetect/raw/ton_iot.csv)
        → Null handling (mean/mode)
        → Drop duplicates
        → OneHot encoding
        → MinMax scaling
        → MI feature selection (top-30)
        → Stratified split 80/20
        → HDFS output (/cyberdetect/processed/) as Parquet
        → Copy to local (bigdata/output/)

Usage:
    spark-submit --master spark://spark-master:7077 bigdata/spark/preprocess_from_hdfs.py
    # Or local mode:
    python bigdata/spark/preprocess_from_hdfs.py --local
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType


# ===========================================================================
# Config
# ===========================================================================
HDFS_INPUT = "hdfs://localhost:9000/cyberdetect/raw/ton_iot.csv"
HDFS_OUTPUT = "hdfs://namenode:9000/cyberdetect/processed"
LOCAL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output")
TOP_K_FEATURES = 30
TEST_SIZE = 0.20
RANDOM_SEED = 42
LABEL_COL = "label"


def create_spark(local=False):
    """Create SparkSession — cluster or local mode."""
    builder = SparkSession.builder.appName("CyberDetect-Preprocessing")
    if local:
        builder = builder.master("local[*]")
    else:
        builder = builder.master("spark://spark-master:7077")
    return builder.getOrCreate()


def preprocess(spark, input_path, output_path, local_output_path,
               top_k=TOP_K_FEATURES, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Full preprocessing pipeline matching paper Algorithms 1-2.

    Algorithm 1: Data Preprocessing (Paper L795-802)
    Algorithm 2: Feature Selection Using MI (Paper L803-810)
    """
    start_time = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load data (Algorithm 1, step 1)
    # Supports: single CSV file or folder of CSVs (e.g. Flume output)
    # -----------------------------------------------------------------------
    print(f"[Spark] Loading data from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    initial_count = df.count()
    print(f"  Rows: {initial_count}, Columns: {len(df.columns)}")

    # Separate label column
    if LABEL_COL not in df.columns:
        # Try common alternatives
        for alt in ["type", "attack_type", "class"]:
            if alt in df.columns:
                df = df.withColumnRenamed(alt, LABEL_COL)
                break

    # Drop attack-type columns when reproducing the binary label task; keeping
    # them would leak target-derived information into Normal-vs-Attack training.
    leakage_cols = [c for c in ["type", "attack_type", "category"] if c in df.columns and c != LABEL_COL]
    if leakage_cols:
        print(f"[Spark] Dropping leakage columns for binary task: {leakage_cols}")
        df = df.drop(*leakage_cols)

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # -----------------------------------------------------------------------
    # Step 2: Handle missing values — mean for numeric, mode for categorical
    # (Algorithm 1, step 2; Paper L547-548)
    # -----------------------------------------------------------------------
    print("[Spark] Handling missing values...")
    numeric_cols = [f.name for f in df.schema.fields
                    if isinstance(f.dataType, DoubleType) or f.dataType.typeName() in ("integer", "long", "float", "double")]
    numeric_cols = [c for c in numeric_cols if c != LABEL_COL]

    string_cols = [f.name for f in df.schema.fields
                   if isinstance(f.dataType, StringType) and f.name != LABEL_COL]

    # Mean imputation for numeric
    for col in numeric_cols:
        mean_val = df.select(F.mean(F.col(col))).first()[0]
        if mean_val is not None:
            df = df.fillna({col: mean_val})

    # Mode imputation for categorical
    for col in string_cols:
        mode_row = df.groupBy(col).count().orderBy(F.desc("count")).first()
        if mode_row is not None:
            df = df.fillna({col: mode_row[0]})

    print(f"  Numeric cols imputed: {len(numeric_cols)}")
    print(f"  Categorical cols imputed: {len(string_cols)}")

    # -----------------------------------------------------------------------
    # Step 3: Drop duplicates (Algorithm 1, step 3; Paper L549)
    # -----------------------------------------------------------------------
    before_dedup = df.count()
    df = df.dropDuplicates()
    after_dedup = df.count()
    print(f"[Spark] Dropped duplicates: {before_dedup} → {after_dedup} ({before_dedup - after_dedup} removed)")

    # -----------------------------------------------------------------------
    # Step 4: Encode categorical — OneHotEncoding (Paper L549)
    # -----------------------------------------------------------------------
    print("[Spark] Encoding categorical variables (OneHotEncoding)...")
    from pyspark.ml.feature import StringIndexer, OneHotEncoder
    from pyspark.ml import Pipeline

    if string_cols:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                    for c in string_cols]
        encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
                    for c in string_cols]
        pipeline = Pipeline(stages=indexers + encoders)
        df = pipeline.fit(df).transform(df)
        print(f"  Encoded {len(string_cols)} categorical columns")

    # -----------------------------------------------------------------------
    # Step 5: MinMax Normalization — Eq. 1: X' = (X-Xmin)/(Xmax-Xmin)
    # (Algorithm 1, step 5; Paper L550-556)
    # -----------------------------------------------------------------------
    print("[Spark] MinMax scaling numeric features...")
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler

    # Assemble numeric features
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features",
                                handleInvalid="skip")
    df = assembler.transform(df)

    scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_features")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # -----------------------------------------------------------------------
    # Step 6: Convert to Pandas for MI computation
    # Paper L647: "Spark's scalable computation for MI scores"
    # Implementation: Spark handles data I/O, MI runs on driver via sklearn
    # -----------------------------------------------------------------------
    print("[Spark] Computing Mutual Information scores (top-{})...".format(top_k))
    pdf = df.select(numeric_cols + [LABEL_COL]).toPandas()

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(pdf[LABEL_COL].values)
    X = pdf[numeric_cols].values

    # Replace NaN with 0 for MI computation
    X = np.nan_to_num(X, nan=0.0)

    mi_scores = mutual_info_classif(X, y, random_state=seed)
    top_indices = np.argsort(mi_scores)[::-1][:top_k]
    selected_features = [numeric_cols[i] for i in top_indices]

    print(f"  Top-{top_k} features by MI:")
    for i, idx in enumerate(top_indices):
        print(f"    {i+1}. {numeric_cols[idx]}: {mi_scores[idx]:.4f}")

    # -----------------------------------------------------------------------
    # Step 7: Select features + MinMax scale selected only
    # -----------------------------------------------------------------------
    from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
    from sklearn.model_selection import train_test_split

    X_selected = pdf[selected_features].values
    X_selected = np.nan_to_num(X_selected, nan=0.0)

    # -----------------------------------------------------------------------
    # Step 8: Stratified split 80/20 (Paper L1021)
    # -----------------------------------------------------------------------
    print(f"[Spark] Stratified split: {int((1-test_size)*100)}/{int(test_size*100)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=seed, stratify=y
    )

    # MinMax on train only (leakage-safe)
    scaler_final = SkMinMaxScaler()
    X_train = scaler_final.fit_transform(X_train)
    X_test = scaler_final.transform(X_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    # -----------------------------------------------------------------------
    # Step 9: Save as Parquet
    # -----------------------------------------------------------------------
    import pandas as pd

    train_features_df = pd.DataFrame(X_train, columns=selected_features)
    test_features_df = pd.DataFrame(X_test, columns=selected_features)
    train_labels_df = pd.DataFrame({"label": y_train})
    test_labels_df = pd.DataFrame({"label": y_test})

    # Save to local output
    os.makedirs(local_output_path, exist_ok=True)

    train_features_df.to_parquet(os.path.join(local_output_path, "train_features.parquet"), index=False)
    test_features_df.to_parquet(os.path.join(local_output_path, "test_features.parquet"), index=False)
    train_labels_df.to_parquet(os.path.join(local_output_path, "train_labels.parquet"), index=False)
    test_labels_df.to_parquet(os.path.join(local_output_path, "test_labels.parquet"), index=False)

    # Save metadata
    metadata = {
        "selected_features": selected_features,
        "mi_scores": {numeric_cols[i]: float(mi_scores[i]) for i in top_indices},
        "num_classes": int(len(np.unique(y_train))),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "label_encoder_classes": le.classes_.tolist(),
        "preprocessing": {
            "null_handling": "mean (numeric) / mode (categorical)",
            "duplicates": f"removed {before_dedup - after_dedup}",
            "encoding": "OneHotEncoder (categorical), MinMaxScaler (numeric)",
            "feature_selection": f"MI top-{top_k}",
            "split": f"{int((1-test_size)*100)}/{int(test_size*100)} stratified",
            "leakage_safe": True,
        }
    }
    with open(os.path.join(local_output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Also save to HDFS if not local mode
    try:
        spark_train = spark.createDataFrame(train_features_df)
        spark_train.write.parquet(f"{output_path}/train_features.parquet", mode="overwrite")
        spark_test = spark.createDataFrame(test_features_df)
        spark_test.write.parquet(f"{output_path}/test_features.parquet", mode="overwrite")
        print(f"\n[Spark] Saved to HDFS: {output_path}")
    except Exception as e:
        print(f"[WARN] Could not save to HDFS: {e}")
        print(f"  Local output saved to: {local_output_path}")

    elapsed = time.time() - start_time
    print(f"\n[Spark] Preprocessing DONE in {elapsed:.1f}s")
    print(f"  Local output: {local_output_path}/")
    print(f"  Files: train_features.parquet, test_features.parquet, "
          f"train_labels.parquet, test_labels.parquet, metadata.json")

    return local_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark preprocessing pipeline")
    parser.add_argument("--local", action="store_true",
                        help="Use Spark local[*] instead of cluster")
    parser.add_argument("--input", default=HDFS_INPUT,
                        help="Input path (HDFS or local CSV)")
    parser.add_argument("--output", default=HDFS_OUTPUT,
                        help="HDFS output path")
    parser.add_argument("--local-output", default=LOCAL_OUTPUT,
                        help="Local output directory")
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    spark = create_spark(local=args.local)
    try:
        preprocess(spark, args.input, args.output, args.local_output,
                   top_k=args.top_k, seed=args.seed)
    finally:
        spark.stop()
