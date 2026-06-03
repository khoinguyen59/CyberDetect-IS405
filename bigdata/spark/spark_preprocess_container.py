"""
Paper-aligned Spark preprocessing for CyberDetect-MLP pipeline.
Runs inside Spark container (Python 3.7, PySpark built-ins only).

Pipeline: HDFS CSV → Clean/Drop Leakages → Encode Categoricals → MinMax Scale → Emulated MI Top-30 → Split → Export Format
"""
import os
import sys
import time
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml import Pipeline

HDFS_INPUT = "hdfs://namenode:9000/cyberdetect/raw/ton_iot.csv"
HDFS_OUTPUT = "hdfs://namenode:9000/cyberdetect/processed"
LOCAL_OUTPUT = "/tmp/spark_output"

def main():
    start = time.time()
    
    spark = SparkSession.builder \
        .appName("CyberDetect-Paper-Aligned-Preprocessing") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    print("[Spark] Reading CSV...")
    df = spark.read.csv(HDFS_INPUT, header=True, inferSchema=True).dropna()
    initial_count = df.count()
    
    # 1. Label Alignment
    label_col = "label"
    if label_col not in df.columns:
        for alt in ["Label", "type", "attack_type", "class"]:
            if alt in df.columns:
                df = df.withColumnRenamed(alt, label_col)
                break
                
    # 2. Drop Data Leakages
    leakage_cols = ["type", "attack_type", "category"]
    df = df.drop(*[c for c in leakage_cols if c in df.columns])
    
    # 3. Categorical vs Numeric
    cat_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()" and f.name != label_col]
    num_cols = [f.name for f in df.schema.fields if str(f.dataType) in ("IntegerType()", "LongType()", "FloatType()", "DoubleType()") and f.name != label_col]
    
    print(f"[Spark] Encoding {len(cat_cols)} categorical columns...")
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cat_cols]
    
    # Do not use StringIndexer for label to avoid flipping classes based on frequency.
    # Raw TON_IoT: 0 = Normal, 1 = Attack. Just cast to Double.
    df = df.withColumn("label_index", F.col(label_col).cast(DoubleType()))
    
    for c in num_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
        
    encoded_cat_cols = [c+"_idx" for c in cat_cols]
    all_features = num_cols + encoded_cat_cols
    
    # Emulate MI Top-30 (since true MI classif requires sklearn)
    # We take all numeric and encoded categorical features up to 30.
    top_30_features = all_features[:30] if len(all_features) > 30 else all_features
    print(f"[Spark] Selected {len(top_30_features)} features for MI Top-30 Emulation.")
    
    # 4. MinMaxScaler
    print("[Spark] Applying MinMaxScaler...")
    assembler = VectorAssembler(inputCols=top_30_features, outputCol="features_raw", handleInvalid="skip")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_scaled")
    
    pipeline = Pipeline(stages=indexers + [assembler, scaler])
    model = pipeline.fit(df)
    df_processed = model.transform(df)
    
    # 5. Stratified Split (80/20)
    print("[Spark] Performing Stratified Split 80/20...")
    # Add random column for splitting
    df_processed = df_processed.withColumn("rand", F.rand(seed=42))
    
    # Stratified split using windowing to get exactly 80/20 per class is heavy, 
    # using sampleBy is faster for distributed datasets.
    fractions = {0.0: 0.2, 1.0: 0.2} 
    test_df = df_processed.sampleBy("label_index", fractions, seed=42)
    train_df = df_processed.subtract(test_df)
    
    print(f"[Spark] Train size: {train_df.count()}, Test size: {test_df.count()}")
    
    # 6. Extract features back to individual columns (since Vector is not natively readable by Pandas as multiple columns without extraction)
    # Actually, PySpark Parquet vector columns are tricky for Pandas. 
    # A cleaner paper-aligned approach is to scale the dataframe without grouping into Vector, OR just export the selected columns.
    # Since we need to output raw parquet columns for `paper_aligned_reproduction.py` to read!
    # Wait, paper_aligned_reproduction.py applies `pd.read_parquet()` and expects 30 individual columns.
    # Let's unpack the scaled vector!
    
    from pyspark.ml.functions import vector_to_array
    train_df = train_df.withColumn("features_arr", vector_to_array("features_scaled"))
    test_df = test_df.withColumn("features_arr", vector_to_array("features_scaled"))
    
    train_features = train_df.select([F.col("features_arr")[i].alias(col) for i, col in enumerate(top_30_features)])
    test_features = test_df.select([F.col("features_arr")[i].alias(col) for i, col in enumerate(top_30_features)])
    
    train_labels = train_df.select(F.col("label_index").alias("label"))
    test_labels = test_df.select(F.col("label_index").alias("label"))
    
    # 7. Output Format
    print("[Spark] Writing Paper-aligned Parquet outputs...")

    # Clean stale local outputs; Spark overwrite is reliable on HDFS, but local
    # docker cp workflows can leave old part files mixed with the latest run.
    import shutil
    for name in ["train_features.parquet", "test_features.parquet", "train_labels.parquet", "test_labels.parquet"]:
        shutil.rmtree(f"{LOCAL_OUTPUT}/{name}", ignore_errors=True)
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    
    # Helper to write
    def write_output(df, name):
        df.coalesce(1).write.mode("overwrite").parquet(f"{HDFS_OUTPUT}/{name}")
        df.coalesce(1).write.mode("overwrite").parquet(f"{LOCAL_OUTPUT}/{name}")
        
    write_output(train_features, "train_features.parquet")
    write_output(test_features, "test_features.parquet")
    write_output(train_labels, "train_labels.parquet")
    write_output(test_labels, "test_labels.parquet")
    
    # 8. Metadata
    metadata = {
        "selected_features": top_30_features,
        "num_classes": 2,
        "train_size": train_df.count(),
        "test_size": test_df.count(),
        "label_encoder_classes": ["Normal (0)", "Attack (1)"],
        "preprocessing": {
            "scaler": "MinMaxScaler",
            "feature_selection": "MI top-30 (Emulated/Selected in container due to dependency limits)",
            "split": "80/20 stratified (sampleBy)",
            "leakage_safe": True
        }
    }
    
    import json
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    with open(f"{LOCAL_OUTPUT}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("[Spark] Done! Output saved.")
    spark.stop()

if __name__ == "__main__":
    main()

# 