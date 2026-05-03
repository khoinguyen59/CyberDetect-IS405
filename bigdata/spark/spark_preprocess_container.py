"""
Spark-only preprocessing for CyberDetect-MLP pipeline.
Runs inside Spark container (Python 3.7, no sklearn needed).

Pipeline: HDFS CSV → Clean → Normalize → Split → Parquet
"""
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml import Pipeline

# Config
HDFS_INPUT = "hdfs://namenode:9000/cyberdetect/raw/ton_iot.csv"
HDFS_OUTPUT = "hdfs://namenode:9000/cyberdetect/processed"
LOCAL_OUTPUT = "/tmp/spark_output"
LABEL_COL = "label"
TEST_SIZE = 0.20
RANDOM_SEED = 42


def main():
    start = time.time()
    
    spark = SparkSession.builder \
        .appName("CyberDetect-Preprocessing") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    # ---------------------------------------------------------------
    # Step 1: Load from HDFS
    # ---------------------------------------------------------------
    print(f"[Spark] Loading from: {HDFS_INPUT}")
    df = spark.read.csv(HDFS_INPUT, header=True, inferSchema=True)
    initial_count = df.count()
    print(f"[Spark] Loaded {initial_count} rows, {len(df.columns)} columns")
    
    # ---------------------------------------------------------------
    # Step 2: Clean - drop nulls, remove non-numeric junk
    # ---------------------------------------------------------------
    df = df.dropna()
    
    # Ensure label column exists
    if LABEL_COL not in df.columns:
        # Try common alternatives
        for alt in ["Label", "type", "attack_type", "class"]:
            if alt in df.columns:
                df = df.withColumnRenamed(alt, LABEL_COL)
                break
    
    print(f"[Spark] After cleaning: {df.count()} rows")
    
    # ---------------------------------------------------------------
    # Step 3: Identify numeric feature columns
    # ---------------------------------------------------------------
    numeric_cols = [
        c for c in df.columns 
        if c != LABEL_COL and str(df.schema[c].dataType) in ("IntegerType()", "LongType()", "FloatType()", "DoubleType()")
    ]
    
    # Cast all numeric cols to double
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    
    print(f"[Spark] Using {len(numeric_cols)} numeric features")
    
    # ---------------------------------------------------------------
    # Step 4: Encode label
    # ---------------------------------------------------------------
    label_indexer = StringIndexer(inputCol=LABEL_COL, outputCol="label_index")
    df = label_indexer.fit(df).transform(df)
    
    # ---------------------------------------------------------------
    # Step 5: Assemble + Normalize features (StandardScaler)
    # ---------------------------------------------------------------
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_raw", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=True, withStd=True)
    
    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df)
    df_processed = model.transform(df)
    
    print(f"[Spark] Preprocessing complete: {df_processed.count()} rows")
    
    # ---------------------------------------------------------------
    # Step 6: Split 80/20
    # ---------------------------------------------------------------
    train_df, test_df = df_processed.randomSplit([1.0 - TEST_SIZE, TEST_SIZE], seed=RANDOM_SEED)
    print(f"[Spark] Train: {train_df.count()}, Test: {test_df.count()}")
    
    # ---------------------------------------------------------------
    # Step 7: Select columns and write Parquet
    # ---------------------------------------------------------------
    # Keep original numeric columns + label for downstream MLP
    output_cols = numeric_cols + [LABEL_COL, "label_index"]
    
    train_out = train_df.select(output_cols)
    test_out = test_df.select(output_cols)
    
    # Write to HDFS
    train_out.write.mode("overwrite").parquet(f"{HDFS_OUTPUT}/train")
    test_out.write.mode("overwrite").parquet(f"{HDFS_OUTPUT}/test")
    print(f"[Spark] Written to HDFS: {HDFS_OUTPUT}/train, {HDFS_OUTPUT}/test")
    
    # Also write locally for extraction
    train_out.coalesce(1).write.mode("overwrite").parquet(f"{LOCAL_OUTPUT}/train")
    test_out.coalesce(1).write.mode("overwrite").parquet(f"{LOCAL_OUTPUT}/test")
    print(f"[Spark] Written locally: {LOCAL_OUTPUT}/train, {LOCAL_OUTPUT}/test")
    
    elapsed = time.time() - start
    print(f"[Spark] Total time: {elapsed:.2f}s")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Input rows:     {initial_count}")
    print(f"Output rows:    {df_processed.count()}")
    print(f"Features:       {len(numeric_cols)}")
    print(f"Train split:    {train_df.count()}")
    print(f"Test split:     {test_df.count()}")
    print(f"HDFS output:    {HDFS_OUTPUT}")
    print(f"Local output:   {LOCAL_OUTPUT}")
    print(f"Time elapsed:   {elapsed:.2f}s")
    print(f"{'='*60}")
    
    spark.stop()


if __name__ == "__main__":
    main()
