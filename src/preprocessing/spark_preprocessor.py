from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, count, when
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer
from pyspark.ml.functions import vector_to_array
import pandas as pd, os

def init_spark(app_name):
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()

def preprocess_data(input_csv, output_path):
    spark = init_spark("CyberDetect-MLP-Preprocessing")
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    
    # Handle missing (paper line 547: mean imputation for continuous, mode imputation for categorical)
    for c in df.columns:
        missing_count = df.select(count(when(col(c).isNull(), c)).alias('missing')).collect()[0][0]
        if missing_count > 0:
            if df.select(c).dtypes[0][1] in ['double', 'int']:
                df = df.na.fill({c: df.select(mean(c)).first()[0]})
            else:
                mode_val = df.groupBy(c).count().orderBy('count', ascending=False).first()[0]
                df = df.na.fill({c: mode_val})
                
    # Drop duplicates (paper line 549)
    df = df.dropDuplicates()
    
    # Drop 'type' column to prevent data leakage when predicting 'label'
    if 'type' in df.columns:
        df = df.drop('type')
    
    # Encode categorical (Label Encoding)
    cat_cols = [c for c, t in df.dtypes if t == 'string' and c != 'label']
    for c in cat_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep").fit(df)
        df = indexer.transform(df)
        df = df.drop(c) # drop original string column
        
    # Scale numeric
    num_cols = [c for c, t in df.dtypes if t in ['double','int'] and c != 'label']
    if num_cols:
        assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features", handleInvalid="keep")
        df = assembler.transform(df)
        scaler = MinMaxScaler(inputCol="num_features", outputCol="scaled_features")
        df = scaler.fit(df).transform(df)
        
        # Flatten scaled vectors back to scalar columns
        df = df.withColumn("scaled_arr", vector_to_array("scaled_features"))
        for i, c in enumerate(num_cols):
            df = df.withColumn(c, col("scaled_arr")[i])
            
        df = df.drop("num_features", "scaled_features", "scaled_arr")
        
    # Clean up any remaining string columns (just in case)
    final_cols = [c for c, t in df.dtypes if t != 'string']
    df = df.select(final_cols)
    
    pandas_df = df.toPandas()
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "preprocessed.csv")
    pandas_df.to_csv(out_file, index=False)
    print(f"[OK] Du lieu da tien xu ly duoc luu tai {out_file}")
    spark.stop()
    return out_file
