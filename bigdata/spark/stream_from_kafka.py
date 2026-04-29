"""
Spark Structured Streaming — Read from Kafka topic, process in real-time.
Paper ref: L511 "ingested using Apache Kafka for real-time streaming"
Paper ref: L634 "Apache Kafka enables real-time data ingestion"

This demonstrates the streaming path of the paper's architecture:
    Kafka topic "toniot-events" → Spark Structured Streaming → console/HDFS

Usage:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
        bigdata/spark/stream_from_kafka.py

    # Or local mode:
    python bigdata/spark/stream_from_kafka.py --local
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType


KAFKA_BOOTSTRAP = "kafka:29092"
KAFKA_TOPIC = "toniot-events"


def create_spark(local=False):
    """Create SparkSession with Kafka support."""
    builder = (SparkSession.builder
               .appName("CyberDetect-KafkaStream")
               .config("spark.jars.packages",
                       "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"))
    if local:
        builder = builder.master("local[*]")
    return builder.getOrCreate()


def stream_from_kafka(spark, bootstrap_servers=KAFKA_BOOTSTRAP,
                      topic=KAFKA_TOPIC, output_mode="console", hdfs_output=None):
    """
    Read from Kafka topic using Spark Structured Streaming.
    Paper Algorithm 4: Real-time Detection Pipeline (L815-821)
    """
    print(f"[Spark Streaming] Reading from Kafka topic: {topic}")
    print(f"[Spark Streaming] Bootstrap servers: {bootstrap_servers}")

    # Read stream from Kafka
    df_stream = (spark.readStream
                 .format("kafka")
                 .option("kafka.bootstrap.servers", bootstrap_servers)
                 .option("subscribe", topic)
                 .option("startingOffsets", "earliest")
                 .load())

    # Parse Kafka value (JSON string → columns)
    df_parsed = (df_stream
                 .selectExpr("CAST(value AS STRING) as json_str",
                             "timestamp as kafka_timestamp")
                 .select(F.from_json(F.col("json_str"), "MAP<STRING, STRING>").alias("data"),
                         F.col("kafka_timestamp"))
                 .select("data.*", "kafka_timestamp"))

    # Add processing metadata
    df_processed = (df_parsed
                    .withColumn("processed_at", F.current_timestamp())
                    .withColumn("batch_id", F.monotonically_increasing_id()))

    if output_mode == "hdfs" and hdfs_output:
        # Write to HDFS as Parquet (Paper: HDFS storage)
        query = (df_processed.writeStream
                 .format("parquet")
                 .option("path", hdfs_output)
                 .option("checkpointLocation", f"{hdfs_output}/_checkpoint")
                 .outputMode("append")
                 .trigger(processingTime="10 seconds")
                 .start())
        print(f"[Spark Streaming] Writing to HDFS: {hdfs_output}")
    else:
        # Write to console for demo
        query = (df_processed.writeStream
                 .format("console")
                 .outputMode("append")
                 .option("truncate", False)
                 .option("numRows", 20)
                 .trigger(processingTime="5 seconds")
                 .start())
        print("[Spark Streaming] Writing to console")

    print("[Spark Streaming] Waiting for data... (Ctrl+C to stop)")
    query.awaitTermination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Streaming from Kafka")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    parser.add_argument("--output", choices=["console", "hdfs"], default="console")
    parser.add_argument("--hdfs-path", default="hdfs://namenode:9000/cyberdetect/stream_output")
    args = parser.parse_args()

    spark = create_spark(local=args.local)
    try:
        stream_from_kafka(spark, args.bootstrap_servers, args.topic,
                          args.output, args.hdfs_path)
    finally:
        spark.stop()
