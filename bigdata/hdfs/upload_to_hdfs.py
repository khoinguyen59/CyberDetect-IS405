"""
HDFS Upload — Kafka consume → HDFS write + Direct CSV upload
Paper ref: L545 "original TON_IoT data are placed in HDFS"
Paper ref: L629 "HDFS for scalable, fault-tolerant storage"

Two modes:
  1. --mode direct : Upload CSV file directly to HDFS (batch ingestion, like Flume)
  2. --mode kafka  : Consume from Kafka topic and write to HDFS

Usage:
    python bigdata/hdfs/upload_to_hdfs.py --mode direct --csv data/ton_iot.csv
    python bigdata/hdfs/upload_to_hdfs.py --mode kafka
"""

import argparse
import json
import os
import sys
import time

try:
    from hdfs import InsecureClient
except ImportError:
    print("[WARN] hdfs package not installed. Run: pip install hdfs")
    InsecureClient = None

try:
    from kafka import KafkaConsumer
except ImportError:
    print("[WARN] kafka-python not installed. Run: pip install kafka-python")
    KafkaConsumer = None


HDFS_URL = "http://localhost:9870"
HDFS_RAW_PATH = "/cyberdetect/raw/ton_iot.csv"
KAFKA_TOPIC = "toniot-events"


def upload_direct(csv_path, hdfs_url=HDFS_URL, hdfs_path=HDFS_RAW_PATH):
    """
    Direct upload CSV to HDFS.
    Replaces Apache Flume batch ingestion (Paper L546, L636).
    """
    if InsecureClient is None:
        print("[ERROR] hdfs package required. pip install hdfs")
        sys.exit(1)

    client = InsecureClient(hdfs_url, user="root")
    hdfs_dir = os.path.dirname(hdfs_path)

    # Create HDFS directory
    client.makedirs(hdfs_dir)
    print(f"[HDFS] Created directory: {hdfs_dir}")

    # Upload file
    print(f"[HDFS] Uploading {csv_path} → {hdfs_path}")
    start = time.time()
    client.upload(hdfs_path, csv_path, overwrite=True)
    elapsed = time.time() - start

    # Verify
    status = client.status(hdfs_path)
    size_mb = status["length"] / (1024 * 1024)
    print(f"[HDFS] DONE")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Upload time: {elapsed:.1f}s")
    print(f"  HDFS path: {hdfs_path}")

    return hdfs_path


def upload_from_kafka(hdfs_url=HDFS_URL, hdfs_path=HDFS_RAW_PATH,
                      bootstrap_servers="localhost:9092", topic=KAFKA_TOPIC,
                      max_records=None, timeout_ms=10000):
    """
    Consume from Kafka topic and write to HDFS as CSV.
    Paper pipeline: Kafka → HDFS (L511-512)
    """
    if InsecureClient is None or KafkaConsumer is None:
        print("[ERROR] hdfs and kafka-python packages required")
        sys.exit(1)

    client = InsecureClient(hdfs_url, user="root")
    hdfs_dir = os.path.dirname(hdfs_path)
    client.makedirs(hdfs_dir)

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        consumer_timeout_ms=timeout_ms,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"[Kafka→HDFS] Consuming from topic: {topic}")
    print(f"[Kafka→HDFS] Writing to: {hdfs_path}")

    records = []
    count = 0
    start = time.time()
    header = None

    for message in consumer:
        row = message.value
        if header is None:
            header = list(row.keys())
        records.append(row)
        count += 1

        if count % 10000 == 0:
            print(f"  Consumed {count} records...")

        if max_records and count >= max_records:
            break

    consumer.close()

    if count == 0:
        print("[WARN] No records consumed. Is Kafka topic empty?")
        return None

    # Write to HDFS as CSV
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header)
    writer.writeheader()
    writer.writerows(records)

    with client.write(hdfs_path, overwrite=True, encoding="utf-8") as f:
        f.write(buffer.getvalue())

    elapsed = time.time() - start
    print(f"\n[Kafka→HDFS] DONE")
    print(f"  Records written: {count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  HDFS path: {hdfs_path}")

    return hdfs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload TON_IoT data to HDFS")
    parser.add_argument("--mode", choices=["direct", "kafka"], default="direct",
                        help="direct=upload CSV file, kafka=consume from Kafka")
    parser.add_argument("--csv", default="data/ton_iot.csv",
                        help="CSV path (for direct mode)")
    parser.add_argument("--hdfs-url", default=HDFS_URL)
    parser.add_argument("--hdfs-path", default=HDFS_RAW_PATH)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.mode == "direct":
            upload_direct(args.csv, args.hdfs_url, args.hdfs_path)
        else:
            upload_from_kafka(args.hdfs_url, args.hdfs_path,
                              args.bootstrap_servers, KAFKA_TOPIC,
                              args.max_records)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("  Make sure Docker containers are running: docker-compose up -d")
        sys.exit(1)
