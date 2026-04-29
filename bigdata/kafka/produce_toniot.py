"""
Kafka Producer — CSV → Kafka topic "toniot-events"
Paper ref: L511 "ingested using Apache Kafka for real-time streaming"

Usage:
    python bigdata/kafka/produce_toniot.py --csv data/ton_iot.csv
"""

import argparse
import csv
import json
import time
import sys

from kafka import KafkaProducer


def create_producer(bootstrap_servers="localhost:9092"):
    """Create Kafka producer with JSON serializer."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
        batch_size=16384,
        linger_ms=10,
    )


def produce_csv_to_kafka(csv_path, topic="toniot-events", batch_size=1000,
                         bootstrap_servers="localhost:9092"):
    """
    Read CSV file and send rows to Kafka topic in batches.
    Each row is sent as a JSON message.

    Paper pipeline: TON_IoT CSV → Kafka → HDFS
    """
    producer = create_producer(bootstrap_servers)
    sent_count = 0
    start_time = time.time()

    print(f"[Kafka Producer] Reading: {csv_path}")
    print(f"[Kafka Producer] Topic: {topic}")
    print(f"[Kafka Producer] Batch size: {batch_size}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []

        for row in reader:
            # Convert numeric strings to numbers where possible
            cleaned_row = {}
            for key, value in row.items():
                try:
                    cleaned_row[key] = float(value)
                    if cleaned_row[key] == int(cleaned_row[key]):
                        cleaned_row[key] = int(cleaned_row[key])
                except (ValueError, TypeError):
                    cleaned_row[key] = value

            producer.send(topic, value=cleaned_row)
            sent_count += 1

            if sent_count % batch_size == 0:
                producer.flush()
                elapsed = time.time() - start_time
                rate = sent_count / elapsed if elapsed > 0 else 0
                print(f"  Sent {sent_count} records ({rate:.0f} records/sec)")

    producer.flush()
    producer.close()

    elapsed = time.time() - start_time
    rate = sent_count / elapsed if elapsed > 0 else 0
    print(f"\n[Kafka Producer] DONE")
    print(f"  Total records: {sent_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Throughput: {rate:.0f} records/sec")

    return sent_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send TON_IoT CSV to Kafka")
    parser.add_argument("--csv", default="data/ton_iot.csv",
                        help="Path to TON_IoT CSV file")
    parser.add_argument("--topic", default="toniot-events",
                        help="Kafka topic name")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    try:
        produce_csv_to_kafka(args.csv, args.topic, args.batch_size,
                             args.bootstrap_servers)
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {args.csv}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Kafka connection failed: {e}")
        print("  Make sure Docker containers are running: docker-compose up -d")
        sys.exit(1)
