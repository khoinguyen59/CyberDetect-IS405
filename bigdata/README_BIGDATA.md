# Big Data Pipeline - CyberDetect-MLP

Pipeline n?y hi?n th?c h?a nh?nh Big Data c?a b?i b?o ? m?c Docker prototype tr?n m?t m?y:

```text
ton_iot.csv
  -> Kafka producer (topic: toniot-events)      [realtime ingestion]
  -> Flume spool directory                      [batch ingestion]
  -> HDFS storage                               [distributed storage]
  -> Spark preprocessing (1 master + 4 workers) [distributed processing]
  -> Parquet output (bigdata/output/)
  -> MLP training (--from-spark)                [model training]
```

Output c?a Spark l? input tr?c ti?p c?a MLP.

## Docker services

| Service | Vai tr? |
|---|---|
| Kafka + Zookeeper | Realtime ingestion |
| Flume | Batch log ingestion v?o HDFS |
| HDFS NameNode + DataNode | Storage |
| Spark Master + 4 Workers | Distributed preprocessing prototype |

L?u ?: ??y l? containerized single-machine prototype, kh?ng ph?i c?m v?t l? 5 m?y.

## Y?u c?u

```bash
pip install kafka-python hdfs pyspark pandas pyarrow scikit-learn
```

C?n Docker Desktop ho?c Docker Engine.

## Ch?y pipeline

T? th? m?c `Nhom28_CyberDetect_MLP_Final`:

```bash
cd bigdata
docker-compose up -d
cd ..
```

Ki?m tra UI:

```text
HDFS:  http://localhost:9870
Spark: http://localhost:8080
Kafka: localhost:9092
```

### C?ch A - Kafka -> HDFS

```bash
python bigdata/kafka/produce_toniot.py --csv data/ton_iot.csv
python bigdata/hdfs/upload_to_hdfs.py --mode kafka
python bigdata/spark/preprocess_from_hdfs.py --local --input hdfs://localhost:9000/cyberdetect/raw/ton_iot.csv
python scripts/paper_aligned_reproduction.py --from-spark bigdata/output --section all
```

### C?ch B - Flume -> HDFS

Flume spool directory ch? nh?n file m?i. N?u file ?? t?ng ???c consume, ??i t?n file tr??c khi copy.

```bash
copy data	on_iot.csv bigdatalume\spool	on_iot_flume.csv
```

Flume ghi v?o:

```text
hdfs://namenode:9000/cyberdetect/raw/
```

Sau ?? c?n d?ng ??ng path file Flume sinh ra trong HDFS ?? ch?y Spark. C? th? xem tr?n HDFS UI:

```text
http://localhost:9870/explorer.html#/cyberdetect/raw
```

### C?ch C - Direct CSV -> HDFS

C?ch n?y thay th? batch ingestion khi ch? c?n test nhanh HDFS/Spark:

```bash
python bigdata/hdfs/upload_to_hdfs.py --mode direct --csv data/ton_iot.csv
python bigdata/spark/preprocess_from_hdfs.py --local --input hdfs://localhost:9000/cyberdetect/raw/ton_iot.csv
python scripts/paper_aligned_reproduction.py --from-spark bigdata/output --section all
```

## Output

Spark preprocessing t?o:

```text
bigdata/output/train_features.parquet
bigdata/output/test_features.parquet
bigdata/output/train_labels.parquet
bigdata/output/test_labels.parquet
bigdata/output/metadata.json
```

MLP training t?o:

```text
results/paper_aligned_table3_metrics.csv
results/paper_aligned_ablation_metrics.csv
models/cyberdetect_mlp_paper_aligned.h5
models/paper_aligned_metadata.json
```

## M?c kh?p b?i b?o

| Paper claim | File tri?n khai | Tr?ng th?i |
|---|---|---|
| Kafka realtime ingestion | `bigdata/kafka/produce_toniot.py` | C? |
| Flume batch ingestion | `bigdata/flume/flume.conf` | C? |
| HDFS storage | `bigdata/hdfs/upload_to_hdfs.py` + Docker HDFS | C? |
| Spark preprocessing | `bigdata/spark/preprocess_from_hdfs.py` | C? |
| Spark Structured Streaming | `bigdata/spark/stream_from_kafka.py` | Demo optional |
| 1 master + 4 workers | `docker-compose.yml` | C? trong Docker |
| N?i sang MLP | `--from-spark bigdata/output` | C? |

Gi?i h?n c?n l?i: ch?y tr?n m?t m?y b?ng Docker containers, kh?ng ph?i c?m v?t l? nhi?u node.
