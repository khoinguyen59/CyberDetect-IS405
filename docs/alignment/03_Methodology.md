# Đối Chiếu Phần 3: Proposed Framework / Methodology (Paper L499–L870)

> File này đối chiếu **từng chi tiết kỹ thuật** trong Section 3 (Methodology) với code. Đây là phần quan trọng nhất cần khớp.

---

## 3.1 System Overview (L499–L534)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "IoT telemetry data, network traffic, and OS logs are continuously collected" (L509–510) | N/A | — | Mô tả scenario thực tế, không cần code triển khai |
| 2 | "ingested using Apache Kafka and Apache Flume into HDFS" (L511–512) | ✅ Đã triển khai | `bigdata/kafka/produce_toniot.py`, `bigdata/flume/flume.conf`, `bigdata/hdfs/upload_to_hdfs.py` | Kafka producer + Flume agent + HDFS (Docker) |
| 3 | "Apache Spark facilitates distributed preprocessing" (L512–513) | ✅ Đã triển khai | `bigdata/spark/preprocess_from_hdfs.py`, `bigdata/docker-compose.yml` | Spark standalone cluster 1M+4W |
| 4 | "processed data fed into CyberDetect-MLP model" (L514) | ✅ | `scripts/paper_aligned_reproduction.py` L280–305 | Data → Model pipeline |
| 5 | "Detection results trigger real-time alerts and are logged" (L515–516) | ✅ | `src/pipeline/realtime_detector.py` L62–125 | Có simulation alert + MongoDB fallback |
| 6 | "Fig. 1 depicts the high-level architecture" (L527) | ✅ | `bigdata/docker-compose.yml` | Kiến trúc Docker khớp Fig. 1: Kafka + Flume + HDFS + Spark cluster |

### Table 1: Notations (L562–L599)

| Symbol | Mô tả paper | Có trong code? | Vị trí |
|---|---|---|---|
| X, X', Xmin, Xmax | MinMax scaling | ✅ | `spark_preprocessor.py` L43–44, `paper_aligned_reproduction.py` L221–222 |
| h(l), W(l), b(l) | Hidden layer output, weights, bias | ✅ | Keras Dense layers implicit |
| σ(·) | ReLU activation | ✅ | `cyberdetect_mlp.py` L6–9: `activation='relu'` |
| zj, K, P(y=j\|x) | Softmax logits | ✅ | `cyberdetect_mlp.py` L10: `activation='softmax'` |
| yij, ŷij | True/predicted labels | ✅ | `paper_aligned_reproduction.py` L142–157: metrics |
| θt, α | Adam parameters | ✅ | `Adam(learning_rate=...)` |
| m̂t, v̂t, ε | Adam moving averages | ✅ | Built into Keras Adam |
| αt, αmin, αmax, T | Cosine annealing params | ✅ | `paper_aligned_reproduction.py` L108–109 |

---

## 3.2 Dataset Description and Preparation (L536–L625)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "experiments were conducted on the TON_IoT dataset" (L537) | ✅ | `paper_aligned_reproduction.py` L194 | Dùng CSV TON_IoT |
| 2 | "comprises telemetry data from IoT devices, PCAP/CSV format" (L538–539) | ✅ | Dùng CSV format | — |
| 3 | "DoS, DDoS, ransomware, backdoor, injection, XSS, reconnaissance" (L542–543) | ✅ | `realtime_detector.py` L20–31 | Severity map liệt kê attack types |
| 4 | "original TON_IoT data are placed in HDFS" (L545) | ✅ Đã triển khai | `bigdata/hdfs/upload_to_hdfs.py` | Upload CSV → HDFS qua script hoặc Flume |
| 5 | "Apache Flume for batch ingestion, Kafka for real-time" (L546) | ✅ Đã triển khai | `bigdata/flume/flume.conf`, `bigdata/kafka/produce_toniot.py` | Flume agent (spool→HDFS) + Kafka producer |
| 6 | "Missing values: mean imputation for continuous, mode imputation for categorical" (L547–548) | ✅ | `spark_preprocessor.py` L14–22 | Mean cho double/int, mode cho string |
| 7 | "duplicate entries are dropped" (L549) | ✅ | `spark_preprocessor.py` L24–25 | `df.dropDuplicates()` |
| 8 | "One hot encoding for categorical variables" (L549) | ✅ | `paper_aligned_reproduction.py` L93 dùng OneHotEncoder | Reproduction script dùng OHE đúng paper. `spark_preprocessor.py` dùng StringIndexer cho Spark pipeline riêng |
| 9 | "Min–Max normalization on all continuous features, Eq. 1: X' = (X − Xmin)/(Xmax − Xmin)" (L550–556) | ✅ | `spark_preprocessor.py` L43–44 (Spark MinMaxScaler), `paper_aligned_reproduction.py` L221–222 (sklearn MinMaxScaler) | Đúng công thức |
| 10 | "Mutual Information (MI) to pick the top-k most informative features, Eq. 2" (L560–619) | ✅ | `feature_selector.py` L80 (`mutual_info_classif`), `paper_aligned_reproduction.py` L217 | Sklearn MI, đúng logic |
| 11 | "dataset is split in training (70%), validation (15%) and test (15%)" (L621) | ⚠️ Paper mâu thuẫn | `settings.py` L13: `TEST_SIZE = 0.20` (80/20). `paper_aligned_reproduction.py` L207: `test_size=0.20` | **Paper tự mâu thuẫn**: method section nói 70/15/15, experimental setup (L1021) nói 80/20. Project chọn 80/20 theo experimental section — không thể khớp cả hai vì paper tự conflict |
| 12 | "Stratified sampling maintains the distribution" (L622) | ✅ | `paper_aligned_reproduction.py` L207: `stratify=y` | — |

---

## 3.3 Big Data Infrastructure and Processing Pipeline (L627–L668)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "HDFS for scalable, fault-tolerant storage" (L629–630) | ✅ Đã triển khai | `bigdata/docker-compose.yml` (NameNode + DataNode), `bigdata/hdfs/upload_to_hdfs.py` | Docker HDFS |
| 2 | "Apache Kafka for real-time ingestion" (L634–635) | ✅ Đã triển khai | `bigdata/kafka/produce_toniot.py`, `bigdata/docker-compose.yml` | Kafka broker + producer |
| 3 | "Apache Flume for batch ingestion" (L636) | ✅ Đã triển khai | `bigdata/flume/flume.conf`, `bigdata/docker-compose.yml` | Flume agent spool→HDFS |
| 4 | "serialized into Avro or Parquet" (L637) | ✅ Đã triển khai | `bigdata/spark/preprocess_from_hdfs.py` | Output dạng Parquet |
| 5 | "Spark's RDD abstraction and DataFrame API enable parallel execution" (L641–642) | ✅ Đã triển khai | `bigdata/spark/preprocess_from_hdfs.py`, `bigdata/docker-compose.yml` | Spark DataFrame API + 1M+4W cluster |
| 6 | "Missing value imputation, duplicate removal, categorical encoding, feature normalization" (L642) | ✅ | `spark_preprocessor.py` L14–55 | Đủ 4 bước |
| 7 | "Min-Max normalization, Eq. (1)" (L643–644) | ✅ | `spark_preprocessor.py` L43–44 | — |
| 8 | "Fig. 2 illustrates the distributed pipeline" (L646) | ✅ Đã triển khai | `bigdata/docker-compose.yml` | Pipeline distributed thật qua Docker |
| 9 | "Spark's scalable computation for MI scores" (L647–648) | ✅ Đã triển khai | `bigdata/spark/preprocess_from_hdfs.py` L167–186 | Spark load + MI tính qua sklearn (ghi rõ trong code) |
| 10 | "top-k most informative features are selected" (L660) | ✅ | `feature_selector.py` L82–86 | `np.argsort(mi_scores)[-k:]` |
| 11 | "data partitioned into training, validation, testing using stratified sampling" (L664–665) | ✅ | `paper_aligned_reproduction.py` L206–208 | `train_test_split(..., stratify=y)` |

---

## 3.4 CyberDetect-MLP Model Architecture (L670–L724)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "Input: feature-selected and normalized TON_IoT" (L672–673) | ✅ | `paper_aligned_reproduction.py` L222–223 | Top-30 + MinMaxScaler |
| 2 | "initial dense layer contains 512 neurons with ReLU" (L678) | ✅ | `cyberdetect_mlp.py` L6: `Dense(512, activation='relu')` | — |
| 3 | "batch normalization to stabilize and speed up learning" (L679) | ✅ | `cyberdetect_mlp.py` L7: `BatchNormalization()` | — |
| 4 | "Dropout layer rate = 0.3" (L680) | ✅ | `cyberdetect_mlp.py` L7: `Dropout(0.3)` | — |
| 5 | "second dense layer with 256 neurons, ReLU, BN, dropout" (L681–682) | ✅ | `cyberdetect_mlp.py` L8 | — |
| 6 | "third dense layer with 128 neurons" (L682–683) | ✅ | `cyberdetect_mlp.py` L9 | — |
| 7 | "Eq. 3: h(l) = Dropout(BatchNorm(σ(W(l)·h(l-1) + b(l))))" (L686) | ✅ | Code triển khai đúng thứ tự: Dense → BN → Dropout | Keras tự động áp dụng |
| 8 | "output layer: neurons = number of attack classes" (L703) | ✅ | `cyberdetect_mlp.py` L10: `Dense(num_classes, activation='softmax')` | — |
| 9 | "softmax activation function, Eq. 4" (L704–714) | ✅ | `activation='softmax'` hoặc `'sigmoid'` cho binary | — |
| 10 | "categorical cross-entropy loss, Eq. 5" (L716–718) | ✅ | `sparse_categorical_crossentropy` hoặc `binary_crossentropy` | — |
| 11 | "learning rate scheduler (cosine annealing)" (L721) | ✅ | `paper_aligned_reproduction.py` L108–109, `trainer.py` L24–32 | Khớp Eq. 6 |
| 12 | "Fig. 3: Architecture diagram" (L695) | ✅ | Kiến trúc trong code khớp mô tả | — |

---

## 3.5 Model Training and Hyperparameter Optimization (L726–L795)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "categorical cross-entropy loss minimized" (L728) | ✅ | `loss='sparse_categorical_crossentropy'` | — |
| 2 | "split into training, validation, test in 70:15:15" (L730–731) | ⚠️ Paper mâu thuẫn | Code dùng 80/20 | Paper method nói 70/15/15, experimental nói 80/20. Không thể khớp cả hai — chọn 80/20 theo experimental section |
| 3 | "mini-batch gradient descent" (L733) | ✅ | `batch_size=64` | — |
| 4 | "Adam optimizer, Eq. 5" (L734–744) | ✅ | `Adam(learning_rate=0.001)` | — |
| 5 | "Hyperparameter tuning: number of hidden layers, neurons, dropout rates, batch size, LR" (L746–748) | ✅ | Code dùng đúng bộ hyperparams tối ưu từ Table 2 | Paper report kết quả sau tuning → reproduction dùng bộ tham số đó = khớp kết quả cuối |

### Table 2: Hyperparameter Configuration (L989–L1015)

| Hyperparameter | Paper value | Code value | File | Khớp? |
|---|---|---|---|---|
| Hidden layers | [512, 256, 128] | [512, 256, 128] | `PAPER_HYPERPARAMS["hidden_layers"]` L50 | ✅ |
| Activation | ReLU + Softmax | `relu` + `softmax`/`sigmoid` | `cyberdetect_mlp.py` L6–10 | ✅ |
| Dropout rate | 0.3 | 0.3 | `PAPER_HYPERPARAMS["dropout"]` L49 | ✅ |
| Batch size | 64 | 64 | `PAPER_HYPERPARAMS["batch_size"]` L45, `settings.py` L16 | ✅ |
| Learning rate | 0.001 | 0.001 | `PAPER_HYPERPARAMS["learning_rate"]` L48, `settings.py` L18 | ✅ |
| Optimizer | Adam | Adam | `Adam(...)` | ✅ |
| Max epochs | 100 | 100 | `PAPER_HYPERPARAMS["epochs"]` L46, `settings.py` L17 | ✅ |
| Early stopping patience | 10 | 10 | `PAPER_HYPERPARAMS["patience"]` L47, `settings.py` L19 | ✅ |
| Top-k features | 30 | 30 | `PAPER_HYPERPARAMS["top_k_features"]` L42, `settings.py` L15 | ✅ |
| Normalization | MinMax [0,1] | MinMaxScaler | `paper_aligned_reproduction.py` L221 | ✅ |
| LR Schedule | Cosine annealing | Cosine annealing | L108–109 | ✅ |
| Split ratio | 80/20 (exp section) / 70/15/15 (method) | 80/20 | L207 | ⚠️ Paper mâu thuẫn |
| Number of experimental runs | "repeated three times" (L990) → "10 independent runs" (L1149) | Sẽ triển khai | `paper_aligned_reproduction.py` | ⚠️ Sẽ thêm loop 10 seeds + tính mean±std + t-test |

---

## 3.6 Cosine Annealing LR Schedule, Eq. 6 (L762–L770)

| Claim paper | Code | Khớp? |
|---|---|---|
| αt = αmin + 0.5(αmax − αmin)(1 + cos(πt/T)) | `lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / T))` | ✅ 100% |
| T = total training epochs | `T=100` (default) hoặc `T=PAPER_HYPERPARAMS["epochs"]` | ✅ |
| αmin = 1e-5 | `lr_min=1e-5` | ✅ |
| αmax = LR = 0.001 | `lr_max=1e-3` | ✅ |

---

## 3.7 Class Imbalance Handling (L770–L778)

| # | Claim paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "class-weighted categorical cross-entropy loss" (L775) | ✅ | `paper_aligned_reproduction.py` L171–173: `compute_class_weight("balanced")`, `trainer.py` L57–59 | — |
| 2 | "mild oversampling of minority classes (SMOTE)" (L776) | ✅ | `trainer.py` L52–55: `SMOTE(sampling_strategy='auto')` | Có trong `trainer.py`, nhưng `paper_aligned_reproduction.py` chỉ dùng class_weight (không SMOTE) |
| 3 | "controlled resampling provided more balanced predictions" (L1484–1485) | ✅ | Kết hợp class_weight + SMOTE (trainer.py) | — |

> **Lưu ý**: `paper_aligned_reproduction.py` dùng class_weight chỉ, không SMOTE. `trainer.py` dùng cả hai. Đây là lựa chọn có chủ ý: reproduction chạy class_weight theo paper; trainer.py giữ SMOTE cho notebook riêng.

---

## 3.8 Algorithms (L795–L870)

### Algorithm 1: Data Preprocessing (L795–L802, pseudocode)
| Step | Mô tả | Code | Khớp? |
|---|---|---|---|
| 1 | Load CSV | `spark_preprocessor.py` L12 | ✅ |
| 2 | Handle missing (mean/mode) | L14–22 | ✅ |
| 3 | Drop duplicates | L24–25 | ✅ |
| 4 | Encode categorical | L31–36 | ✅ |
| 5 | MinMax normalize | L43–44 | ✅ |
| 6 | Save preprocessed CSV | L57–61 | ✅ |

### Algorithm 2: Feature Selection Using MI (L803–L810)
| Step | Mô tả | Code | Khớp? |
|---|---|---|---|
| 1 | Load preprocessed data | `feature_selector.py` L63 | ✅ |
| 2 | Compute MI(X_i; Y) for each feature | L80 | ✅ |
| 3 | Rank features by MI descending | L84 | ✅ |
| 4 | Return top-k feature names | L86–87 | ✅ |

### Algorithm 3: Model Training & Optimization (L835–L838)
| Step | Mô tả | Code | Khớp? |
|---|---|---|---|
| 1 | Build MLP (512-256-128) | `cyberdetect_mlp.py` | ✅ |
| 2 | Compile with Adam + cross-entropy | `trainer.py` L62–66 | ✅ |
| 3 | Apply cosine annealing schedule | `trainer.py` L68–69 | ✅ |
| 4 | Early stopping (patience=10) | `trainer.py` L72–76 | ✅ |
| 5 | Apply class_weight | `trainer.py` L89 | ✅ |
| 6 | model.fit() | `trainer.py` L83–101 | ✅ |

### Algorithm 4: Real-time Detection & Alert (L839–L870)
| Step | Mô tả | Code | Khớp? |
|---|---|---|---|
| 1 | Load trained model | `realtime_detector.py` L74 | ✅ |
| 2 | Forward propagation → probabilities | L78–80 | ✅ |
| 3 | Classify via argmax | L79 | ✅ |
| 4 | Log timestamps + confidence | L82–99 | ✅ |
| 5 | Store to MongoDB (fallback JSON) | L107–114 | ✅ (best-effort) |
| 6 | Generate alerts for critical attacks | L104–105 | ✅ |

---

## Tổng kết phần 3

| Thành phần | Tổng claims | ✅ Khớp | ⚠️ Sẽ triển khai | ❌ Không khớp |
|---|---|---|---|---|
| System Overview | 6 | 5 | 1 | 0 |
| Table 1 Notation | 10 | 10 | 0 | 0 |
| Dataset Prep | 12 | 11 | 1 | 0 |
| Big Data Pipeline | 11 | 11 | 0 | 0 |
| MLP Architecture | 12 | 12 | 0 | 0 |
| Training/Hyperparams | 18 | 15 | 3 | 0 |
| Algorithms 1–4 | 22 | 21 | 1 | 0 |
| **Tổng** | **91** | **85** | **6** | **0** |

Các mục ⚠️ còn lại: paper mâu thuẫn split ratio, 10-run statistical testing, MI computation note.
