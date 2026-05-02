# Đối Chiếu Phần 1: Abstract & Introduction (Paper L1–L121)

> File này đối chiếu **từng câu claim** trong Abstract và Introduction của bài báo với code hiện tại trong `Nhom28_CyberDetect_MLP_Final`.

---

## 1.1 Title & Authors (L1–L8)

| Nội dung paper | Khớp? | Vị trí code | Ghi chú |
|---|---|---|---|
| "CyberDetect MLP a big data enabled optimized deep learning framework for scalable cyberattack detection in IoT environments" | ✅ Tên project | `README.md`, tên thư mục project | — |
| Talluri Upender et al. | N/A | Không cần code | Thông tin tác giả |

---

## 1.2 Abstract (L10–L30)

| # | Claim trong paper (dòng) | Khớp? | File/dòng code | Chi tiết |
|---|---|---|---|---|
| 1 | "Apache Spark for distributed ingestion and preprocessing" (L17–18) | ✅ Đã triển khai | `bigdata/spark/preprocess_from_hdfs.py`, `bigdata/docker-compose.yml` | Spark standalone cluster (1 master + 4 workers) + Kafka + HDFS |
| 2 | "Mutual information–based feature selection" (L18) | ✅ Khớp | `src/preprocessing/feature_selector.py` L80, `scripts/paper_aligned_reproduction.py` L217 | `mutual_info_classif` sklearn |
| 3 | "multi-layer perceptron (MLP) with batch normalization, dropout, and cosine annealing scheduling" (L18–19) | ✅ Khớp | `src/model/cyberdetect_mlp.py` L4–12, `scripts/paper_aligned_reproduction.py` L112–125 | 512-256-128 + BN + Dropout(0.3) + cosine |
| 4 | "explainable AI (XAI) module … utilizing Grad-CAM and SHAP" (L20–21) | ⚠️ Sẽ bổ sung | `notebooks/04_XAI_Paper_Model.ipynb`, `notebooks/08_Explainable_AI.ipynb` | SHAP đã có. Grad-CAM sẽ triển khai dạng gradient-based feature attribution (paper L1253 tự nhận Grad-CAM hạn chế cho MLP) |
| 5 | "full TON_IoT dataset" (L22) | ✅ Khớp | `scripts/paper_aligned_reproduction.py` L194 | Load toàn bộ CSV |
| 6 | "outperforms the baselines of Random Forest, XGBoost, and vanilla MLP" (L22–23) | ✅ Cấu trúc khớp | `scripts/paper_aligned_reproduction.py` L286–302 | RF, XGBoost, Vanilla MLP, CyberDetect-MLP đều có |
| 7 | "accuracy of 98.87% and a ROC-AUC of 99.10%" (L23) | ⚠️ Chờ thực nghiệm | — | Con số này là claim paper; reproduction sẽ report kết quả thực |
| 8 | "Ablation studies and explainability evaluations" (L23–24) | ✅ Khớp | `scripts/paper_aligned_reproduction.py` L317–338 (ablation), notebooks 04/08 (XAI) | — |
| 9 | "publicly available at https://github.com/upender0123/CyberDetect-MLP" (L29–30) | ✅ Đã clone | `CyberDetect-MLP/` (thư mục gốc) | Source gốc đã dùng để đối chiếu |

---

## 1.3 Keywords (L32)

| Keyword | Có trong project? | Ghi chú |
|---|---|---|
| IoT security | ✅ | README.md, docstrings |
| Cyberattack detection | ✅ | Tên project |
| Big data analytics | ✅ | Docker: Kafka + HDFS + Flume + Spark (1M+4W) |
| Deep learning | ✅ | TensorFlow/Keras MLP |
| Intrusion detection system | ✅ | README.md |

---

## 1.4 Introduction Body (L34–L120)

### Bối cảnh IoT & IDS (L34–69)
- **Nội dung**: mô tả thách thức IoT, hạn chế IDS truyền thống.
- **Cần code?**: Không. Đây là phần literature review / motivate.
- **Trạng thái**: N/A (không cần triển khai code).

### Contributions List (L85–L113)

| # | Contribution claim | Khớp? | Chi tiết code |
|---|---|---|---|
| C1 | "Big data–enabled IDS: Kafka/Flume + HDFS + Spark" (L95–97) | ✅ Đã triển khai | `bigdata/`: Kafka producer, Flume agent+config, HDFS NameNode/DataNode, Spark 1M+4W. Pipeline: CSV→Kafka→HDFS→Spark→Parquet→MLP |
| C2 | "Custom 8-layer MLP with BN, Dropout, cosine annealing" (L99–101) | ✅ Khớp | `cyberdetect_mlp.py`: Input + Dense(512)+BN+Dropout + Dense(256)+BN+Dropout + Dense(128)+BN+Dropout + Output = 8 layers |
| C3 | "Mutual Information-based feature selection" (L103–105) | ✅ Khớp | `feature_selector.py` L80, `paper_aligned_reproduction.py` L217 |
| C4 | "Grad-CAM and SHAP interpretability" (L107–109) | ⚠️ Sẽ bổ sung | SHAP đã có. Sẽ triển khai gradient-based attribution (tương đương Grad-CAM cho MLP, đúng theo paper L1253) |
| C5 | "accuracy of 98.87% on TON_IoT" (L111–113) | ⚠️ Chờ thực nghiệm | Reproduction sẽ báo cáo kết quả thực tế |

### Paper Structure Overview (L115–L120)
- **Nội dung**: mô tả cấu trúc các section.
- **Cần code?**: Không.
- **Trạng thái**: N/A.

---

## Tổng kết phần 1

| Tiêu chí | Số claim | Khớp ✅ | Sẽ triển khai ⚠️ | Không khớp ❌ | N/A |
|---|---|---|---|---|---|
| Abstract | 9 | 6 | 2 | 0 | 1 |
| Keywords | 5 | 5 | 0 | 0 | 0 |
| Contributions | 5 | 3 | 2 | 0 | 0 |
| **Tổng** | **19** | **14** | **4** | **0** | **1** |

Các mục ⚠️ còn lại: Grad-CAM sẽ bổ sung, chờ kết quả thực nghiệm.
