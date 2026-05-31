# 🚀 ĐỒ ÁN MÔN DỮ LIỆU LỚN (IS405.Q21)
## CẢI TIẾN FRAMEWORK PHÁT HIỆN TẤN CÔNG MẠNG IOT TRÊN NỀN TẢNG DỮ LIỆU LỚN
### (Cross-dataset Generalization · Class Imbalance Handling · Explainable AI)

**Nhóm 28:** Nguyễn Trọng Khôi (23520783) · Phạm Duy Hoàng (23520537)

**Bài báo nền (Base Paper):** Upender T. et al., *"CyberDetect-MLP: A Big Data-Enabled Optimized Deep Learning Framework for Scalable Cyberattack Detection in IoT Environments"*, Scientific Reports (Nature, Q1), 2025. DOI: https://doi.org/10.1038/s41598-025-24459-w

---

## 1. Giới thiệu
Dự án **tái hiện (reproduce)** framework CyberDetect-MLP trên dataset **TON_IoT**, đồng thời **kiểm toán (audit)** và **cải tiến** theo các khoảng trống nghiên cứu: cross-dataset (retrain thay vì chỉ zero-shot), xử lý mất cân bằng lớp, Explainable AI, và đánh giá concept drift.

> **Nguyên tắc của nhóm:** mọi số liệu trong gói này là **đo thật** từ mô hình (không có dữ liệu/kết quả giả lập). Một số con số "không đẹp" (MLP thua tree-model, SMOTE-ENN làm giảm hiệu năng, CSE-CIC khó tổng quát) được giữ nguyên vì đó chính là **phát hiện kiểm toán có giá trị**.

---

## 2. Kiến trúc & 2 luồng thực thi
- **Luồng học thuật (chính — chạy 10 notebook):** `CSV → sklearn (One-hot + True MI top-30) → MLP Keras (đa lớp)`. Chạy trên Google Colab, dùng **Mutual Information thật** của sklearn.
- **Luồng Big Data (prototype minh hoạ kiến trúc):** `Kafka/Flume → HDFS → Spark → Parquet` chạy bằng **Docker single-machine** trong `bigdata/`.

> **Giới hạn trung thực:** phần Big Data chỉ là prototype Docker đơn máy (không phải cụm 5 node như bài gốc). Benchmark ở NB09 đo **suy luận MLP trên đơn máy**, không phải throughput cụm phân tán.

---

## 3. Kết quả thực nghiệm (SỐ THẬT — phân loại ĐA LỚP 10 lớp)

> ⚠️ Bài báo gốc công bố 98.87% nhưng mã nguồn gốc cấu hình **nhị phân** (sigmoid). Nhóm chuẩn hoá đúng **đa lớp** (softmax + categorical cross-entropy, 10 lớp `type`). Vì vậy accuracy thực tế ~87% — **trung thực hơn** con số 98.87% nhị phân.

### 3.1. Table 3 — So sánh baselines (NB02)
| Model | Accuracy | F1-macro | ROC-AUC |
|---|---|---|---|
| Random Forest | 98.96% | 0.969 | 0.9994 |
| XGBoost | 98.90% | 0.964 | 0.9998 |
| Vanilla MLP | 91.61% | 0.851 | 0.9927 |
| **CyberDetect-MLP** | **87.45%** | **0.809** | **0.9889** |

> Tree-model (RF/XGB) vượt MLP trên dữ liệu dạng bảng — kết quả trung thực, đúng với thực tế tabular data. CyberDetect-MLP dùng class-weight (ưu tiên lớp thiểu số) nên accuracy tổng thấp hơn Vanilla MLP.

### 3.2. Table 5 — Ablation (NB03)
| Biến thể | Accuracy | F1-macro |
|---|---|---|
| Full Model | 87.45% | 0.809 |
| MLP-NoFeatureSelect (124 feature) | 85.01% | 0.786 |
| **MLP-NoBatchNorm** ⭐ | **88.41%** | **0.819** |
| MLP-NoDropout | 87.05% | 0.802 |
| MLP-NoScheduler | 86.14% | 0.792 |

> **Phát hiện audit:** Bỏ BatchNorm cho kết quả TỐT NHẤT → kiến trúc paper báo cáo (có BatchNorm) không tối ưu trên tabular IoT. MI feature selection có ích (NoFeatureSelect tệ nhất).

### 3.3. Xử lý mất cân bằng — F1 per-class (NB06)
| Lớp | Baseline | Class-Weight | SMOTE-ENN | Hybrid |
|---|---|---|---|---|
| ransomware | 0.948 | 0.944 | 0.706 | 0.718 |
| backdoor | 0.989 | 0.975 | 0.919 | 0.923 |
| xss | 0.860 | 0.856 | **0.000** | 0.000 |
| mitm (hiếm nhất) | 0.201 | **0.273** | 0.204 | 0.190 |

> **Phát hiện audit:** SMOTE-ENN mạnh tay **làm GIẢM** hiệu năng hầu hết các lớp (xss sập về 0). Class-weight nhẹ nhàng giúp lớp hiếm nhất `mitm` → củng cố luận điểm "bài gốc lạm dụng resampling".

### 3.4. Concept Drift (NB07)
- Rolling accuracy (đo thật) giảm **~87% → ~72%** khi tiêm covariate shift (×50 trên đặc trưng byte/packet) tại t=10,000.
- KS-test: KS 0.03→0.32, p-value ~1e-45; phát hiện drift đầu tiên tại **t=10,300 (độ trễ 300 sự kiện, p=0.00122)**.

### 3.5. Cross-Dataset (NB08)
| Dataset | Zero-shot | Retrain |
|---|---|---|
| UNSW-NB15 | ~32% | **93.79%** |
| BoT-IoT | ~0% | **99.99%** *(accuracy cao do mất cân bằng cực đoan — xem F1/recall lớp Normal)* |
| CSE-CIC-IDS2018 (CICFlowMeter, 500k, class_weight) | — | **85.81%** (ROC-AUC 0.848; recall Attack 0.41 → tổng quát MỘT PHẦN) |

> Zero-shot thất bại do domain shift → Retrain phục hồi mạnh. CSE-CIC (họ đặc trưng khác hẳn) tổng quát một phần.

### 3.6. Kiểm định thống kê (NB10) & Benchmark (NB09)
- **10 runs:** Accuracy 0.8694 ± 0.0035, F1-macro **0.8031 ± 0.0044**, ROC-AUC 0.9880 ± 0.0006 → mô hình **ổn định** (Std rất nhỏ).
- **Suy luận (đơn máy):** ~**0.053 ms/mẫu**, throughput ~**18,800 sự kiện/giây** (CPU/Mem đo bằng psutil). Bài gốc 98,500/s là cụm 5 node Spark.

### 3.7. Explainable AI (NB04)
Thay Grad-CAM (vốn cho CNN) bằng **SHAP (global)** + **Integrated Gradients (local)** — phù hợp MLP/tabular. Xem `results/xai/`.

---

## 4. Cấu trúc thư mục
```text
Nhom28_CyberDetect_MLP_Final/
├── notebooks/        # 10 notebook Colab (01→10) — có sẵn kết quả đã chạy
├── scripts/          # Script ML gọi bởi notebook 06–10
├── src/              # model/ preprocessing/ modules(resampling, cross_dataset)/ pipeline/ config/
├── results/          # CSV + PNG kết quả thật (Table 3/5, ablation, drift, cross-dataset, statistical, benchmark, xai)
├── models/           # colab_cyberdetect_mlp.h5, colab_preprocessor.joblib, metadata, model CSE-CIC
├── data/             # ton_iot.csv + data/raw (UNSW-NB15, BoT-IoT) + colab_processed
├── bigdata/          # Prototype Docker (Kafka/Flume/HDFS/Spark)
├── app_dashboard.py  # Dashboard Streamlit (demo)
├── RUN_ORDER.md      # ★ Hướng dẫn chạy chi tiết + thứ tự + kỳ vọng kết quả
└── README.md         # File này
```

## 5. Danh sách 10 notebook
| # | Notebook | Nội dung |
|---|---|---|
| 01 | Preprocessing_MI_FeatureSelection | Tiền xử lý đa lớp, One-hot, MI top-30 |
| 02 | Table3_Baseline_Comparison | RF/XGB/Vanilla MLP/CyberDetect-MLP |
| 03 | Table5_Ablation_Study | Vai trò BatchNorm/Dropout/Scheduler/MI |
| 04 | Explainable_AI_SHAP_IG | SHAP + Integrated Gradients |
| 05 | Final_Audit_Checklist | Kiểm tra đầy đủ artifacts |
| 06 | Class_Imbalance_Resampling | F1 per-class: baseline/class-weight/SMOTE-ENN/hybrid |
| 07 | Concept_Drift_Analysis | Mô phỏng drift + KS-test |
| 08 | Cross_Dataset_Retrain | UNSW-NB15, BoT-IoT, CSE-CIC-IDS2018 |
| 09 | Benchmark_Spark_Performance | Latency/throughput suy luận |
| 10 | Statistical_Significance_Testing | 10 runs, Mean±Std, t-test |

## 6. Hướng dẫn chạy
**Xem chi tiết trong [`RUN_ORDER.md`](RUN_ORDER.md).** Tóm tắt:
1. Upload thư mục này lên `MyDrive/Nhom28_CyberDetect_MLP_Final` (đúng tên).
2. Colab: chọn **GPU T4/L4/A100 + High-RAM** (tránh GPU Blackwell/RTX PRO 6000).
3. Chạy **NB01 → NB02 trước** (sinh artifacts + model), rồi 03–10 (thứ tự bất kỳ; NB04/07/09 cần model từ NB02; NB08 tự tải CSE-CIC từ AWS).

Thư viện: `pip install tensorflow scikit-learn imbalanced-learn pandas numpy pyarrow matplotlib seaborn shap xgboost joblib psutil scipy`.

---
*Mọi số liệu đọc trực tiếp từ thư mục `results/`. Báo cáo chi tiết + slide xem trong gói nộp kèm.*

# 
# 