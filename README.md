# 🚀 ĐỒ ÁN MÔN DỮ LIỆU LỚN (IS405.Q21)
## CẢI TIẾN FRAMEWORK PHÁT HIỆN TẤN CÔNG MẠNG IOT TRÊN NỀN TẢNG DỮ LIỆU LỚN

**Nhóm sinh viên thực hiện:** 
- Nguyễn Trọng Khôi (Nhóm 28)
- Phạm Duy Hoàng (Nhóm 28)

**Bài báo tham chiếu (Base Paper):** 
Upender T. et al., *"CyberDetect-MLP: A Big Data-Enabled Optimized Deep Learning Framework for Scalable Cyberattack Detection in IoT Environments"*, Scientific Reports (Nature, Q1), 11/2025. 
DOI: https://doi.org/10.1038/s41598-025-24459-w

---

## 📖 1. Giới thiệu dự án
Dự án này là mã nguồn hoàn chỉnh của đồ án môn Dữ liệu lớn (IS405.Q21). Mục tiêu của đồ án là tái hiện (reproduce) và cải tiến framework bảo mật **CyberDetect-MLP** chuyên dùng để phát hiện các cuộc tấn công mạng trong môi trường IoT. 

Framework kết hợp công nghệ xử lý Dữ liệu lớn (Apache Spark) và mô hình Học Sâu (Deep Learning - MLP) để xử lý lượng lớn dữ liệu mạng theo thời gian thực. Nhóm tái hiện nghiên cứu gốc kết hợp với việc kiểm toán (Audit) phương pháp luận để khắc phục lỗi Data Leakage, đồng thời đánh giá khả năng thích nghi trên nhiều bộ dữ liệu IoT khác nhau sau khi Retrain và xử lý vấn đề mất cân bằng lớp (Class Imbalance).

---

## 🏗️ 2. Kiến trúc Framework

Hệ thống được chia thành 3 tầng (Layers) cốt lõi:
1. **Tầng Dữ Liệu Lớn (Big Data Ingestion & Preprocessing):** 
   - Sử dụng **Apache Spark (Local Prototype)** để mô phỏng tiền xử lý các tệp dữ liệu IoT khổng lồ.
   - Các tác vụ bao gồm Null handling, StringIndexer, OneHotEncoder và MinMax Scaling.
2. **Tầng Đặc Trưng (Feature Engineering):**
   - Áp dụng thuật toán **Mutual Information (MI)** để chọn lọc Top 30 features có tính quyết định cao nhất.
   - Tích hợp thuật toán **SMOTE-ENN** và **ADASYN** để cân bằng dữ liệu, xử lý triệt để việc các nhãn tấn công nguy hiểm (Ransomware, Backdoor) chỉ chiếm < 2% tập dữ liệu.
3. **Tầng Mô Hình (Deep Learning Model):**
   - Kiến trúc MLP 8 lớp (512 → 256 → 128 neurons) kết hợp Batch Normalization, Dropout (0.3).
   - Tối ưu hóa quá trình hội tụ với hàm lên lịch học **Cosine Annealing** và kỹ thuật **Class-weighted Loss**.

---

## 🏆 3. Kết quả thực nghiệm

![Accuracy Comparison](results/fig_accuracy_comparison.png)

### 3.1. Tái hiện trên TON_IoT (Reproduce — Binary: Normal vs Attack)
Huấn luyện CyberDetect-MLP trên bộ dữ liệu **TON_IoT** (~460K records, phân loại nhị phân Normal/Attack). Chạy thống kê 10 lần (10 seeds) để đánh giá độ ổn định:

| Metric | Kết quả (trung bình ± std) |
|--------|---------------------------|
| Accuracy | **70.95% ± 15.22%** |
| Precision | 73.91% ± 8.18% |
| Recall | 80.39% ± 8.71% |
| ROC-AUC | **97.96% ± 1.51%** |

**Per-class (lần chạy tốt nhất — Accuracy 76.85%):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.49 | 1.00 | 0.66 | 8,408 |
| Attack (1) | 1.00 | 0.70 | 0.83 | 29,687 |
| **macro avg** | **0.74** | **0.85** | **0.74** | 38,095 |

> Kết quả cho thấy mô hình có ROC-AUC rất cao (~98%), nhưng Accuracy dao động mạnh do mất cân bằng dữ liệu nghiêm trọng (Normal chiếm ~22%, Attack chiếm ~78%). Kết quả tốt nhất đạt Accuracy 76.85%, trung bình 3 lần chạy chính đạt 74.07%.

**So sánh Baseline (Table 3):**

![Baseline Models Comparison](results/fig_baseline_models.png)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Random Forest | 99.99% | 99.98% | 99.98% | 99.98% | 100.00% |
| XGBoost | 100.00% | 99.99% | 100.00% | 100.00% | 100.00% |
| Vanilla MLP | 98.27% | 97.78% | 97.17% | 97.47% | 96.56% |
| **CyberDetect-MLP** | **73.56%** | **72.72%** | **82.99%** | **71.04%** | **95.65%** |

> Lưu ý: Đây là kết quả thực nghiệm tái lập (Reproduction) theo protocol nghiêm ngặt không có Data Leakage. Kết quả cho thấy các Tree-based models (RF, XGBoost) đạt hiệu năng cực cao, không tái lập được kết luận "CyberDetect-MLP vượt trội hơn XGBoost/RF" như trong bài báo gốc.

**Ablation Study (Table 5):**

![Ablation Study](results/fig_ablation_study.png)

| Biến thể | Accuracy |
|----------|----------|
| Full Model | 93.41% |
| Bỏ BatchNorm | 98.80% |
| Bỏ Dropout | 72.63% |
| Bỏ Scheduler | 93.81% |

### 3.2. So sánh các phương pháp cân bằng dữ liệu trên TON_IoT
Thí nghiệm so sánh 3 chiến lược Resampling trên TON_IoT (binary: Normal vs Attack):

![Resampling Comparison](results/fig_resampling_comparison.png)

| Phương pháp | Accuracy | Precision | Recall | F1 | Thời gian Resampling |
|-------------|----------|-----------|--------|-----|---------------------|
| Baseline (không resampling) | **99.96%** | 1.00 | 1.00 | 1.00 | — |
| SMOTE-ENN | 99.89% | 1.00 | 1.00 | 1.00 | 35.78s |
| **ADASYN** | **99.96%** | 1.00 | 1.00 | 1.00 | **4.62s** |

> **Nhận xét:** Cả 3 phương pháp đều đạt ~99.9%+ trên binary classification. ADASYN vượt trội về tốc độ (4.62s vs 35.78s) với kết quả tương đương SMOTE-ENN. Trên bài toán binary với dữ liệu đã qua tiền xử lý Spark, mô hình đã học tốt ngay cả khi không cần resampling.

### 3.3. Cross-Dataset — Đánh giá tổng quát hóa

![Zero-shot vs Retrain](results/fig_zeroshot_vs_retrain.png)

| Dataset | Phương pháp | Accuracy | Precision | Recall | F1 |
|---------|------------|----------|-----------|--------|-----|
| UNSW-NB15 | Zero-shot Transfer | 32.67% | 0.43 (macro) | 0.48 | 0.28 |
| UNSW-NB15 | **Retrain + SMOTE-ENN** | **94.23%** | Normal 0.90 / Attack 0.96 | Normal 0.92 / Attack 0.95 | 0.93 (macro) |
| BoT-IoT | Zero-shot Transfer | 31.64% | 0.50 | 0.60 | 0.24 |
| BoT-IoT | **Retrain + SMOTE-ENN** | **97.58%** | 0.50 | 0.90 | 0.50 |

> **Kết luận:** Chiến lược Retrain + SMOTE-ENN đã khôi phục hiệu năng từ ~32% lên **94-97%** trên cả hai bộ dữ liệu ngoài, cho thấy khả năng thích nghi (adaptability) của framework sau khi retraining kết hợp cân bằng dữ liệu. Mô hình không có khả năng Zero-shot generalization mạnh.

### 3.4. Đánh giá Per-class F1 và Xử lý Mất cân bằng (Multiclass 10 lớp)
Tiến hành thực nghiệm trên bộ dữ liệu gốc của TON_IoT với đầy đủ 9 loại tấn công + Normal (tổng 10 lớp):

| Phương pháp | Accuracy | Thời gian |
|-------------|----------|-----------|
| Multiclass Baseline | 95.51% | 74.99s |
| Multiclass + SMOTE-ENN | 94.12% | 108.97s |

**Hiệu ứng của SMOTE-ENN lên F1-Score của các lớp thiểu số (Minority):**

| Loại tấn công | F1 Trước (Baseline) | F1 Sau (SMOTE-ENN) | Chênh lệch |
|--------------|---------------------|--------------------|-----------|
| **Backdoor** | 0.9555 | 0.9563 | 🟢 +0.0008 |
| **DoS** | 0.7977 | 0.8050 | 🟢 +0.0073 |
| **MITM** | 0.9992 | 0.9988 | 🔴 -0.0004 |
| **Injection**| 0.7820 | 0.4792 | 🔴 -0.3028 |

> **Kết luận:** Trái với giả định phổ biến, SMOTE-ENN không phải luôn mang lại lợi ích cho mọi lớp thiểu số. Nó cải thiện tốt cho `Backdoor` và `DoS`, nhưng lại làm giảm mạnh hiệu suất nhận diện `Injection` (giảm 0.3) do loại bỏ quá nhiều mẫu ở biên quyết định (decision boundary).

### 3.5. Benchmark Spark Pipeline
- Thời gian tiền xử lý Spark (local[*]): **3.597 giây** cho 190,474 records.
- Suy luận realtime: **0.05 ms/mẫu** (~19,672 sự kiện/giây throughput).

---

## 🗂️ 4. Cấu trúc thư mục mã nguồn

```text
Nhom28_CyberDetect_MLP_Final/
│
├── src/                                  # Hệ thống mã nguồn lõi
│   ├── config/                           # Siêu tham số (TOP_K=30, EPOCHS=100)
│   ├── model/                            # Kiến trúc MLP, Trainer, Evaluator
│   ├── modules/                          # SMOTE-ENN, ADASYN, Data Loaders
│   ├── preprocessing/                    # Spark pipeline, Feature Selection
│   └── pipeline/                         # Luồng dữ liệu realtime
│
├── scripts/                              # Các kịch bản thực thi chính
│   ├── main.py                           # Huấn luyện Baseline TON_IoT
│   ├── ablation_study.py                 # Đánh giá từng thành phần MLP
│   ├── baseline_comparison.py            # So sánh RF, XGBoost, Vanilla MLP
│   ├── retrain_resampling.py             # Cân bằng dữ liệu SMOTE-ENN
│   ├── retrain_cross_dataset.py          # Kiểm chứng Domain Shift
│   └── benchmark_latency.py              # Đo hiệu năng Spark
│
├── notebooks/                            # Sổ tay kết quả (Jupyter)
│   ├── 01_Reproduce_TON_IoT(_kq).ipynb   # Tái hiện bài báo gốc
│   ├── 02_Resampling_SMOTE_ENN(_kq).ipynb# Cân bằng dữ liệu
│   ├── 03_Retrain_UNSW_NB15_kq.ipynb     # Retrain trên UNSW-NB15
│   ├── 04_ZeroShot_BoT_IoT(_kq).ipynb    # Zero-shot trên BoT-IoT
│   ├── 05_Retrain_BoT_IoT(_kq).ipynb     # Retrain trên BoT-IoT
│   ├── 06_Benchmark_Spark(_kq).ipynb     # Benchmark hiệu năng
│   └── 07_ClassWeight_PerClass_kq.ipynb  # Đánh giá Per-class F1 10 Lớp
│
├── results/                              # Thư mục chứa các biểu đồ báo cáo
│
└── README.md                             # File này
```

---

## 🚀 5. Hướng dẫn sử dụng (How to Run)

**Môi trường yêu cầu:**
- Python 3.9 - 3.11 (Không dùng 3.12 để tương thích TensorFlow 2.13).
- Java JDK 17 & Apache Spark 3.4.x hoặc 3.5.x.
- (Windows) Bắt buộc có Hadoop winutils.

**Cài đặt thư viện:**
```bash
pip install pyspark tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib shap seaborn
```

**Thực thi:**
```bash
# Huấn luyện Baseline trên TON_IoT
python scripts/main.py

# Tái huấn luyện Cross-dataset (BoT-IoT)
python scripts/retrain_cross_dataset.py

# Đo hiệu năng Apache Spark
python scripts/benchmark_latency.py
```

> **Lưu ý:** Các notebook trong thư mục `notebooks/` được thiết kế để chạy trực tiếp trên Google Colab (có sẵn mount Google Drive). File có hậu tố `_kq` chứa kết quả đã chạy sẵn.
