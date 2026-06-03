# HƯỚNG DẪN CHẠY THỰC NGHIỆM & THỨ TỰ THỰC THI (RUN ORDER)
**Đồ án môn học: Dữ liệu lớn (IS405) - Nhóm 28**
**Đề tài:** Cải tiến Framework phát hiện tấn công mạng IoT trên nền tảng dữ liệu lớn: Mở rộng Cross-dataset, Xử lý mất cân bằng và Giải thích mô hình.

---

## 📂 Cấu trúc thư mục chuẩn bị trên Google Drive

Để đảm bảo toàn bộ pipeline chạy mượt mà không bị lỗi đường dẫn (`PROJECT_PATH`), vui lòng tải toàn bộ thư mục `Nhom28_CyberDetect_MLP_Final` lên Google Drive cá nhân của bạn tại đường dẫn chính xác sau:
```
/content/drive/MyDrive/Nhom28_CyberDetect_MLP_Final
```

Cấu trúc thư mục dữ liệu đầu vào:
* **`data/ton_iot.csv`**: Tệp dữ liệu mạng TON_IoT (~29MB, 211k dòng, 10 lớp `type`) — dùng cho NB01–07, 10.
* **`data/raw/UNSW-NB15/UNSW_NB15_training-set.csv`**: Dữ liệu UNSW-NB15 (Notebook 08).
* **`data/raw/BoT-IoT/UNSW_2018_IoT_Botnet_Final_10_Best.csv`**: Dữ liệu BoT-IoT (Notebook 08).
* **CSE-CIC-IDS2018**: KHÔNG cần chuẩn bị — Notebook 08 **tự tải từ AWS Open Data** về `data/raw/CSE-CIC-IDS2018/` (~6.5GB, ~10 phút) rồi lấy mẫu 500k.

---

## 🛠️ Yêu cầu môi trường phần cứng khuyến nghị (Google Colab)
* **GPU**: **T4 / L4 / A100** đều chạy tốt (model nhỏ nên T4 miễn phí là đủ). ⚠️ **TRÁNH GPU Blackwell / RTX PRO 6000 (compute capability 12.0)** vì TensorFlow hiện chưa hỗ trợ → lỗi `CUDA_ERROR_INVALID_PTX`. NB09 (chỉ suy luận) có thể để CPU.
* **System RAM**: Khuyến nghị bật chế độ **High-RAM (RAM > 12GB)** nếu có gói Colab Pro để tránh tràn bộ nhớ (OOM) khi chạy thuật toán khử nhiễu `SMOTE-ENN` trên tập dữ liệu lớn.
* **Cài đặt thư viện**: Cell cài đặt các thư viện cần thiết (`imbalanced-learn`, `pyarrow`, `joblib`, `psutil`, `tqdm`, `scipy`) đã được chèn sẵn ở đầu mỗi notebook từ 06 đến 10 để đảm bảo tính độc lập.

---

## 🔄 Thứ tự thực thi các Notebook (Run Order)

Do các Notebook hoạt động theo cơ chế kế thừa kết quả tiền xử lý (artifacts) và mô hình đã được huấn luyện, bạn **bắt buộc phải chạy theo đúng thứ tự** sau:

### Giai đoạn 1: Chuẩn bị dữ liệu và huấn luyện mô hình cơ sở
1. **`01_Preprocessing_MI_FeatureSelection.ipynb`**
   * *Nhiệm vụ*: Đọc dữ liệu gốc `data/ton_iot.csv`, chuẩn hóa Min-Max, mã hóa One-hot cho các đặc trưng danh mục, trích chọn **Top-30 đặc trưng** bằng thuật toán Mutual Information (MI) song song.
   * *Kết quả*: Sinh ra các file nén parquet trong `data/colab_processed/` và preprocessor được đóng gói tại `models/colab_preprocessing_metadata.json`.
2. **`02_Table3_Baseline_Comparison.ipynb`**
   * *Nhiệm vụ*: Huấn luyện mô hình đề xuất **CyberDetect-MLP** (kiến trúc 512→256→128, Batch Normalization, Dropout=0.3, Cosine Annealing learning rate) và so sánh với các baselines (Random Forest, XGBoost, Vanilla MLP) trên bài toán phân loại đa lớp (10 lớp tấn công).
   * *Kết quả*: Sinh ra file mô hình huấn luyện chuẩn **`models/colab_cyberdetect_mlp.h5`** và các file so sánh trong thư mục `results/`.

### Giai đoạn 2: Đánh giá chi tiết và Phân tích chuyên sâu
Bạn có thể chạy song song hoặc bất kỳ thứ tự nào các notebook sau khi đã hoàn thành Giai đoạn 1:
* **`03_Table5_Ablation_Study.ipynb`**: Thực hiện nghiên cứu loại bỏ (ablation study) để kiểm chứng vai trò của Batch Normalization và Dropout đối với sự hội tụ của mô hình.
* **`04_Explainable_AI_SHAP_IG.ipynb`**: Trực quan hóa mức độ quan trọng của đặc trưng bằng mô hình giải thích SHAP và Integrated Gradients (phù hợp với bài toán mạng MLP đa lớp).
* **`05_Final_Audit_Checklist.ipynb`**: Quét và lập báo cáo kiểm toán tính đầy đủ của các artifacts kết quả.
* **`06_Class_Imbalance_Resampling.ipynb`**: Chạy script phân tích mất cân bằng dữ liệu 10 lớp, đo lường F1-score thật của từng lớp khi áp dụng và không áp dụng SMOTE-ENN nâng cao đối với các lớp tấn công thiểu số cực đoan (`ransomware`, `backdoor`).
* **`07_Concept_Drift_Analysis.ipynb`**: Đo lường sự suy giảm độ chính xác và dịch chuyển phân phối dữ liệu (Covariate Shift) trên luồng dữ liệu mô phỏng bằng thuật toán kiểm định Kolmogorov-Smirnov (KS-test).
* **`08_Cross_Dataset_Retrain.ipynb`**: Đánh giá khả năng thích nghi miền (Domain Adaptation) trên hai tập dữ liệu UNSW-NB15 và BoT-IoT. Thực hiện retrain kết hợp phân tầng lấy mẫu (stratified sampling) và so sánh trực tiếp với Zero-shot Transfer.
* **`09_Benchmark_Spark_Performance.ipynb`**: Đo lường độ trễ suy luận thời gian thực (inference latency per sample) và khả năng mở rộng phần cứng (CPU, GPU, Memory thông qua `psutil`).
* **`10_Statistical_Significance_Testing.ipynb`**: Huấn luyện lặp lại mô hình với 10 random seeds khác nhau và thực hiện kiểm định t-test cặp (paired t-test) để chứng minh độ ổn định thống kê của mô hình.

---

## ⚠️ Kỳ vọng về kết quả (Academic Insight & Expected Outcomes)

Khi chạy thực nghiệm thật trên Colab, vui lòng lưu ý hai điểm quan trọng sau để đưa vào phần thảo luận của Báo cáo đồ án:

1. **Độ chính xác đa lớp thực tế sẽ đạt khoảng ~70% - 90% (tùy lớp):**
   * *Giải thích*: Có sự khác biệt lớn so với con số 98.87% công bố trong bài báo gốc. Qua quá trình kiểm toán mã nguồn của bài báo gốc, nhóm phát hiện tác giả bài báo gốc công bố làm phân loại đa lớp (Multiclass) nhưng trong mã nguồn thực tế lại cấu hình bài toán nhị phân (Binary - sigmoid + binary_crossentropy). Nhóm đã sửa đổi và thực hiện chuẩn hóa đúng bài toán đa lớp (softmax + categorical_crossentropy) theo đúng Methodology của bài báo. Độ chính xác ~74% là kết quả phản ánh **trung thực, khoa học** của bài toán đa lớp trên tập dữ liệu TON_IoT.
2. **Hiện tượng mất cân bằng cực đoan trong Cross-dataset (Notebook 08):**
   * Tập dữ liệu BoT-IoT sau khi lấy mẫu có tỷ lệ lệch lớp rất cao (chỉ có vài chục mẫu Normal trên hàng trăm nghìn mẫu Attack). Vì vậy độ chính xác (Accuracy) tổng thể sẽ luôn tiệm cận 99.9%. Nhóm khuyến nghị giảng viên đánh giá hiệu năng dựa trên **F1-Score và Recall của lớp Normal** trong bảng báo cáo chi tiết `classification_report`, tránh bị đánh lừa bởi chỉ số Accuracy tổng thể.
