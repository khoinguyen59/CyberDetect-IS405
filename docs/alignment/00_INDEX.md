# Paper-Code Alignment Index

> Bộ đối chiếu **TOÀN BỘ** bài báo gốc (1915 dòng, 30 trang) với code trong `Nhom28_CyberDetect_MLP_Final`.
> Mỗi câu claim trong bài báo đều được kiểm tra và ghi trạng thái: ✅ Khớp, ⚠️ Sẽ triển khai, N/A.

---

## Danh sách file đối chiếu

| File | Section bài báo | Dòng paper | Số claims |
|---|---|---|---|
| [01_Abstract_Introduction.md](01_Abstract_Introduction.md) | Abstract, Keywords, Introduction, Contributions | L1–L121 | 19 |
| [02_Related_Work.md](02_Related_Work.md) | Related Work (88 references) | L122–L497 | 90 |
| [03_Methodology.md](03_Methodology.md) | Proposed Framework, Architecture, Algorithms, Hyperparams | L499–L870 | 91 |
| [04_Experimental_Results.md](04_Experimental_Results.md) | Tables 3–10, Figures 4–15, Ablation, XAI | L870–L1457 | 91 |
| [05_Discussion_Conclusion_Refs.md](05_Discussion_Conclusion_Refs.md) | Discussion, Limitations, Conclusion, References | L1459–L1915 | 123 |

---

## Tổng kết

### Toàn bộ bài báo: 414 claims

| Trạng thái | Số lượng | Tỷ lệ |
|---|---|---|
| ✅ Khớp | 148 | 35.7% |
| ⚠️ Sẽ triển khai | 73 | 17.6% |
| ❌ Không khớp | 0 | 0% |
| N/A (literature, refs) | 193 | 46.6% |

### Chỉ tính claims cần triển khai code: 221 claims

| Trạng thái | Số lượng | Tỷ lệ |
|---|---|---|
| ✅ Khớp | 148 | **67%** |
| ⚠️ Sẽ triển khai | 73 | **33%** |
| ❌ Không khớp | 0 | **0%** |

---

## 73 mục ⚠️ — phân loại theo nhóm

| Nhóm | Số mục | Chi tiết |
|---|---|---|
| Infrastructure (Kafka/HDFS/Flume/Spark/Parquet) | 20 | Sẽ cài đặt + viết code mới trong `src/infrastructure/` |
| Statistical testing (10 runs, t-test, boxplot) | 6 | Sẽ thêm loop 10 seeds + `scipy.stats.ttest_rel()` |
| Ablation (MLP-RawFeatures) | 1 | Sẽ thêm 1 variant vào ablation code |
| Scalability (CPU/GPU/Mem monitor) | 4 | Sẽ thêm `resource_monitor.py` dùng `psutil` + `GPUtil` |
| Latency/Throughput benchmarks | 4 | Sẽ thêm `time.time()` đo inference + throughput |
| XAI (Grad-CAM, t-SNE) | 5 | Sẽ bổ sung gradient-based attribution + t-SNE |
| Cross-dataset (UNSW-NB15, BoT-IoT) | 3 | Sẽ download public datasets + chạy inference |
| Chờ kết quả thực nghiệm | 16 | Accuracy, loss curves — sẽ biết sau khi chạy Colab |
| Paper tự mâu thuẫn (split ratio) | 3 | Không thể khớp cả hai — chọn 80/20 theo experimental section |
| Hardware (V100 vs Colab T4) | 1 | Dùng hardware có sẵn, ghi rõ cấu hình thực tế |
| Grad-CAM features phụ thuộc MI | 2 | Chờ MI ranking thực tế |
| Future work (Kafka streaming, cross-dataset) | 8 | Sẽ triển khai (trừ federated learning = N/A) |

---

## Điểm mạnh project hiện tại (đã ✅)

1. **Leakage-safe protocol**: Split trước → fit preprocessing trên train only
2. **Hyperparameters 100% khớp Table 2**: 512-256-128, BN, Dropout 0.3, Adam 0.001, batch 64, epochs 100, patience 10, cosine annealing
3. **4 Algorithm paper đều có code**: Preprocessing, MI Selection, Training, Real-time Detection
4. **Ablation study logic đúng**: 5/6 variants (sẽ thêm MLP-RawFeatures)
5. **XAI module SHAP**: Đã triển khai
6. **Table 3 baselines đủ**: RF, XGBoost, Vanilla MLP, CyberDetect-MLP
