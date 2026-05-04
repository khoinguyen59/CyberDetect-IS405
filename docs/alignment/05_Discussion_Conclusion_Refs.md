# Đối Chiếu Phần 5: Discussion, Limitations, Conclusion, References (Paper L1459–L1915)

> File này đối chiếu phần cuối bài báo: Discussion, Limitations, Conclusion, Future Work, Data/Code Availability, và References.

---

## 5.1 Discussion (L1459–L1496)

| # | Claim paper (dòng)                                                      | Cần code?              | Trạng thái                                         | Chi tiết                                                                |
| - | ------------------------------------------------------------------------ | ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| 1 | "Traditional IDS insufficient for novel threats" (L1461–1462)           | Không                  | N/A                                                  | Motivation text                                                          |
| 2 | "CyberDetect-MLP bridges scalability, explainability gaps" (L1467–1468) | Không                  | N/A                                                  | Discussion claim                                                         |
| 3 | "leverages the full TON_IoT dataset" (L1469)                             | Có                     | ✅                                                   | `paper_aligned_reproduction.py` L194: load full CSV                    |
| 4 | "Apache Spark for distributed preprocessing and model training" (L1470)  | ✅ Đã triển khai    | `bigdata/spark/preprocess_from_hdfs.py`, Docker 1M+4W  | Spark standalone cluster                                |
| 5 | "Grad-CAM enables interpretability" (L1472)                              | ⚠️ Sẽ triển khai    | Notebook 04, sẽ bổ sung gradient-based attribution | Paper tự nhận hạn chế cho MLP                                        |
| 6 | "per-class performance metrics" (L1475)                                  | Có                     | ✅                                                   | `classification_report()` trong `paper_aligned_reproduction.py` L309 |
| 7 | "class weighting combined with controlled resampling" (L1484)            | Có                     | ✅                                                   | `class_weight` + SMOTE (trainer.py)                                    |
| 8 | "accuracy of 98.87%" (L1488)                                             | ⚠️ Chờ thực nghiệm | Reproduction sẽ report actual result                | Sẽ biết sau khi chạy                                                  |
| 9 | "Ablation studies and visualization-supported analysis" (L1489–1490)    | Có                     | ✅                                                   | Ablation code L317–338                                                  |

---

## 5.2 Limitations of the Study (L1498–L1505)

| # | Limitation stated by paper                                                                                   | Project address? | Chi tiết                                                                                 |
| - | ------------------------------------------------------------------------------------------------------------ | ---------------- | ----------------------------------------------------------------------------------------- |
| 1 | "not validated on cross-domain datasets such as CIC-IDS2017 or UNSW-NB15" (L1500–1501)                      | ⚠️             | `src/modules/cross_dataset/` exists nhưng là extension, chưa chạy trên CIC-IDS2017 |
| 2 | "current implementation uses offline evaluation; real-time performance under streaming not assessed" (L1502) | ✅             | `bigdata/spark/stream_from_kafka.py` + `realtime_detector.py`                |
| 3 | "Grad-CAM primarily designed for convolutional architectures, limited for MLP" (L1503–1504)                 | ✅ Acknowledged  | Project ghi rõ trong checklist: "Grad-CAM is not faithful for tabular MLP"               |

> **Quan trọng**: Bài báo **TỰ NHẬN** 3 hạn chế trên. Project hiện tại đã acknowledge đúng các hạn chế này.

---

## 5.3 Conclusion and Future Work (L1507–L1528)

| # | Claim paper (dòng)                                       | Cần code?           | Trạng thái                                            |
| - | --------------------------------------------------------- | -------------------- | ------------------------------------------------------- |
| 1 | "fast, scalable, and interpretable IDS framework" (L1508) | Tổng hợp           | ✅ Phần MLP + MI khớp; scalability claim ⚠️         |
| 2 | "big data pipelines (Kafka/Flume-HDFS-Spark)" (L1509)     | ✅ Đã triển khai | `bigdata/`: Kafka + Flume + HDFS + Spark 1M+4W           |
| 3 | "outperforms several standard baselines" (L1512)          | Có                  | ✅ Logic code Table 3 đủ baseline                     |
| 4 | "Grad-CAM and SHAP improved trust" (L1513)                | ⚠️ Sẽ bổ sung    | SHAP có, Grad-CAM sẽ thêm gradient-based attribution |

### Future Work Directions (L1521–L1528)

| Direction                                                     | Project status                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| "validating across CIC-IDS2017, UNSW-NB15" (L1522)            | ⚠️ Sẽ triển khai — download UNSW-NB15/BoT-IoT, chạy cross-dataset |
| "real-time streaming using Apache Kafka" (L1523)              | ✅ Đã triển khai — `bigdata/kafka/produce_toniot.py`, `bigdata/spark/stream_from_kafka.py` |
| "additional XAI methods for non-convolutional models" (L1524) | ✅ SHAP đã dùng                                                      |
| "edge-computing or federated learning" (L1526)                | N/A — paper ghi future work, không yêu cầu triển khai              |

---

## 5.4 Data Availability (L1530–L1531)

| Claim                                             | Trạng thái                      |
| ------------------------------------------------- | --------------------------------- |
| "Data is available with the corresponding author" | N/A — TON_IoT là public dataset |

---

## 5.5 Code Availability (L1536–L1539)

| Claim                                                                  | Trạng thái                            |
| ---------------------------------------------------------------------- | --------------------------------------- |
| "publicly available at https://github.com/upender0123/CyberDetect-MLP" | ✅ Đã clone vào `CyberDetect-MLP/` |

---

## 5.6 References (L1543–L1877)

Bài báo có **98 references** (ref [1] đến [98]). Phần này không yêu cầu code.

| Ref #  | Tác giả                                   | Có ảnh hưởng code? | Ghi chú                                       |
| ------ | ------------------------------------------- | ---------------------- | ---------------------------------------------- |
| 93     | Mohammad et al. — TON_IoT dataset          | ✅                     | Dataset chính được sử dụng               |
| 94     | Sundararajan et al. — Integrated Gradients | ⚠️                   | Paper tham chiếu, code dùng SHAP thay thế   |
| 95     | Lundberg & Lee — SHAP                      | ✅                     | SHAP được triển khai trong notebooks       |
| 96     | Selvaraju et al. — Grad-CAM                | ⚠️                   | Paper dùng nhưng tự nhận hạn chế cho MLP |
| 97–98 | Attribution/relevance propagation           | Không trực tiếp     | —                                             |
| 1–92  | Các nghiên cứu liên quan                | Không                 | Literature only                                |

---

## 5.7 Author Contributions & Declarations (L1878–L1915)

| Nội dung                 | Cần code? | Trạng thái |
| ------------------------- | ---------- | ------------ |
| Author contributions      | Không     | N/A          |
| Competing interests: none | Không     | N/A          |
| Open Access CC BY 4.0     | Không     | N/A          |

---

## Tổng kết phần 5

| Thành phần           | Tổng claims  | ✅ Khớp     | ⚠️ Sẽ triển khai | ❌ Không khớp | N/A          |
| ---------------------- | ------------- | ------------ | -------------------- | --------------- | ------------ |
| Discussion             | 9             | 6            | 3                    | 0               | 0            |
| Limitations            | 3             | 2            | 1                    | 0               | 0            |
| Conclusion             | 4             | 2            | 2                    | 0               | 0            |
| Future Work            | 4             | 2            | 1                    | 0               | 1            |
| Data/Code Availability | 2             | 1            | 0                    | 0               | 1            |
| References             | 98            | 2            | 2                    | 0               | 94           |
| Author/Declarations    | 3             | 0            | 0                    | 0               | 3            |
| **Tổng**        | **123** | **15** | **9**         | **0**     | **99** |

---

# === TỔNG KẾT TOÀN BỘ BÀI BÁO ===

## Bảng Tổng Hợp 5 Phần

| File MD         | Section                    | Tổng claims  | ✅            | ⚠️         | ❌          | N/A           |
| --------------- | -------------------------- | ------------- | ------------- | ------------ | ----------- | ------------- |
| 01              | Abstract & Introduction    | 19            | 14            | 4            | 0           | 1             |
| 02              | Related Work               | 90            | 2             | 2            | 0           | 86            |
| 03              | Methodology                | 91            | 85            | 6            | 0           | 0             |
| 04              | Experimental Results       | 91            | 53            | 31           | 0           | 7             |
| 05              | Discussion/Conclusion/Refs | 123           | 15            | 9            | 0           | 99            |
| **TỔNG** | **Toàn bài**       | **414** | **169** | **52** | **0** | **193** |

### Loại bỏ N/A (literature, refs, author info):

|                                      | Tổng claims cần code | ✅ Khớp            | ⚠️ Sẽ triển khai |                  |
| ------------------------------------ | ---------------------- | ------------------- | -------------------- | ---------------- |
| **Chỉ tính claims có code** | **221**          | **169 (76%)** | **52 (24%)**   | **0 (0%)** |

### Không còn mục ❌ nào. Các mục ⚠️ còn lại chủ yếu là chờ kết quả thực nghiệm:

| Nhóm                                                   | Số mục | Chi tiết                                   |
| ------------------------------------------------------- | -------- | ------------------------------------------- |
| Infrastructure (Kafka/HDFS/Flume/Spark/Parquet)         | 0        | ✅ Đầy đủ thành phần kiến trúc ở mức Docker prototype |
| Statistical testing (10 runs, t-test, boxplot)          | 6        | Sẽ thêm vào script                       |
| Ablation (MLP-RawFeatures)                              | 1        | Sẽ thêm 1 variant                         |
| Scalability (CPU/GPU/Mem monitor)                       | 4        | Sẽ thêm resource monitor                  |
| Latency/Throughput benchmarks                           | 3        | Sẽ thêm timing code                       |
| XAI (Grad-CAM, t-SNE)                                   | 5        | Sẽ bổ sung gradient attribution + t-SNE   |
| Cross-dataset (UNSW-NB15, BoT-IoT)                      | 3        | Sẽ download data + chạy                   |
| Chờ kết quả thực nghiệm (accuracy, loss curves)    | 16       | Sẽ biết sau khi chạy Colab               |
| Paper tự mâu thuẫn (split ratio 70/15/15 vs 80/20)   | 3        | Không thể khớp cả hai — chọn 80/20    |
| Grad-CAM heatmap features (phụ thuộc MI)              | 2        | Chờ MI ranking thực tế                   |
| Hardware (V100 vs Colab T4)                             | 1        | Dùng hardware có sẵn, ghi rõ cấu hình |

> **Giới hạn triển khai**:
> 1. Tất cả thành phần kiến trúc (Kafka, Flume, HDFS, Spark 1M+4W) đã có code và Docker config, nhưng chạy trên **một máy bằng Docker containers**, chưa phải cụm vật lý 5 node như môi trường thực nghiệm của bài báo. Đây là containerized prototype, không phải production cluster.
> 2. Spark container job dùng `MI top-30 emulated/selected`, không phải MI thật bằng sklearn do giới hạn dependency của PySpark trong Alpine.
> 3. Kết quả Table 3 (Accuracy 99.24%) được ghi nhận là kết quả theo **Spark-output protocol**, không phải direct sklearn-MI protocol như bản gốc của paper.
