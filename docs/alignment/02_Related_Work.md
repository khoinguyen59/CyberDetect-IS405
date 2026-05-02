# Đối Chiếu Phần 2: Related Work (Paper L122–L497)

> File này đối chiếu phần **Related Work** (Section 2) của bài báo. Phần này chủ yếu là literature review, không yêu cầu triển khai code trực tiếp. Mục đích đối chiếu: đảm bảo project không overclaim so với những gì bài báo viết.

---

## 2.1 Tổng quan Related Work (L122–L128)

| Nội dung paper | Cần code? | Trạng thái |
|---|---|---|
| "reviews existing approaches in cyberattack detection for IoT environments" | Không | N/A — Literature review |
| "establishes the research gap that motivates the development of the proposed CyberDetect-MLP" | Không | N/A |

---

## 2.2 ML/DL for Cyberattack Detection (L129–L214)

Phần này review 22 nghiên cứu liên quan (ref [1]–[22]). Không yêu cầu code.

| Nội dung tóm tắt | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| Review SVM, RF, k-NN, CNN, RNN, LSTM | Không | Chỉ liệt kê |
| Ref [1] Achuthan: blockchain, smart grids | Không | — |
| Ref [2] Nadhir: RL healthcare IoT | Không | — |
| Ref [3] Lai: ensemble ML, Bayesian | Không | — |
| Ref [4] Muthubalaji: AEFS-KENN 99.5% | Không | — |
| Ref [5] Hussen: FSBDL 99.93% | Không | — |
| Ref [6] Ahmad: big data cybersecurity | Không | — |
| Ref [7] Wylde: blockchain, ML, big data | Không | — |
| Ref [8] Bravos: CNN-LSTM 99.52% CICIDS2017 | Không | — |
| Ref [9] Sarker: CNN-LSTM 99.52% | Không | — |
| Ref [10] Ferrag: federated DL | Không | — |
| Ref [11] Elsisi: XGBoost GIS | Không | — |
| Ref [12] Panda: ML/DL IoT botnets | Không | — |
| Ref [13] Usman: dynamic malware analysis | Không | — |
| Ref [14] Yahyaoui: Spark Streaming, Flink | Không | — |
| Ref [15] Latif: DnRaNN 99.14% | Không | — |
| Ref [16] Liu: PSO-LightGBM | Không | — |
| Ref [17] Mishra: IoT DDoS survey | Không | — |
| Ref [18] Vitorino: IoT-23 dataset | Không | — |
| Ref [19] Sarhan: NetFlow, SHAP | Không | — |
| Ref [20] Awotunde: hybrid AI 99.75% | Không | — |
| Ref [21] Roy: B-Stacking lightweight ML | Không | — |
| Ref [22] Abbas: ensemble IDS | Không | — |

---

## 2.3 IoT-Specific IDS Techniques (L216–L275)

Review ref [23]–[37]. Không yêu cầu code.

| Nội dung tóm tắt | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| Ref [23] Ravi: GRU-based SDN-IoT | Không | — |
| Ref [24] Ajay Kumar: NBIPS | Không | — |
| Ref [25] Saheed: ML-IDS 99.9% | Không | — |
| Ref [26] Pampapathi: Filtered DL TON_IoT 96.12% | **Tham khảo** | Cùng dataset TON_IoT, baseline comparison tham chiếu |
| Ref [27] Jayalaxmi: IDS/IPS survey | Không | — |
| Ref [28] Asif: MR-IMID MapReduce 97.6% | Không | — |
| Ref [29] Mohy-Eddine: RF IIoT 99.99% | Không | — |
| Ref [30] Wang: Res-TranBiLSTM 99.56% | Không | — |
| Ref [31] Huo: IoT cloud IDS | Không | — |
| Ref [32] Zohourian: IoT-PRIDS | Không | — |
| Ref [33] Sarhan: PCA/AE/LDA NIDS | Không | — |
| Ref [34] Kaushik: TLBO-IDS | Không | — |
| Ref [35] Singh: SecureFlow SDN | Không | — |
| Ref [36] Fares: TFKAN 99.96% | Không | Cùng domain nhưng khác model |
| Ref [37] Ragab: CNN-DBN 99.21% | Không | — |
| "CyberDetect-MLP provides a bridging architecture" (L274) | Không | Motivation statement |

---

## 2.4 Big Data Frameworks for Security Analytics (L277–L334)

| Nội dung tóm tắt | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| Ref [38] Aboalela: DDoS DL 99.52% | Không | — |
| Ref [39] George: FL IoT IDS | Không | — |
| Ref [40] Chen: SICNN IoT | Không | — |
| Ref [41] Hnamte: DCNNBiLSTM | Không | — |
| Ref [42] Alkhonaini: Blockchain+CNN | Không | — |
| Ref [43] Rajathi: Hybrid Learning 99.98% | Không | — |
| Ref [44] Tsimenidis: DL IoT survey | Không | — |
| Ref [45] Tran: IoT-DNN CNC | Không | — |
| Ref [46] Abdalzaher: ML/IoT | Không | — |
| Ref [47] Lopez: EVL+SCARGC | Không | — |
| Ref [48] Hnamte: LSTM-AE 99.99% | Không | — |
| Ref [49] Du: NIDS-CNNLSTM | Không | — |

---

## 2.5 Model Interpretability and Explainability in IDS (L336–L378)

| Nội dung tóm tắt | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| "Grad-CAM, SHAP, LIME, saliency maps" (L338) | **Có** | Project triển khai SHAP. Grad-CAM paper tự nhận hạn chế (L1503) |
| Ref [50]–[60]: các nghiên cứu XAI | Không | Literature review |
| "Integrated Gradients (ref 94) and SHAP (ref 95)" (L1251) | **Có** | SHAP: `notebooks/04_XAI_Paper_Model.ipynb` |

---

## 2.6 Benchmarking & Evaluation Using Public Datasets (L380–L420)

| Nội dung | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| "NSL-KDD, CIC-IDS2017, UNSW-NB15, and TON_IoT" (L381–382) | **Có** | Project dùng TON_IoT chính. Cross-dataset với UNSW-NB15 trong extension |
| Ref [61]–[68] | Không | Literature |

---

## 2.7 Optimization Techniques & Hybrid Models (L431–L497)

| Nội dung | Có ảnh hưởng code? | Ghi chú |
|---|---|---|
| "ensemble learning, neural architecture optimization" (L432) | Không trực tiếp | CyberDetect-MLP không phải ensemble |
| Ref [74]–[88], [89]–[92] | Không | Literature |
| "gap in big data pipeline + explainable DL" (L469–470) | Khớp motivation | — |

---

## 2.8 System Architecture Diagram (L480–L481)

| Nội dung paper | Khớp? | Ghi chú |
|---|---|---|
| "Fig. 1. system architecture of the proposed big data-enabled cyberattack detection framework" | ⚠️ Bán phần | Kiến trúc tổng trong paper có Kafka/HDFS/MongoDB. Project triển khai phần MLP + Spark preprocessing + XAI. Kiến trúc phân tán ghi "paper architecture only" |

---

## Tổng kết phần 2

Phần Related Work **không yêu cầu triển khai code**. Toàn bộ là literature review (88 references). Project hiện tại:
- ✅ Tham chiếu đúng dataset TON_IoT
- ✅ Triển khai XAI (SHAP) theo tinh thần bài báo
- ⚠️ Không thể triển khai so sánh trực tiếp với tất cả 88 nghiên cứu — đây là mong đợi bình thường
- ❌ Kiến trúc tổng (Fig. 1) với Kafka/HDFS/MongoDB không được triển khai thật

| Tiêu chí | Tổng claims | N/A | Có ảnh hưởng code | Khớp ✅ | Bán phần ⚠️ |
|---|---|---|---|---|---|
| Literature review refs | 88 | 85 | 3 | 2 | 1 |
| Architecture claims | 1 | 0 | 1 | 0 | 1 |
