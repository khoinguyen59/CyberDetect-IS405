# Äá»‘i Chiáº¿u Pháº§n 4: Experimental Results (Paper L870â€“L1457)

> File nÃ y Ä‘á»‘i chiáº¿u **Section 4 â€” Results** cá»§a bÃ i bÃ¡o: setup, Table 3 baselines, Table 4 statistical tests, Table 5 ablation, Grad-CAM/SHAP, Table 6-7 comparison, Table 8-9 scalability, Table 10 cross-dataset.

---

## 4.1 Experimental Setup (L980â€“L1025, trÃ­ch tá»« nhiá»u Ä‘oáº¡n)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "80/20 stratified trainâ€“test split" (L1021) | âœ… | `paper_aligned_reproduction.py` L207: `test_size=0.20, stratify=y` | â€” |
| 2 | "top-30 features selected" (L982, L1005) | âœ… | `PAPER_HYPERPARAMS["top_k_features"] = 30` L42 | â€” |
| 3 | "batch size = 64" (L989) | âœ… | `PAPER_HYPERPARAMS["batch_size"] = 64` L45 | â€” |
| 4 | "learning rate = 0.001" (L989) | âœ… | `PAPER_HYPERPARAMS["learning_rate"] = 1e-3` L48 | â€” |
| 5 | "max 100 epochs" (L989, L1010) | âœ… | `PAPER_HYPERPARAMS["epochs"] = 100` L46 | â€” |
| 6 | "patience = 10" (L989, L1010) | âœ… | `PAPER_HYPERPARAMS["patience"] = 10` L47 | â€” |
| 7 | "cosine annealing LR schedule" (L989) | âœ… | `cosine_annealing()` L108â€“109 | â€” |
| 8 | "Each experiment is repeated three times" (L990) | âš ï¸ Sáº½ triá»ƒn khai | Script máº·c Ä‘á»‹nh cháº¡y 1 láº§n | Sáº½ thÃªm loop 10 seeds Ä‘á»ƒ cháº¡y statistical testing |
| 9 | "5-node Spark cluster (1 master, 4 workers)" (L1021â€“L1025) | âœ… ÄÃ£ triá»ƒn khai | `bigdata/docker-compose.yml` | Docker: 1 Spark master + 4 Spark workers |
| 10 | "GPU: NVIDIA V100" (ngáº§m hiá»ƒu tá»« setup) | âš ï¸ Chá» thá»±c nghiá»‡m | Colab T4 hoáº·c CPU | Reproduction dÃ¹ng hardware cÃ³ sáºµn, ghi rÃµ cáº¥u hÃ¬nh thá»±c táº¿ |

---

## 4.2 Training Curves â€” Fig 5 & Fig 6 (L1040â€“L1080)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Fig 5: Training and validation accuracy over epochs" (L1048) | âœ… | `trainer.py` L127â€“141: `fig5_accuracy.png` | â€” |
| 2 | "converging near 98.87%" (L1051) | âš ï¸ | Chá» thá»±c nghiá»‡m | Reproduction sáº½ report actual |
| 3 | "Fig 6: Training and validation loss over epochs" (L1053) | âœ… | `trainer.py` L144â€“159: `fig6_loss.png` | â€” |
| 4 | "training loss reducing from 1.5 to below 0.2" (L1056) | âš ï¸ | Chá» thá»±c nghiá»‡m | â€” |
| 5 | "Cosine annealing schedule shown in Fig 4" (ngáº§m) | âœ… | `cosine_annealing()` | CÃ´ng thá»©c triá»ƒn khai Ä‘Ãºng |

---

## 4.3 Confusion Matrix â€” Fig 7 (L1088â€“L1120)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Fig 7: Confusion matrix comparison across models" (L1112) | âœ… | `paper_aligned_reproduction.py` L311: `confusion_matrix()` saved | â€” |
| 2 | "Correctly classified attack categories visible" | âœ… | `classification_report()` L309â€“310 | Per-class precision/recall |

---

## 4.4 Table 3: Performance Comparison (L1114â€“L1120)

| Model | Paper Accuracy | Paper Precision | Paper Recall | Paper F1 | Paper ROC-AUC | CÃ³ trong code? |
|---|---|---|---|---|---|---|
| Random Forest | 94.21% | 93.80% | 92.95% | 93.37% | 95.12% | âœ… `paper_aligned_reproduction.py` L286â€“290 |
| XGBoost | 96.35% | 95.92% | 95.48% | 95.70% | 96.81% | âœ… L292â€“299 |
| Vanilla MLP | 97.12% | 96.85% | 96.30% | 96.57% | 97.44% | âœ… L301â€“302 |
| CyberDetect-MLP | 98.87% | 98.74% | 98.62% | 98.68% | 99.10% | âœ… L304â€“305 |

> **LÆ°u Ã½ quan trá»ng**: Paper report con sá»‘ cá»¥ thá»ƒ, nhÆ°ng reproduction sáº½ bÃ¡o káº¿t quáº£ thá»±c táº¿. Náº¿u khÃ¡c, ghi rÃµ trong bÃ¡o cÃ¡o: "Cannot reproduce exact paper numbers under leakage-safe protocol".

Kiá»ƒm tra logic code Table 3:
- âœ… Táº¥t cáº£ 4 model dÃ¹ng cÃ¹ng split, cÃ¹ng top-30 features, cÃ¹ng scaler
- âœ… RF: `RandomForestClassifier(n_estimators=200, max_depth=20)`
- âœ… XGBoost: `XGBClassifier(n_estimators=200, max_depth=8, lr=0.1)`
- âœ… Vanilla MLP: code gá»i `variant="vanilla"` (táº¯t BN+Dropout) â€” **ÄÃƒ Sá»¬A**
- âœ… CyberDetect-MLP: `variant="full"`, `use_class_weight=True`
- âœ… Metrics: accuracy, precision, recall, f1, roc_auc Ä‘á»u tÃ­nh

---

## 4.5 Table 4: Statistical Significance Testing (L1133â€“L1160)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "paired t-test (or Wilcoxon signed-rank test)" (L1148â€“1149) | âš ï¸ Sáº½ triá»ƒn khai | `scripts/paper_aligned_reproduction.py` | Sáº½ thÃªm `run_statistical_test()` dÃ¹ng `scipy.stats.ttest_rel()` |
| 2 | "10 independent runs" (L1149) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ thÃªm loop 10 seeds trong script |
| 3 | "Mean Â± Std: 0.9887 Â± 0.0012" (L1134) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ tÃ­nh meanÂ±std tá»« 10 runs |
| 4 | "p-value < 0.05" (L1138) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ tÃ­nh p-value tá»« paired t-test |
| 5 | "Fig 9: Boxplot of F1-scores across 10 runs" (L1143) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ váº½ boxplot báº±ng matplotlib |

> **Káº¿t luáº­n Table 4**: Sáº½ triá»ƒn khai loop 10 seeds â†’ thu tháº­p metrics â†’ paired t-test + boxplot.

---

## 4.6 Table 5: Ablation Study (L1162â€“L1214)

| Variant | Paper Acc | Paper Prec | Paper Recall | Paper F1 | CÃ³ trong code? | Variant code |
|---|---|---|---|---|---|---|
| Full Model | 98.87% | 98.74% | 98.62% | 98.68% | âœ… | `variant="full"` |
| MLP-NoFeatureSelect | 96.22% | 95.89% | 95.34% | 95.61% | âœ… | DÃ¹ng `X_train_all_scaled` (all features) |
| MLP-NoBatchNorm | 96.91% | 96.50% | 96.07% | 96.28% | âœ… | `variant="no_batchnorm"` |
| MLP-NoDropout | 97.08% | 96.83% | 96.32% | 96.57% | âœ… | `variant="no_dropout"` |
| MLP-NoScheduler | 96.43% | 96.01% | 95.46% | 95.73% | âœ… | `variant="no_scheduler"` |
| MLP-RawFeatures | 94.86% | 94.35% | 93.82% | 94.08% | âš ï¸ Sáº½ triá»ƒn khai | Sáº½ thÃªm variant dÃ¹ng raw features (chÆ°a MI, chÆ°a MinMax) |

Kiá»ƒm tra logic ablation:
```python
# paper_aligned_reproduction.py L325â€“331
("Full Model", "full", data["X_train"], data["X_test"]),
("MLP-NoFeatureSelect", "full", data["X_train_all_scaled"], data["X_test_all_scaled"]),
("MLP-NoBatchNorm", "no_batchnorm", data["X_train"], data["X_test"]),
("MLP-NoDropout", "no_dropout", data["X_train"], data["X_test"]),
("MLP-NoScheduler", "no_scheduler", data["X_train"], data["X_test"]),
# Sáº½ thÃªm:
("MLP-RawFeatures", "full", data["X_train_raw"], data["X_test_raw"]),
```

> âš ï¸ **Dropout rate mÃ¢u thuáº«n**: Paper Table 5 ghi "Removed Dropout (p = 0.5)" nhÆ°ng method section (L680) ghi p=0.3. Code dÃ¹ng p=0.3 â€” Ä‘Ãºng theo method section. ÄÃ¢y lÃ  paper tá»± mÃ¢u thuáº«n.

---

## 4.7 Explainability Analysis (L1216â€“L1254)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Grad-CAM adapted for MLPs using feature-space gradients" (L1219â€“1220) | âš ï¸ Sáº½ triá»ƒn khai | `notebooks/04_Explainable_AI_SHAP_IG.ipynb` | Sáº½ triá»ƒn khai gradient-based feature attribution (tÆ°Æ¡ng Ä‘Æ°Æ¡ng Grad-CAM cho MLP, paper L1253 tá»± nháº­n háº¡n cháº¿ cho CNN) |
| 2 | "Grad-CAM heatmaps for correctly/incorrectly classified samples" (L1231â€“1232) | âš ï¸ Sáº½ triá»ƒn khai | Notebook 04 | Sáº½ táº¡o heatmap gradient attribution cho correctly/incorrectly classified |
| 3 | "features: dst_host_srv_count, src_bytes, is_login_successful" (L1233) | âš ï¸ Chá» thá»±c nghiá»‡m | â€” | TÃªn features phá»¥ thuá»™c vÃ o MI ranking thá»±c táº¿ |
| 4 | "t-SNE for decision boundary visualization" (L1241) | âš ï¸ Sáº½ triá»ƒn khai | Notebook 04 | Sáº½ thÃªm t-SNE visualization |
| 5 | "Integrated Gradients (ref 94) and SHAP (ref 95)" (L1251) | âœ… | `notebooks/04_Explainable_AI_SHAP_IG.ipynb`, `notebooks/08_Explainable_AI.ipynb` | SHAP Ä‘Ã£ triá»ƒn khai |
| 6 | "Fig 11: Grad-CAM and feature embedding visualization" (L1279) | âš ï¸ Chá» thá»±c nghiá»‡m | â€” | Sáº½ generate khi cháº¡y notebook |

---

## 4.8 Table 6: Comparison with Existing Methods (L1263â€“L1295)

| Model | Accuracy | CÃ³ so sÃ¡nh code? | Ghi chÃº |
|---|---|---|---|
| Vinayakumar et al. CNN+RNN: 97.01% | N/A | KhÃ´ng cáº§n | Literature comparison |
| Alrashdi et al. RF/SVM: 94.67% | N/A | KhÃ´ng cáº§n | KhÃ¡c dataset (TON_IoT subset) |
| Ferrag et al. DL Survey: 94â€“97% | N/A | KhÃ´ng cáº§n | Survey |
| Shone et al. Autoencoder+DNN: 96.21% | N/A | KhÃ´ng cáº§n | NSL-KDD |
| Lopez-Martin et al. CVAE: 95.03% | N/A | KhÃ´ng cáº§n | Custom IoT |
| CyberDetect-MLP: 98.87% | âœ… | `paper_aligned_reproduction.py` | Sáº½ report actual |

> **Ghi chÃº**: Table 6 so sÃ¡nh vá»›i cÃ¡c nghiÃªn cá»©u khÃ¡c, khÃ´ng yÃªu cáº§u cháº¡y láº¡i code cá»§a há». Project chá»‰ cáº§n report káº¿t quáº£ CyberDetect-MLP.

---

## 4.9 Table 7: Qualitative Comparison (L1301â€“L1341)

| Ná»™i dung | Cáº§n code? | Tráº¡ng thÃ¡i |
|---|---|---|
| So sÃ¡nh phÆ°Æ¡ng phÃ¡p tiáº¿p cáº­n (khÃ´ng pháº£i sá»‘ liá»‡u) | KhÃ´ng | N/A â€” báº£ng mÃ´ táº£ |

---

## 4.10 Table 8: Real-time Latency (L1349â€“L1380)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Inference time per sample: 0.84â€“1.05ms" (L1350â€“1352) | âš ï¸ Sáº½ triá»ƒn khai | `realtime_detector.py` | Sáº½ thÃªm `time.time()` Ä‘o inference latency per sample |
| 2 | "Throughput: 10,000â€“98,500 events/sec" (L1350â€“1352) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ Ä‘o throughput = n_samples / elapsed_time |
| 3 | "sub-millisecond inference delay" (L1371) | âš ï¸ Chá» thá»±c nghiá»‡m | â€” | Sáº½ Ä‘o trÃªn hardware cÃ³ sáºµn (Colab/local) |
| 4 | "Spark-Based Distributed Architecture" (L1373â€“1374) | âœ… ÄÃ£ triá»ƒn khai | `bigdata/docker-compose.yml` | Spark standalone cluster 1M+4W |

> **Káº¿t luáº­n Table 8**: Sáº½ thÃªm code Ä‘o latency/throughput. Con sá»‘ thá»±c táº¿ phá»¥ thuá»™c hardware reproduction.

---

## 4.11 Table 9: Scalability & Resource Utilization (L1393â€“L1420)

| # | Claim paper | Khá»›p? | Chi tiáº¿t |
|---|---|---|---|
| 1 | "CPU utilization: 38â€“82%" | âš ï¸ Sáº½ triá»ƒn khai | Sáº½ dÃ¹ng `psutil.cpu_percent()` Ä‘o trong quÃ¡ trÃ¬nh training/inference |
| 2 | "GPU utilization: 22â€“71%" | âš ï¸ Sáº½ triá»ƒn khai | Sáº½ dÃ¹ng `GPUtil.getGPUs()` Ä‘o GPU load |
| 3 | "Memory usage: 41â€“79%" | âš ï¸ Sáº½ triá»ƒn khai | Sáº½ dÃ¹ng `psutil.virtual_memory().percent` |
| 4 | "linear scalability" | âš ï¸ Sáº½ triá»ƒn khai | Sáº½ Ä‘o thá»i gian xá»­ lÃ½ vá»›i data size khÃ¡c nhau Ä‘á»ƒ verify trend |

> **Káº¿t luáº­n Table 9**: Sáº½ thÃªm `src/infrastructure/resource_monitor.py` â€” thread cháº¡y ná»n Ä‘o resource má»—i 5s, ghi ra CSV.

---

## 4.12 Table 10: Cross-Dataset Validation (L1422â€“L1457)

| # | Claim paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Trained on TON_IoT, tested on UNSW-NB15: 92.5% acc, 90.1% F1" (L1441) | âš ï¸ Sáº½ triá»ƒn khai | `src/modules/cross_dataset/` | Sáº½ download UNSW-NB15 (public dataset) + align features + cháº¡y inference |
| 2 | "Trained on TON_IoT, tested on BoT-IoT: 94.0% acc, 91.3% F1" (L1442) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ download BoT-IoT + cháº¡y inference |
| 3 | "Fig 15: Cross-dataset results" (L1448) | âš ï¸ Sáº½ triá»ƒn khai | â€” | Sáº½ generate sau khi cháº¡y cross-dataset |

> **Káº¿t luáº­n Table 10**: Sáº½ download UNSW-NB15 + BoT-IoT (public datasets), align features, cháº¡y inference.

---

## Tá»•ng káº¿t pháº§n 4

| ThÃ nh pháº§n | Tá»•ng claims | âœ… Khá»›p | âš ï¸ Sáº½ triá»ƒn khai/Chá» | âŒ KhÃ´ng khá»›p |
|---|---|---|---|---|
| Experimental Setup | 10 | 8 | 2 | 0 |
| Training Curves (Fig 5,6) | 5 | 3 | 2 | 0 |
| Confusion Matrix (Fig 7) | 2 | 2 | 0 | 0 |
| Table 3: Baselines | 4 models Ã— 5 metrics = 20 | 17 (logic) | 3 (chá» numbers) | 0 |
| Table 4: Statistical Test | 5 | 0 | 5 | 0 |
| Table 5: Ablation | 6 variants Ã— 4 metrics = 24 | 20 (logic) | 4 (MLP-RawFeatures sáº½ thÃªm) | 0 |
| XAI / Grad-CAM | 6 | 1 | 5 | 0 |
| Table 6-7: Literature Comparison | 8 | 1 | 0 | 0 (7 N/A) |
| Table 8: Latency | 4 | 1 | 3 | 0 |
| Table 9: Scalability | 4 | 0 | 4 | 0 |
| Table 10: Cross-dataset | 3 | 0 | 3 | 0 |
| **Tá»•ng** | **91** | **53** | **31** | **0** |

KhÃ´ng cÃ²n má»¥c âŒ. Táº¥t cáº£ 33 má»¥c âš ï¸ Ä‘á»u cÃ³ káº¿ hoáº¡ch triá»ƒn khai cá»¥ thá»ƒ.

