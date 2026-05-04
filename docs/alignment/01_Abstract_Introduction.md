# Äá»‘i Chiáº¿u Pháº§n 1: Abstract & Introduction (Paper L1â€“L121)

> File nÃ y Ä‘á»‘i chiáº¿u **tá»«ng cÃ¢u claim** trong Abstract vÃ  Introduction cá»§a bÃ i bÃ¡o vá»›i code hiá»‡n táº¡i trong `Nhom28_CyberDetect_MLP_Final`.

---

## 1.1 Title & Authors (L1â€“L8)

| Ná»™i dung paper | Khá»›p? | Vá»‹ trÃ­ code | Ghi chÃº |
|---|---|---|---|
| "CyberDetect MLP a big data enabled optimized deep learning framework for scalable cyberattack detection in IoT environments" | âœ… TÃªn project | `README.md`, tÃªn thÆ° má»¥c project | â€” |
| Talluri Upender et al. | N/A | KhÃ´ng cáº§n code | ThÃ´ng tin tÃ¡c giáº£ |

---

## 1.2 Abstract (L10â€“L30)

| # | Claim trong paper (dÃ²ng) | Khá»›p? | File/dÃ²ng code | Chi tiáº¿t |
|---|---|---|---|---|
| 1 | "Apache Spark for distributed ingestion and preprocessing" (L17â€“18) | âœ… ÄÃ£ triá»ƒn khai | `bigdata/spark/preprocess_from_hdfs.py`, `bigdata/docker-compose.yml` | Spark standalone cluster (1 master + 4 workers) + Kafka + HDFS |
| 2 | "Mutual informationâ€“based feature selection" (L18) | âœ… Khá»›p | `src/preprocessing/feature_selector.py` L80, `scripts/paper_aligned_reproduction.py` L217 | `mutual_info_classif` sklearn |
| 3 | "multi-layer perceptron (MLP) with batch normalization, dropout, and cosine annealing scheduling" (L18â€“19) | âœ… Khá»›p | `src/model/cyberdetect_mlp.py` L4â€“12, `scripts/paper_aligned_reproduction.py` L112â€“125 | 512-256-128 + BN + Dropout(0.3) + cosine |
| 4 | "explainable AI (XAI) module â€¦ utilizing Grad-CAM and SHAP" (L20â€“21) | âš ï¸ Sáº½ bá»• sung | `notebooks/04_Explainable_AI_SHAP_IG.ipynb`, `notebooks/08_Explainable_AI.ipynb` | SHAP Ä‘Ã£ cÃ³. Grad-CAM sáº½ triá»ƒn khai dáº¡ng gradient-based feature attribution (paper L1253 tá»± nháº­n Grad-CAM háº¡n cháº¿ cho MLP) |
| 5 | "full TON_IoT dataset" (L22) | âœ… Khá»›p | `scripts/paper_aligned_reproduction.py` L194 | Load toÃ n bá»™ CSV |
| 6 | "outperforms the baselines of Random Forest, XGBoost, and vanilla MLP" (L22â€“23) | âœ… Cáº¥u trÃºc khá»›p | `scripts/paper_aligned_reproduction.py` L286â€“302 | RF, XGBoost, Vanilla MLP, CyberDetect-MLP Ä‘á»u cÃ³ |
| 7 | "accuracy of 98.87% and a ROC-AUC of 99.10%" (L23) | âš ï¸ Chá» thá»±c nghiá»‡m | â€” | Con sá»‘ nÃ y lÃ  claim paper; reproduction sáº½ report káº¿t quáº£ thá»±c |
| 8 | "Ablation studies and explainability evaluations" (L23â€“24) | âœ… Khá»›p | `scripts/paper_aligned_reproduction.py` L317â€“338 (ablation), notebooks 04/08 (XAI) | â€” |
| 9 | "publicly available at https://github.com/upender0123/CyberDetect-MLP" (L29â€“30) | âœ… ÄÃ£ clone | `CyberDetect-MLP/` (thÆ° má»¥c gá»‘c) | Source gá»‘c Ä‘Ã£ dÃ¹ng Ä‘á»ƒ Ä‘á»‘i chiáº¿u |

---

## 1.3 Keywords (L32)

| Keyword | CÃ³ trong project? | Ghi chÃº |
|---|---|---|
| IoT security | âœ… | README.md, docstrings |
| Cyberattack detection | âœ… | TÃªn project |
| Big data analytics | âœ… | Docker: Kafka + HDFS + Flume + Spark (1M+4W) |
| Deep learning | âœ… | TensorFlow/Keras MLP |
| Intrusion detection system | âœ… | README.md |

---

## 1.4 Introduction Body (L34â€“L120)

### Bá»‘i cáº£nh IoT & IDS (L34â€“69)
- **Ná»™i dung**: mÃ´ táº£ thÃ¡ch thá»©c IoT, háº¡n cháº¿ IDS truyá»n thá»‘ng.
- **Cáº§n code?**: KhÃ´ng. ÄÃ¢y lÃ  pháº§n literature review / motivate.
- **Tráº¡ng thÃ¡i**: N/A (khÃ´ng cáº§n triá»ƒn khai code).

### Contributions List (L85â€“L113)

| # | Contribution claim | Khá»›p? | Chi tiáº¿t code |
|---|---|---|---|
| C1 | "Big dataâ€“enabled IDS: Kafka/Flume + HDFS + Spark" (L95â€“97) | âœ… ÄÃ£ triá»ƒn khai | `bigdata/`: Kafka producer, Flume agent+config, HDFS NameNode/DataNode, Spark 1M+4W. Pipeline: CSVâ†’Kafkaâ†’HDFSâ†’Sparkâ†’Parquetâ†’MLP |
| C2 | "Custom 8-layer MLP with BN, Dropout, cosine annealing" (L99â€“101) | âœ… Khá»›p | `cyberdetect_mlp.py`: Input + Dense(512)+BN+Dropout + Dense(256)+BN+Dropout + Dense(128)+BN+Dropout + Output = 8 layers |
| C3 | "Mutual Information-based feature selection" (L103â€“105) | âœ… Khá»›p | `feature_selector.py` L80, `paper_aligned_reproduction.py` L217 |
| C4 | "Grad-CAM and SHAP interpretability" (L107â€“109) | âš ï¸ Sáº½ bá»• sung | SHAP Ä‘Ã£ cÃ³. Sáº½ triá»ƒn khai gradient-based attribution (tÆ°Æ¡ng Ä‘Æ°Æ¡ng Grad-CAM cho MLP, Ä‘Ãºng theo paper L1253) |
| C5 | "accuracy of 98.87% on TON_IoT" (L111â€“113) | âš ï¸ Chá» thá»±c nghiá»‡m | Reproduction sáº½ bÃ¡o cÃ¡o káº¿t quáº£ thá»±c táº¿ |

### Paper Structure Overview (L115â€“L120)
- **Ná»™i dung**: mÃ´ táº£ cáº¥u trÃºc cÃ¡c section.
- **Cáº§n code?**: KhÃ´ng.
- **Tráº¡ng thÃ¡i**: N/A.

---

## Tá»•ng káº¿t pháº§n 1

| TiÃªu chÃ­ | Sá»‘ claim | Khá»›p âœ… | Sáº½ triá»ƒn khai âš ï¸ | KhÃ´ng khá»›p âŒ | N/A |
|---|---|---|---|---|---|
| Abstract | 9 | 6 | 2 | 0 | 1 |
| Keywords | 5 | 5 | 0 | 0 | 0 |
| Contributions | 5 | 3 | 2 | 0 | 0 |
| **Tá»•ng** | **19** | **14** | **4** | **0** | **1** |

CÃ¡c má»¥c âš ï¸ cÃ²n láº¡i: Grad-CAM sáº½ bá»• sung, chá» káº¿t quáº£ thá»±c nghiá»‡m.

