# Paper-Code Alignment Checklist

This checklist separates paper-aligned reproduction from project extensions.

## Reproduction Protocol Chosen

- Dataset: TON_IoT CSV.
- Main target: binary `label` (Normal vs Attack), matching the core Table 3 IDS comparison.
- Split: 80/20 stratified train-test split, because the experimental setup section states 80/20 even though the method section also mentions 70/15/15.
- Leakage control: split first; fit categorical preprocessing, MI top-30 and MinMaxScaler on training data only.
- Model: MLP 512-256-128 with ReLU, BatchNorm, Dropout 0.3, Adam LR=0.001, batch size 64, cosine annealing, max 100 epochs, early stopping patience 10, class_weight.

## Alignment Matrix

| Paper claim | Project status | Note |
|---|---|---|
| TON_IoT experiment | Matched | Core reproduction uses TON_IoT. |
| 80/20 stratified split | Matched | Chosen over the contradictory 70/15/15 statement. |
| MI top-30 | Matched | Fit on train only to avoid leakage. |
| Spark MI | Partially matched | Project uses Spark-assisted handoff/sklearn MI on driver, not full distributed MI. |
| MinMax normalization | Matched | Scaler fit on train only. |
| MLP 512-256-128 | Matched | Implemented in reproduction script. |
| BatchNorm + Dropout 0.3 | Matched | Full CyberDetect-MLP variant. |
| Adam LR 0.001, batch 64 | Matched | Paper Table 2. |
| Cosine annealing | Matched | T equals max epochs. |
| Max 100 epochs, patience 10 | Matched | Early stopping may stop before 100. |
| Class-weighted loss | Matched | Keras class_weight. |
| RF/XGBoost/Vanilla MLP baselines | Matched | Same preprocessing artifacts as CyberDetect-MLP. |
| Ablation study | Matched in logic | Full, NoFeatureSelect, NoBatchNorm, NoDropout, NoScheduler. |
| SHAP | Extension/reproduction-compatible | Run on saved paper-aligned model if needed. |
| Grad-CAM | Not primary | Grad-CAM is not faithful for tabular MLP; paper itself notes this limitation. |
| Kafka/Flume/HDFS/MongoDB | Not implemented | Treated as paper architecture, not reproduced infrastructure. |
| 5-node Spark cluster | Not implemented | Colab/local reproduction only. |
| Claim CyberDetect > RF/XGBoost | To be empirically checked | Report actual reproduction result; do not force paper numbers. |

## Extension Files Excluded From Core Reproduction

- `07_ClassWeight_PerClass*`: multiclass/per-class extension.
- `08_Explainable_AI*`: XAI extension unless re-run on paper-aligned model.
- `09_Modern_Baselines*`: modern neural baselines extension.
- `10_Strong_Tabular_Baselines*`: strong practical tabular baselines extension.
