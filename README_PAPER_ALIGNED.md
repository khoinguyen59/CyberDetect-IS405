# CyberDetect-MLP - Paper-Aligned Run Guide

This is the main source folder to upload to Google Drive. Do not use old `CyberDetect-Phase2` or `CyberDetect-Phase3` folders for the paper reproduction run.

## Google Drive Layout

Upload the whole folder as:

```text
MyDrive/Nhom28_CyberDetect_MLP_Final/
??? data/
?   ??? ton_iot.csv
?   ??? raw/ton_iot.csv
??? notebooks/
??? scripts/
??? src/
??? models/
??? results/
```

The local project folder now includes `data/ton_iot.csv` and `data/raw/ton_iot.csv` copied from the DLL workspace. After uploading the whole folder, Colab should find the dataset without using old Phase folders.

## Main Notebook

Run this notebook on Colab:

```text
notebooks/01_PaperAligned_Reproduction.ipynb
```

It treats `MyDrive/Nhom28_CyberDetect_MLP_Final` as both the source folder and output folder.

## Outputs

The reproduction script writes directly into the same project folder:

```text
Nhom28_CyberDetect_MLP_Final/models/cyberdetect_mlp_paper_aligned.h5
Nhom28_CyberDetect_MLP_Final/models/paper_aligned_metadata.json
Nhom28_CyberDetect_MLP_Final/results/paper_aligned_table3_metrics.csv
Nhom28_CyberDetect_MLP_Final/results/paper_aligned_ablation_metrics.csv
Nhom28_CyberDetect_MLP_Final/results/cyberdetect_classification_report.txt
Nhom28_CyberDetect_MLP_Final/results/cyberdetect_confusion_matrix.csv
```

## Scope

This run is for the paper reproduction only:

- TON_IoT binary `label` task.
- 80/20 stratified split.
- Train-only preprocessing, MI top-30 and MinMax scaling.
- CyberDetect-MLP with class weights, cosine annealing, max 100 epochs and patience 10.
- RF/XGBoost/Vanilla MLP baseline comparison.
- Ablation variants.

The extension notebooks (`07` to `10`) are not required for this paper-aligned rerun.

## Notebook Order

Run these Colab notebooks in order:

```text
notebooks/01_Data_Preprocessing_MI.ipynb
notebooks/02_Table3_Model_Baselines.ipynb
notebooks/03_Ablation_Study.ipynb
notebooks/04_XAI_Paper_Model.ipynb
notebooks/05_Final_Audit_Checklist.ipynb
```

The old extension/result notebooks were archived outside this source folder.
