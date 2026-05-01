# -*- coding: utf-8 -*-
"""
CSE-CIC-IDS2018 loader (CICFlowMeter features) - DO THAT, khong sinh du lieu gia.

- Doc NHIEU file CSV (10 ngay, moi ngay 1 loai tan cong khac nhau) tu mot THU MUC.
- Lay mau NGAU NHIEN trai deu qua cac file -> dam bao da dang loai tan cong + tranh skew.
- Xu ly ten cot linh hoat (strip BOM/khoang trang), ep numeric, khu Infinity/NaN.
- Thieu du lieu that -> raise FileNotFoundError (KHONG bia du lieu).
"""
import os
import glob
import pandas as pd
import numpy as np


def load_and_preprocess_ids2018(data_path, sample_rows=500000):
    """
    data_path: THU MUC chua nhieu CSV CSE-CIC, hoac 1 file CSV.
    sample_rows: tong so dong lay mau (ngau nhien, trai deu qua cac file).
    Tra ve: X (float32), y_raw (nhan goc dang chuoi), feature_names (list).
    """
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    elif os.path.isfile(data_path):
        files = [data_path]
    else:
        files = []

    if not files:
        raise FileNotFoundError(
            f"Khong co CSV CSE-CIC tai '{data_path}'. Day la dataset MO RONG (~7GB); "
            "script se thu tu tai tu AWS, hoac dat file that vao thu muc nay. "
            "Khong sinh du lieu gia de tranh ket qua ao."
        )

    per_file = max(1, sample_rows // len(files))
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  [bo qua] {os.path.basename(f)}: {str(e)[:60]}")
            continue
        d.columns = [str(c).strip().replace("﻿", "") for c in d.columns]
        n0 = len(d)
        if n0 > per_file:
            d = d.sample(n=per_file, random_state=42)
        parts.append(d)
        print(f"  {os.path.basename(f)}: lay {len(d):,}/{n0:,} dong")

    if not parts:
        raise RuntimeError("Khong doc duoc file CSV CSE-CIC nao.")

    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Bo cot thoi gian (khong phai dac trung)
    for c in list(df.columns):
        if c.lower() in ("timestamp",):
            df = df.drop(columns=[c])

    # Tim cot nhan
    label_col = next((c for c in df.columns if c.lower() == "label"), df.columns[-1])
    y_raw = df[label_col].values
    X_df = df.drop(columns=[label_col])

    # Ep tat ca dac trung ve numeric, khu Infinity/NaN (CSE-CIC hay co 'Infinity' o Flow Byts/s)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()

    print(f"[OK] Tong mau: {X.shape[0]:,} dong, {X.shape[1]} dac trung (label='{label_col}').")
    return X, y_raw, feature_names
