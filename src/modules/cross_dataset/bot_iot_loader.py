import pandas as pd
import numpy as np
import os

def load_and_preprocess_botiot(data_path):
    print(f"Loading BoT-IoT from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop identifier columns if they exist
    cols_to_drop = ['pkSeqID', 'seq']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Label is 'attack'
    y = df['attack'].values
    df = df.drop(columns=['attack', 'category', 'subcategory'], errors='ignore')
    
    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
        
    df = df.fillna(0)
    X = df.values.astype(np.float32)
    
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y
