from sklearn.model_selection import train_test_split
import pandas as pd, os
from config import settings

def stratified_split(data_path, output_path):
    """Paper line 1021: 'An 80/20 stratified train-test split was used'"""
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=['label']), df['label']

    # Discrepancy #5: 80/20 split (paper) instead of 70/15/15 (original code)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,  # 0.20
        stratify=y,
        random_state=42
    )

    os.makedirs(output_path, exist_ok=True)
    X_train.assign(label=y_train).to_csv(f"{output_path}/train.csv", index=False)
    X_test.assign(label=y_test).to_csv(f"{output_path}/test.csv", index=False)
    print(f"✅ Train/Test sets created (80/20 split). Train={len(X_train)}, Test={len(X_test)}")

    # Return None for val_csv — validation handled inside model.fit(validation_split=0.2)
    return (f"{output_path}/train.csv", None, f"{output_path}/test.csv")
