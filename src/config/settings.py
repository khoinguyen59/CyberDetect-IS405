import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw")
PROC_PATH = os.path.join(DATA_DIR, "processed")
FEATURE_PATH = os.path.join(DATA_DIR, "features")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models")

SPARK_APP = "CyberDetectMLP"
SPARK_MASTER = "local[*]"

TEST_SIZE = 0.20          # Paper line 1021: "An 80/20 stratified train–test split"
VAL_SIZE = 0.0            # Validation handled via validation_split in model.fit()
TOP_K_FEATURES = 30       # Paper Table 2: "top-30 features selected" (line 982, 1005)
BATCH_SIZE = 64
EPOCHS = 100              # Paper Table 2: "Max 100 epochs" (line 989, 1010)
LEARNING_RATE = 0.001
PATIENCE = 10             # Paper Table 2: "patience = 10" (line 989, 1010)
VALIDATION_SPLIT = 0.2    # For EarlyStopping inside model.fit() when using 80/20 outer split
