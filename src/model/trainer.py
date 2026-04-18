"""
CyberDetect-MLP Model Trainer.

Paper references:
  - Algorithm 3 (line 835): Model training and optimization
  - Line 775-778: class-weighted cross-entropy + mild oversampling (SMOTE)
  - Line 762-765, Eq.6: Cosine annealing LR schedule
  - Table 2 (line 989): epochs=100, patience=10, batch_size=64
  - Line 990: "Each experiment is repeated three times"
  - Fig 5 (line 1071): Training/validation accuracy over epochs
  - Fig 6 (line 1076): Training/validation loss over epochs
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd, numpy as np, math, os
from .cyberdetect_mlp import build_model
from config import settings


def cosine_annealing(epoch, lr_min=1e-5, lr_max=1e-3, T=None):
    """
    Paper Eq. 6 (line 762-765): Cosine annealing learning rate schedule.
    α_t = α_min + 0.5 * (α_max - α_min) * (1 + cos(π * t / T))
    T = total training epochs
    """
    if T is None:
        T = settings.EPOCHS
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / T))


def train_model(train_csv, val_csv, model_path):
    """
    Train CyberDetect-MLP with all paper-aligned techniques.

    Paper line 774-778:
      (i)  class-weighted categorical cross-entropy loss
      (ii) mild oversampling of minority classes (SMOTE)
    """
    train_df = pd.read_csv(train_csv)
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    num_classes = len(le.classes_)

    # --- Paper line 776: "some mild oversampling of the minority classes" ---
    # Apply SMOTE to training data BEFORE fitting model
    print("[WAIT] Dang ap dung SMOTE (mild oversampling) cho du lieu huan luyen...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"[OK] Da ap dung SMOTE. Kich thuoc tap train: {X_train.shape[0]} samples")

    # --- Paper line 775: "class-weighted categorical cross-entropy loss" ---
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    model = build_model(X_train.shape[1], num_classes)
    model.compile(
        optimizer=Adam(learning_rate=settings.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Paper Eq.6: Cosine annealing, T = total epochs ---
    lr_sched = LearningRateScheduler(lambda e: cosine_annealing(e, T=settings.EPOCHS))

    # --- Paper Table 2: patience=10, monitor val_loss ---
    early = EarlyStopping(
        patience=settings.PATIENCE,
        monitor='val_loss',
        restore_best_weights=True
    )

    # 80/20 outer split — use validation_split for EarlyStopping
    if val_csv and os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        X_val = val_df.drop('label', axis=1).values
        y_val = le.transform(val_df['label'].values)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=[lr_sched, early],
            class_weight=class_weight_dict,
            verbose=2
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_split=settings.VALIDATION_SPLIT,
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=[lr_sched, early],
            class_weight=class_weight_dict,
            verbose=2
        )

    os.makedirs(model_path, exist_ok=True)
    model.save(f"{model_path}/cyberdetect_mlp.h5")
    print("[OK] Model da huan luyen va duoc luu.")
    return model, le, history


def plot_training_history(history, output_dir="results/training"):
    """
    Generate Fig 5 and Fig 6 from paper.

    Paper references:
      - Fig 5 (line 1048-1051): Training and validation accuracy over epochs
        "Both curves show a consistent upward trend, converging near 98.87%"
      - Fig 6 (line 1053-1056): Training and validation loss over epochs
        "training loss reducing from 1.5 to below 0.2"
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    epochs_range = range(1, len(history.history['accuracy']) + 1)

    # --- Fig 5: Accuracy (line 1071) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history.history['accuracy'], 'b-o',
             label='Training Accuracy', markersize=3)
    plt.plot(epochs_range, history.history['val_accuracy'], 'r-s',
             label='Validation Accuracy', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy of CyberDetect-MLP (Fig 5)',
              fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_accuracy.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 5 da duoc luu tai {output_dir}/fig5_accuracy.png")

    # --- Fig 6: Loss (line 1076) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history.history['loss'], 'b-o',
             label='Training Loss', markersize=3)
    plt.plot(epochs_range, history.history['val_loss'], 'r-s',
             label='Validation Loss', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss of CyberDetect-MLP (Fig 6)',
              fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_loss.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 6 da duoc luu tai {output_dir}/fig6_loss.png")
