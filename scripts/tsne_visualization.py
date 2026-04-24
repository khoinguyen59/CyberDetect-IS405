"""
t-SNE Feature Embedding Visualization — Fig (line 1241-1244).

Paper reference:
  - Line 1241-1244: "projecting high-dimensional feature embeddings into a 2D space using t-SNE"
  - Shows how different attack classes cluster in learned representation space
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_tsne(model_path, test_csv, output_dir="results/tsne",
                  n_samples=3000, perplexity=30):
    """
    Generate t-SNE 2D projection of model's hidden layer activations.

    Paper line 1241-1244: projects high-dimensional feature embeddings 
    from the last hidden layer into 2D space using t-SNE.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and data
    model = load_model(model_path)
    df = pd.read_csv(test_csv)
    le = LabelEncoder()

    X = df.drop('label', axis=1).values
    y = le.fit_transform(df['label'].values)
    class_names = le.classes_

    # Sample for speed (t-SNE is O(n^2))
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Extract last hidden layer activations (128-dim, layer before output)
    # Paper: "feature embeddings" = penultimate layer output
    hidden_layer_model = Model(inputs=model.input,
                                outputs=model.layers[-2].output)
    embeddings = hidden_layer_model.predict(X)

    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Running t-SNE (n={len(X)}, perplexity={perplexity})...")

    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    X_2d = tsne.fit_transform(embeddings)

    # Also do raw feature t-SNE for comparison
    tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
    X_2d_raw = tsne_raw.fit_transform(X)

    # =====================================================
    # Plot: 2 panels — raw features vs learned embeddings
    # =====================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Raw feature t-SNE
    for c in range(len(class_names)):
        mask = y == c
        axes[0].scatter(X_2d_raw[mask, 0], X_2d_raw[mask, 1],
                        label=class_names[c], alpha=0.5, s=10)
    axes[0].set_title('(a) t-SNE of Raw Input Features', fontweight='bold')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    axes[0].legend(fontsize=7, loc='best')

    # Panel 2: Learned embedding t-SNE
    for c in range(len(class_names)):
        mask = y == c
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        label=class_names[c], alpha=0.5, s=10)
    axes[1].set_title('(b) t-SNE of Learned Embeddings (Layer 128)', fontweight='bold')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].legend(fontsize=7, loc='best')

    plt.suptitle('t-SNE Feature Embedding Visualization — CyberDetect-MLP',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_embeddings.png"), dpi=150)
    plt.close()
    print(f"[OK] Bieu do t-SNE da duoc luu tai {output_dir}/tsne_embeddings.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="t-SNE embedding visualization (Paper line 1241-1244)")
    parser.add_argument("model_path", help="Path to trained model (.h5)")
    parser.add_argument("test_csv", help="Test CSV with label column")
    parser.add_argument("--output_dir", default="results/tsne")
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()
    generate_tsne(args.model_path, args.test_csv, args.output_dir,
                  args.n_samples, args.perplexity)
