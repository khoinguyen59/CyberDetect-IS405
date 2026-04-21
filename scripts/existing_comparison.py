"""
Comparison with Existing Methods — Table 6 (line 1287-1295), Fig 12 (line 1299).

Paper reference:
  - Table 6 (line 1293): Performance comparison with 5 benchmark models
  - Fig 12 (line 1299): Accuracy comparison bar chart

Paper Table 6 data:
  | Model/Study                | Approach        | Dataset              | Accuracy |
  |----------------------------|-----------------|----------------------|----------|
  | Vinayakumar et al. (2019)  | CNN + RNN       | NSL-KDD, UNSW-NB15   | 97.01%   |
  | Alrashdi et al. (2019)     | ML (RF, SVM)    | TON_IoT (subset)     | 94.67%   |
  | Ferrag et al. (2020)       | DL Survey       | Various IDS          | 95.50%   |
  | Shone et al. (2018)        | Autoencoder+DNN | NSL-KDD              | 96.21%   |
  | Lopez-Martin et al. (2017) | CVAE (VAE)      | Custom IoT           | 95.03%   |
  | CyberDetect-MLP (proposed) | Optimized MLP   | TON_IoT (full)       | 98.87%   |
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Paper Table 6 — hardcoded from published results (line 1287-1293)
EXISTING_METHODS = [
    {
        "Model": "Vinayakumar et al. (2019)",
        "Approach": "CNN + RNN",
        "Dataset": "NSL-KDD, UNSW-NB15",
        "Accuracy": 97.01,
        "Scalability": "Moderate",
        "Explainability": "Low",
    },
    {
        "Model": "Alrashdi et al. (2019)",
        "Approach": "ML (RF, SVM)",
        "Dataset": "TON_IoT (subset)",
        "Accuracy": 94.67,
        "Scalability": "Limited",
        "Explainability": "Medium",
    },
    {
        "Model": "Ferrag et al. (2020)",
        "Approach": "Deep Learning Survey",
        "Dataset": "Various IDS Datasets",
        "Accuracy": 95.50,
        "Scalability": "Varies",
        "Explainability": "N/A",
    },
    {
        "Model": "Shone et al. (2018)",
        "Approach": "Autoencoder + DNN",
        "Dataset": "NSL-KDD",
        "Accuracy": 96.21,
        "Scalability": "Low",
        "Explainability": "Low",
    },
    {
        "Model": "Lopez-Martin et al. (2017)",
        "Approach": "CVAE (VAE)",
        "Dataset": "Custom IoT dataset",
        "Accuracy": 95.03,
        "Scalability": "Low",
        "Explainability": "Medium",
    },
]


def generate_table6_and_fig12(cyberdetect_accuracy=98.87, output_dir="results/comparison"):
    """
    Generate Table 6 and Fig 12 from paper.

    Args:
        cyberdetect_accuracy: Accuracy achieved by our model (default from paper: 98.87%)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add our model to the comparison
    all_methods = EXISTING_METHODS + [{
        "Model": "CyberDetect-MLP (proposed)",
        "Approach": "Optimized MLP",
        "Dataset": "TON_IoT (full)",
        "Accuracy": cyberdetect_accuracy,
        "Scalability": "High",
        "Explainability": "High (Grad-CAM)",
    }]

    # Save Table 6
    df = pd.DataFrame(all_methods)
    df.to_csv(os.path.join(output_dir, "table6_existing_comparison.csv"), index=False)
    print(f"[OK] Table 6 da duoc luu tai {output_dir}/table6_existing_comparison.csv")
    print(df.to_string(index=False))

    # Generate Fig 12 — Accuracy comparison bar chart
    generate_fig12(all_methods, output_dir)

    return df


def generate_fig12(methods, output_dir):
    """
    Paper Fig 12 (line 1269-1275): Accuracy comparison bar chart.

    "CyberDetect-MLP achieves the highest recorded accuracy of 98.87%.
    This superior result is attributed to the model's adaptive learning dynamics."
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    names = [m['Model'] for m in methods]
    accuracies = [m['Accuracy'] for m in methods]

    # Color: proposed model in distinct color
    colors = ['#3498db'] * (len(methods) - 1) + ['#e74c3c']

    bars = ax.bar(range(len(names)), accuracies, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' et al.', '\net al.') for n in names],
                        rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison of CyberDetect-MLP with Existing Models (Fig 12)',
                 fontweight='bold', fontsize=13)

    # Set y-axis to show differences clearly
    ax.set_ylim(93, 100)
    ax.axhline(y=98.87, color='red', linestyle='--', alpha=0.3, label='CyberDetect-MLP')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig12_existing_comparison.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 12 da duoc luu tai {output_dir}/fig12_existing_comparison.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Comparison with existing methods (Paper Table 6, Fig 12)")
    parser.add_argument("--accuracy", type=float, default=98.87,
                        help="CyberDetect-MLP accuracy to compare")
    parser.add_argument("--output_dir", default="results/comparison")
    args = parser.parse_args()
    generate_table6_and_fig12(args.accuracy, args.output_dir)
