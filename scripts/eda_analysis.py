"""
EDA Visualizations — Fig 4 (line 1030-1039).

Paper reference:
  - Line 1030-1039: "Figure 4 presents a comprehensive exploratory data analysis"
  - Fig 4a: Class distribution across attack types (bar chart)
  - Fig 4b: Top 10 features by MI scores (horizontal bar)
  - Fig 4c: PCA-based 2D projection (scatter plot)
  - Fig 4d: Distribution of top feature across classes (boxplot)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_eda(data_csv, mi_report_csv=None, output_dir="results/eda"):
    """
    Generate all 4 subfigures of Paper Fig 4.
    
    Args:
        data_csv: Preprocessed CSV with 'label' column
        mi_report_csv: MI scores report CSV (from feature_selector.py)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_csv)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    X = df.drop('label', axis=1)
    class_names = le.classes_

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # =====================================================
    # Fig 4a: Class distribution (line 1031-1032)
    # "class distribution across five attack types, 
    #  highlighting class imbalance"
    # =====================================================
    class_counts = pd.Series(y).map(lambda i: class_names[i]).value_counts()
    colors_a = sns.color_palette("viridis", len(class_counts))
    axes[0, 0].bar(range(len(class_counts)), class_counts.values, color=colors_a)
    axes[0, 0].set_xticks(range(len(class_counts)))
    axes[0, 0].set_xticklabels(class_counts.index, rotation=30, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('(a) Class Distribution', fontweight='bold')
    # Add count labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + v*0.01, str(v), ha='center', fontsize=8)

    # =====================================================
    # Fig 4b: Top 10 features by MI scores (line 1033-1034)
    # "ranks the top 10 features by Mutual Information scores"
    # =====================================================
    if mi_report_csv and os.path.exists(mi_report_csv):
        mi_df = pd.read_csv(mi_report_csv)
        top10 = mi_df.nlargest(10, 'MI_Score')
        axes[0, 1].barh(top10['Feature'].values[::-1], top10['MI_Score'].values[::-1],
                        color=sns.color_palette("magma", 10))
    else:
        # Compute MI on the spot
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X.values, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns).nlargest(10)
        axes[0, 1].barh(mi_series.index[::-1], mi_series.values[::-1],
                        color=sns.color_palette("magma", 10))
    axes[0, 1].set_xlabel('Mutual Information Score')
    axes[0, 1].set_title('(b) Top 10 Features by MI Score', fontweight='bold')

    # =====================================================
    # Fig 4c: PCA 2D projection (line 1034-1036)
    # "PCA-based 2D projection of the feature space"
    # =====================================================
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.values[:5000])  # Sample for speed
    y_sample = y[:5000]

    scatter = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_sample,
                                  cmap='Set1', alpha=0.5, s=10)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 0].set_title('(c) PCA 2D Projection', fontweight='bold')
    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plt.cm.Set1(i / max(len(class_names)-1, 1)),
               markersize=8, label=class_names[i])
               for i in range(min(len(class_names), 10))]
    axes[1, 0].legend(handles=handles, fontsize=7, loc='best')

    # =====================================================
    # Fig 4d: Top feature distribution by class (line 1036-1037)
    # "distribution of the top-ranked feature across attack classes"
    # =====================================================
    if mi_report_csv and os.path.exists(mi_report_csv):
        top_feature = pd.read_csv(mi_report_csv).nlargest(1, 'MI_Score')['Feature'].values[0]
    else:
        top_feature = X.columns[np.argmax(mi_scores)] if 'mi_scores' in dir() else X.columns[0]

    box_df = pd.DataFrame({
        'value': df[top_feature].values[:5000],
        'class': [class_names[i] for i in y[:5000]]
    })
    sns.boxplot(data=box_df, x='class', y='value', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_xlabel('Attack Class')
    axes[1, 1].set_ylabel(top_feature)
    axes[1, 1].set_title(f'(d) Distribution of "{top_feature}" by Class', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=30)

    plt.suptitle('Exploratory Data Analysis — TON_IoT Dataset (Fig 4)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_eda.png"), dpi=150)
    plt.close()
    print(f"[OK] Fig 4 (EDA) da duoc luu tai {output_dir}/fig4_eda.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EDA visualization (Paper Fig 4)")
    parser.add_argument("data_csv", help="Preprocessed CSV with label column")
    parser.add_argument("--mi_report", default=None, help="MI scores report CSV")
    parser.add_argument("--output_dir", default="results/eda")
    args = parser.parse_args()
    generate_eda(args.data_csv, args.mi_report, args.output_dir)
