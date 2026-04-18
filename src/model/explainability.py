"""
CyberDetect-MLP Explainability Module.

Paper references:
  - Line 1251: "we mainly use Integrated Gradients and SHAP"
  - Line 1219: "Grad-CAM adapted for multi-layer perceptrons using feature-space gradients"
  - Line 1221: "visualize which input features most significantly influenced the model's decisions"
  - Line 955: "Using SHAP values, the system can highlight the most influential features"

Implements three XAI methods:
  1. SHAP (SHapley Additive exPlanations) — primary method
  2. Integrated Gradients — gradient-based attribution
  3. 1D Grad-CAM — adapted for dense layers per paper line 1219
"""

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def explain_shap(model, sample_csv, output_dir="results/xai"):
    """
    SHAP explainability — Paper line 955, 1251.

    Generates:
      - SHAP summary plot (beeswarm)
      - SHAP bar plot (mean absolute SHAP values)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(sample_csv)
    X = df.drop('label', axis=1)

    # Use 200 background samples for KernelExplainer efficiency
    background = X.sample(min(200, len(X)), random_state=42)
    test_samples = X.sample(min(200, len(X)), random_state=123)

    # KernelExplainer handles multiclass softmax output reliably
    # Paper line 955: "Using SHAP values, the system can highlight
    # the most influential features contributing to a specific attack prediction"
    explainer = shap.KernelExplainer(model.predict, background.values)
    shap_values = explainer.shap_values(test_samples.values)

    # For multiclass, shap_values is a list of arrays (one per class)
    # Use mean absolute values across all classes for summary
    if isinstance(shap_values, list):
        shap_mean = np.mean(np.abs(np.array(shap_values)), axis=0)
    else:
        shap_mean = shap_values

    # Summary plot (beeswarm)
    shap.summary_plot(shap_mean, test_samples, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # Bar plot (mean |SHAP|)
    shap.summary_plot(shap_mean, test_samples, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), bbox_inches='tight', dpi=150)
    plt.close()

    print(f"[OK] Cac bieu do SHAP da duoc luu tai {output_dir}/")
    return shap_values


def integrated_gradients(model, input_sample, baseline=None, steps=50):
    """
    Integrated Gradients — Paper line 1251.

    Computes attribution scores by integrating gradients along the path
    from a baseline (zero vector) to the actual input.

    Paper ref: "we mainly use Integrated Gradients (Sundararajan et al., 2017)"

    Formula: IG_i(x) = (x_i - x'_i) * ∫₀¹ (∂F(x' + α(x - x')) / ∂x_i) dα

    Args:
        model: Trained Keras model
        input_sample: Single input sample (1D numpy array)
        baseline: Baseline input (default: zero vector)
        steps: Number of interpolation steps (higher = more precise)

    Returns:
        Attribution scores for each feature (same shape as input)
    """
    if baseline is None:
        baseline = np.zeros_like(input_sample)

    input_tensor = tf.cast(input_sample.reshape(1, -1), tf.float32)
    baseline_tensor = tf.cast(baseline.reshape(1, -1), tf.float32)

    # Generate interpolated inputs along the path
    alphas = tf.cast(np.linspace(0, 1, steps + 1).reshape(-1, 1), tf.float32)
    interpolated = baseline_tensor + alphas * (input_tensor - baseline_tensor)  # (steps+1, n_features)

    # Compute gradients at each interpolated point
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated)
        target_class = tf.argmax(predictions[-1])  # Class of actual input
        target_scores = predictions[:, target_class]

    gradients = tape.gradient(target_scores, interpolated)  # (steps+1, n_features)

    # Approximate integral using trapezoidal rule
    avg_gradients = tf.reduce_mean(gradients, axis=0)  # (n_features,)

    # Scale by (input - baseline)
    ig_attributions = (input_tensor - baseline_tensor) * avg_gradients  # (1, n_features)

    return ig_attributions.numpy().flatten()


def explain_integrated_gradients(model, sample_csv, output_dir="results/xai", n_samples=100):
    """
    Batch Integrated Gradients explanation.

    Computes IG attributions for multiple samples and generates
    a feature importance bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(sample_csv)
    X = df.drop('label', axis=1)
    feature_names = X.columns.tolist()

    samples = X.sample(min(n_samples, len(X)), random_state=42).values

    all_attributions = []
    for i, sample in enumerate(samples):
        attr = integrated_gradients(model, sample)
        all_attributions.append(np.abs(attr))

    # Average absolute attributions across samples
    mean_attr = np.mean(all_attributions, axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_attr)[::-1][:20]  # Top 20
    top_features = [feature_names[i] for i in sorted_idx]
    top_scores = mean_attr[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_scores[::-1], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Mean |Integrated Gradients Attribution|')
    ax.set_title('Integrated Gradients — Top 20 Feature Attributions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "integrated_gradients.png"), dpi=150)
    plt.close()

    print(f"[OK] Bieu do Integrated Gradients da duoc luu tai {output_dir}/integrated_gradients.png")
    return mean_attr


def mlp_grad_cam_1d(model, input_sample, target_class=None, layer_name=None):
    """
    1D Grad-CAM adapted for MLP — Paper line 1219.

    Paper ref: "Grad-CAM adapted for multi-layer perceptrons using feature-space gradients"

    For MLP (fully connected), we compute:
      1. Forward pass → get activation of the last hidden Dense layer (Dense(128))
      2. Compute gradient of target class score w.r.t. that activation
      3. Global average pooling of gradients → importance weights
      4. Weighted sum of activations → 1D heatmap

    This is the dense-layer equivalent of Grad-CAM:
      L^c = ReLU(Σ_k α^c_k · A_k)
      where α^c_k = (1/Z) Σ_i ∂y^c/∂A^k_i

    Args:
        model: Trained Keras model
        input_sample: Single input (1D numpy array)
        target_class: Class index to explain (default: predicted class)
        layer_name: Name of target hidden layer (default: last Dense before output)

    Returns:
        1D heatmap array of feature-space importance
    """
    input_tensor = tf.cast(input_sample.reshape(1, -1), tf.float32)

    # Find the target layer (last Dense layer before output)
    if layer_name is None:
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        if len(dense_layers) < 2:
            raise ValueError("Model needs at least 2 Dense layers for Grad-CAM")
        target_layer = dense_layers[-2]  # Second-to-last Dense (Dense(128))
    else:
        target_layer = model.get_layer(layer_name)

    # Build sub-model that outputs both the target layer activation and final prediction
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        layer_output, predictions = grad_model(input_tensor)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_score = predictions[:, target_class]

    # Gradient of class score w.r.t. target layer output
    grads = tape.gradient(class_score, layer_output)  # (1, 128)

    # Global Average Pooling of gradients → importance weights (α_k)
    # For 1D: simply take mean across batch dimension
    weights = tf.reduce_mean(grads, axis=0)  # (128,)

    # Weighted combination of activations
    cam = tf.reduce_sum(weights * layer_output[0], axis=0)  # scalar per neuron
    # For dense layer, cam is a scalar — we need the per-neuron weighted activation
    cam_1d = tf.nn.relu(weights * layer_output[0]).numpy()  # (128,)

    return cam_1d


def explain_grad_cam(model, sample_csv, output_dir="results/xai", n_samples=50):
    """
    Batch 1D Grad-CAM explanation — Paper line 1219, 1231.

    Generates heatmap visualizations for correctly and incorrectly classified samples,
    matching Fig 10 of the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(sample_csv)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    samples = X[:min(n_samples, len(X))]
    labels = y[:min(n_samples, len(y))]

    correct_cams, incorrect_cams = [], []
    preds = model.predict(samples)
    pred_classes = preds.argmax(axis=1)

    for i in range(len(samples)):
        cam = mlp_grad_cam_1d(model, samples[i])
        if pred_classes[i] == labels[i]:
            correct_cams.append(cam)
        else:
            incorrect_cams.append(cam)

    # Plot average Grad-CAM for correct vs incorrect predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if correct_cams:
        avg_correct = np.mean(correct_cams, axis=0)
        axes[0].bar(range(len(avg_correct)), avg_correct, color='green', alpha=0.7)
        axes[0].set_title(f'Grad-CAM — Correct Predictions (n={len(correct_cams)})')
        axes[0].set_xlabel('Neuron Index (Dense-128 layer)')
        axes[0].set_ylabel('Activation × Gradient')

    if incorrect_cams:
        avg_incorrect = np.mean(incorrect_cams, axis=0)
        axes[1].bar(range(len(avg_incorrect)), avg_incorrect, color='red', alpha=0.7)
        axes[1].set_title(f'Grad-CAM — Incorrect Predictions (n={len(incorrect_cams)})')
        axes[1].set_xlabel('Neuron Index (Dense-128 layer)')
        axes[1].set_ylabel('Activation × Gradient')
    else:
        axes[1].text(0.5, 0.5, 'No misclassifications\nin sample',
                     ha='center', va='center', fontsize=12)
        axes[1].set_title('Grad-CAM — Incorrect Predictions (n=0)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grad_cam_1d.png"), dpi=150)
    plt.close()

    print(f"[OK] Cac bieu do 1D Grad-CAM da duoc luu tai {output_dir}/grad_cam_1d.png")


def explain(model, sample_csv, output_dir="results/xai"):
    """
    Full explainability pipeline — runs all three XAI methods.

    Paper line 870: "real-time inference with XAI-based interpretation (Grad-CAM and SHAP)"
    Paper line 1251: "we mainly use Integrated Gradients and SHAP"
    """
    print("\n" + "-"*50)
    print("[CHECK] Dang chay Phan tich Giai thich (3 phuong phap)...")
    print("-" * 50)

    # Method 1: SHAP (Paper line 955, 1251)
    print("\n--- 1/3: SHAP ---")
    explain_shap(model, sample_csv, output_dir)

    # Method 2: Integrated Gradients (Paper line 1251)
    print("\n--- 2/3: Integrated Gradients ---")
    explain_integrated_gradients(model, sample_csv, output_dir)

    # Method 3: 1D Grad-CAM (Paper line 1219)
    print("\n--- 3/3: 1D Grad-CAM ---")
    explain_grad_cam(model, sample_csv, output_dir)

    print(f"\n[OK] Tat ca 3 phuong phap XAI da hoan tat. Ket qua tai {output_dir}/")
