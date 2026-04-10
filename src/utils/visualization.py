"""
Visualization utilities for HQC-PINN results.

Generates publication-quality figures matching the paper's style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_training_curves(
    histories: dict,
    save_path: str = "results/figures/convergence.pdf",
    target_loss: float = 0.40,
):
    """Plot convergence comparison curves (Figure 2 in paper).

    Args:
        histories: Dict mapping model name -> training history dict.
        save_path: Output file path.
        target_loss: Dashed line for target convergence threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"HQC-PINN": "#1f77b4", "cPINN": "#ff7f0e", "VQC-only": "#2ca02c"}

    for name, history in histories.items():
        color = colors.get(name, None)
        epochs = range(1, len(history["val_loss"]) + 1)

        axes[0].plot(epochs, history["val_loss"], label=name, color=color, linewidth=2)
        axes[1].plot(epochs, history["val_accuracy"], label=name, color=color, linewidth=2)

    axes[0].axhline(y=target_loss, color="gray", linestyle="--", alpha=0.7, label=f"Target ({target_loss})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Convergence Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Accuracy Progression")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = "results/figures/confusion_matrix.pdf",
    title: str = "HQC-PINN Confusion Matrix",
):
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        save_path: Output file path.
        title: Plot title.
    """
    if class_names is None:
        class_names = ["No Flood", "Low", "Moderate", "Severe"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_uncertainty(
    predictions: dict,
    save_path: str = "results/figures/uncertainty.pdf",
):
    """Plot uncertainty quantification analysis (Figure 3 in paper).

    Args:
        predictions: Dict with 'entropy', 'aleatoric_uncertainty',
                     'probs_mean', 'logits_std'.
        save_path: Output file path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if "entropy" in predictions:
        entropy = predictions["entropy"]
        if hasattr(entropy, "numpy"):
            entropy = entropy.numpy()
        axes[0].hist(entropy, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Predictive Entropy")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Uncertainty Distribution")

    if "probs_mean" in predictions:
        probs = predictions["probs_mean"]
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        max_probs = np.max(probs, axis=1)
        axes[1].hist(max_probs, bins=50, color="#2ca02c", alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Maximum Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Confidence Distribution")

    if "aleatoric_uncertainty" in predictions:
        aleatoric = predictions["aleatoric_uncertainty"]
        if hasattr(aleatoric, "numpy"):
            aleatoric = aleatoric.numpy()
        axes[2].hist(aleatoric, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="black")
        axes[2].set_xlabel("Aleatoric Uncertainty")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Aleatoric Uncertainty Distribution")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
