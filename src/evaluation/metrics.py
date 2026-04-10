"""
Evaluation metrics for HQC-PINN models.

Classification metrics, uncertainty calibration, and convergence analysis
as reported in Tables I-V of the paper.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> dict:
    """Compute classification metrics (Table II).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for AUC-ROC).

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    probs_mean: np.ndarray,
    entropy: np.ndarray,
    aleatoric: np.ndarray,
    confidence_level: float = 0.90,
) -> dict:
    """Compute uncertainty quantification metrics (Table IV).

    Args:
        y_true: Ground truth labels.
        probs_mean: Mean predicted probabilities from shot sampling.
        entropy: Predictive entropy for each sample.
        aleatoric: Aleatoric uncertainty for each sample.
        confidence_level: Nominal coverage level (default 90%).

    Returns:
        Dictionary of UQ metrics.
    """
    predicted_classes = np.argmax(probs_mean, axis=1)
    max_probs = np.max(probs_mean, axis=1)

    threshold = np.percentile(max_probs, (1 - confidence_level) * 100)
    confident_mask = max_probs >= threshold
    coverage = np.mean(
        (predicted_classes == y_true) | (~confident_mask)
    )

    return {
        "coverage": coverage,
        "mean_entropy": float(np.mean(entropy)),
        "mean_aleatoric": float(np.mean(aleatoric)),
        "mean_max_probability": float(np.mean(max_probs)),
        "confident_accuracy": float(
            np.mean(predicted_classes[confident_mask] == y_true[confident_mask])
        ) if confident_mask.any() else 0.0,
        "confidence_level": confidence_level,
    }


def compute_convergence_metrics(history: dict, target_loss: float = 0.40) -> dict:
    """Compute convergence analysis metrics (Table I).

    Args:
        history: Training history with 'val_loss' list.
        target_loss: Target validation loss threshold.

    Returns:
        Dictionary with convergence metrics.
    """
    val_losses = history["val_loss"]

    epochs_to_target = None
    for i, loss in enumerate(val_losses):
        if loss <= target_loss:
            epochs_to_target = i + 1
            break

    return {
        "epochs_to_target": epochs_to_target,
        "target_loss": target_loss,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "total_epochs": len(val_losses),
    }
