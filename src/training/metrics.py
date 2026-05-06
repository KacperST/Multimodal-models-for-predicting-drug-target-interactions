from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: list | np.ndarray,
    probs: list | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        labels: Ground-truth binary labels.
        probs: Predicted probabilities (after sigmoid).
        threshold: Decision threshold for converting probs to predictions.

    Returns:
        Dictionary with keys: auc, auprc, f1, precision, recall.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs > threshold).astype(int)

    return {
        "auc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def compute_confusion_matrix(
    labels: list | np.ndarray,
    probs: list | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Compute confusion matrix values.

    Returns:
        Dictionary with keys: tp, fp, fn, tn.
    """
    labels = np.asarray(labels)
    preds = (np.asarray(probs) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
