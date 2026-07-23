"""Evaluation metrics (paper Sec V-C): Accuracy, Precision, Recall, Specificity,
F1-score, plus confusion matrix and mean +/- std aggregation over the 5 runs.

Positive class = 1 (AD / cognitively impaired), matching the paper's clinical
framing where Recall = sensitivity to impaired subjects.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "specificity"]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def confusion_counts(preds: Sequence[int], labels: Sequence[int]) -> Tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) for binary labels with positive class = 1."""
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return tp, tn, fp, fn


def compute_metrics(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    """Compute the five reported metrics (as fractions in [0, 1])."""
    tp, tn, fp, fn = confusion_counts(preds, labels)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # sensitivity
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def aggregate_runs(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-run metrics into {metric: {"mean":..., "std":...}}."""
    out: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out
