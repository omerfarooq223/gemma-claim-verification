"""Evaluation metrics, per-class breakdown, and confusion matrix computation."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from .constants import ALLOWED_LABELS, LabelEnum


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute overall accuracy, macro-F1, per-class metrics, and confusion matrix."""
    if labels is None:
        labels = ALLOWED_LABELS

    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatch in length: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    n_samples = len(y_true)
    if n_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}

    # Confusion matrix index mapping
    label_to_idx = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)
    cm = [[0] * num_classes for _ in range(num_classes)]

    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
        t_idx = label_to_idx.get(yt)
        p_idx = label_to_idx.get(yp)
        if t_idx is not None and p_idx is not None:
            cm[t_idx][p_idx] += 1

    accuracy = correct / n_samples

    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        support = sum(cm[i][c] for c in range(num_classes))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": n_samples,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def format_evaluation_summary(metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics as a human-readable table."""
    lines = [
        f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}% ({metrics['correct']}/{metrics['total']})",
        f"Macro-F1 Score:   {metrics['macro_f1'] * 100:.2f}%",
        "",
        "Per-Class Metrics:",
        f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
        "-" * 68,
    ]

    for label, stats in metrics["per_class"].items():
        lines.append(
            f"{label:<20} | {stats['precision'] * 100:>9.2f}% | {stats['recall'] * 100:>9.2f}% | {stats['f1'] * 100:>9.2f}% | {stats['support']:>8}"
        )

    lines.extend([
        "",
        "Confusion Matrix (Rows = True, Columns = Predicted):",
        f"Order: {metrics['labels']}",
    ])
    for row in metrics["confusion_matrix"]:
        lines.append(f"  {row}")

    return "\n".join(lines)
