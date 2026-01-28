"""Metrics for binary sarcasm classification.

Headline metrics:

- **macro-F1** — average of per-class F1, treats both classes equally
  (the right call here, since both sarcastic and non-sarcastic matter).
- **per-class precision / recall / F1** — for digging into where the
  model misses.
- **accuracy** — included for completeness, but it's a weak metric on
  near-balanced data + we don't want it driving decisions.

All public functions return plain Python floats / dicts so the values
are JSON-serialisable straight into the API and Streamlit responses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    """Precision/recall/F1/support for a single class."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Aggregated metrics for the whole binary task."""

    accuracy: float
    macro_f1: float
    not_sarcastic: PerClassMetrics
    sarcastic: PerClassMetrics
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-averaged F1 score across both classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


def per_class_metrics(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[PerClassMetrics, PerClassMetrics]:
    """Return (metrics_for_class_0, metrics_for_class_1)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0.0
    )
    return (
        PerClassMetrics(
            precision=float(precision[0]),
            recall=float(recall[0]),
            f1=float(f1[0]),
            support=int(support[0]),
        ),
        PerClassMetrics(
            precision=float(precision[1]),
            recall=float(recall[1]),
            f1=float(f1[1]),
            support=int(support[1]),
        ),
    )


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> ClassificationReport:
    """One-shot evaluation. Returns the full classification report."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    not_sarcastic, sarcastic = per_class_metrics(y_true_arr, y_pred_arr)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        macro_f1=macro_f1(y_true_arr, y_pred_arr),
        not_sarcastic=not_sarcastic,
        sarcastic=sarcastic,
        confusion=cm.tolist(),
    )
