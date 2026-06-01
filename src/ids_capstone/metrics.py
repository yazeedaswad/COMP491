from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    report: dict
    confusion_matrix: list[list[int]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        report=classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )
    return asdict(metrics)

