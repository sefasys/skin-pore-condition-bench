"""Metrics for nominal and ordinal pore-condition evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = ["very_good", "good", "normal", "poor", "very_poor"]


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        accuracy = correct[mask].mean()
        mean_confidence = confidences[mask].mean()
        error += mask.mean() * abs(float(accuracy - mean_confidence))
    return float(error)


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None = None,
) -> dict:
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_array,
        prediction_array,
        labels=labels,
        zero_division=0,
    )
    result = {
        "exact_accuracy": float(accuracy_score(target_array, prediction_array)),
        "within_one_class_accuracy": float(
            (np.abs(target_array - prediction_array) <= 1).mean()
        ),
        "two_or_more_class_error_rate": float(
            (np.abs(target_array - prediction_array) >= 2).mean()
        ),
        "macro_f1": float(
            f1_score(target_array, prediction_array, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                target_array,
                prediction_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_array, prediction_array)
        ),
        "mean_absolute_class_error": float(
            np.abs(target_array - prediction_array).mean()
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                target_array,
                prediction_array,
                weights="quadratic",
                labels=labels,
            )
        ),
        "per_class": {
            name: {
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for (
                name,
                class_precision,
                class_recall,
                class_f1,
                class_support,
            ) in zip(CLASS_NAMES, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": confusion_matrix(
            target_array,
            prediction_array,
            labels=labels,
        ).tolist(),
    }

    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        expected_shape = (len(target_array), len(CLASS_NAMES))
        if probability_array.shape != expected_shape:
            raise ValueError(
                f"Expected probability shape {expected_shape}, "
                f"received {probability_array.shape}"
            )
        probability_array = probability_array / probability_array.sum(
            axis=1,
            keepdims=True,
        )
        one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[target_array]
        result["calibration"] = {
            "expected_calibration_error_10_bins": expected_calibration_error(
                target_array,
                probability_array,
                bins=10,
            ),
            "multiclass_brier_score": float(
                np.square(probability_array - one_hot).sum(axis=1).mean()
            ),
            "negative_log_likelihood": float(
                log_loss(target_array, probability_array, labels=labels)
            ),
        }
        if len(np.unique(target_array)) == len(CLASS_NAMES):
            result["discrimination"] = {
                "macro_ovr_roc_auc": float(
                    roc_auc_score(
                        target_array,
                        probability_array,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "macro_average_precision": float(
                    average_precision_score(one_hot, probability_array, average="macro")
                ),
            }
    return result
