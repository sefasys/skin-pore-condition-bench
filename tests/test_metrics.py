import numpy as np

from pore_assessment.metrics import classification_metrics


def test_perfect_predictions_have_perfect_ordinal_metrics() -> None:
    targets = np.asarray([0, 1, 2, 3, 4])
    probabilities = np.eye(5)
    metrics = classification_metrics(targets, targets, probabilities)

    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mean_absolute_class_error"] == 0.0
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["calibration"]["expected_calibration_error_10_bins"] == 0.0
