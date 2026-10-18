"""Tests for lightweight production TSND forecasting steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tsnd.training.forecaster import ForecasterSettings, _fit_eval_lightgbm

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
N_SAMPLES = 20
N_FEATURES = 4
N_CLASSES = 2
N_PER_CLASS = N_SAMPLES // N_CLASSES
TOY_FEATURE_SEED = 17
NEGATIVE_FEATURE_CENTER = -1.5
POSITIVE_FEATURE_CENTER = 1.5
FEATURE_NOISE_SCALE = 0.1
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
CLASSIFICATION_THRESHOLD = 0.5
EXPECTED_PREDICTED_CLASSES = 2


def _forecaster_config(columns: tuple[str, ...], seed: int) -> dict[str, object]:
    """Return the smallest production forecaster configuration."""
    return {
        "forecaster": {
            "feature_groups": {"toy": list(columns)},
            "seeds": [seed],
            "label_column": "label",
            "calibration_fraction": 0.2,
            "min_calibration_samples": 100,
            "min_cohort_samples": 2,
            "bootstrap_samples": 30,
            "ci_alpha": 0.05,
            "ece_bins": 5,
            "precision_k": 5,
            "lightgbm_params": {
                "n_estimators": 12,
                "learning_rate": 0.1,
                "num_leaves": 4,
                "max_depth": 2,
                "min_child_samples": 1,
                "n_jobs": 1,
                "verbosity": -1,
            },
            "use_survival_model": False,
        },
        "compute": {"n_process": 1},
    }


def _production_fit(
    features: NDArray[np.float64],
    labels: NDArray[np.int_],
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    """Run the submitted LightGBM fit and return scores plus PR-AUC."""
    columns = tuple(f"x{index}" for index in range(features.shape[1]))
    frame = pd.DataFrame(features, columns=columns)
    frame["label"] = labels
    settings = ForecasterSettings.from_config(_forecaster_config(columns, seed))
    result = _fit_eval_lightgbm(frame, frame.copy(), columns, seed, settings)
    return result.y_score, float(result.metrics["pr_auc"])


def _make_toy_feature_table() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create a separable binary toy feature table."""
    rng = np.random.default_rng(TOY_FEATURE_SEED)
    negative_features = rng.normal(
        loc=NEGATIVE_FEATURE_CENTER,
        scale=FEATURE_NOISE_SCALE,
        size=(N_PER_CLASS, N_FEATURES),
    )
    positive_features = rng.normal(
        loc=POSITIVE_FEATURE_CENTER,
        scale=FEATURE_NOISE_SCALE,
        size=(N_PER_CLASS, N_FEATURES),
    )
    features = np.vstack((negative_features, positive_features)).astype(np.float64)
    labels = np.array(
        [NEGATIVE_LABEL] * N_PER_CLASS + [POSITIVE_LABEL] * N_PER_CLASS,
        dtype=np.int_,
    )
    return features, labels


def test_forecaster_training_step_does_fit_and_predict_both_classes() -> None:
    """Assert production probabilities and predictions are valid."""
    features, labels = _make_toy_feature_table()

    probabilities, _ = _production_fit(features, labels, TOY_FEATURE_SEED)
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(np.int_)

    assert np.all(probabilities >= PROBABILITY_MIN)
    assert np.all(probabilities <= PROBABILITY_MAX)
    assert len(np.unique(predictions)) == EXPECTED_PREDICTED_CLASSES


def test_single_class_cohort_does_return_nan_score() -> None:
    """Assert all-zero labels produce NaN rather than a misleading score."""
    features, _ = _make_toy_feature_table()
    labels = np.zeros(N_SAMPLES, dtype=np.int_)

    _, score = _production_fit(features, labels, TOY_FEATURE_SEED)

    assert np.isnan(score)
