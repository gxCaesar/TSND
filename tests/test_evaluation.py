"""Tests for production TSND evaluation metrics and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsnd.evaluation.metrics import (
    bootstrap_ci,
    brier_score,
    ece,
    paired_bootstrap_delta_ci,
    pr_auc,
)

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
N_CLASSES = 2
PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0
PR_AUC_EXPECTED_PERFECT = 1.0
BRIER_EXPECTED_PERFECT = 0.0
HIGH_SCORE = 0.9
SECOND_HIGH_SCORE = 0.8
LOW_SCORE = 0.2
LOWER_SCORE = 0.1
PAIRED_BASE_SCORE = 0.55
PAIRED_AUG_HIGH = 0.95
PAIRED_AUG_LOW = 0.05
ECE_N_SAMPLES = 100
ECE_N_BINS = 10
ECE_HALF_SAMPLES = ECE_N_SAMPLES // N_CLASSES
ECE_CONFIDENT_POSITIVE = 0.99
ECE_CONFIDENT_NEGATIVE = 0.01
ECE_THRESHOLD = 0.05
BOOTSTRAP_N_SAMPLES = 50
BOOTSTRAP_N_RESAMPLES = 300
BOOTSTRAP_SEED = 314
BOOTSTRAP_SCORE_SHIFT = 0.3
BOOTSTRAP_NOISE_SCALE = 0.2

PERFECT_RANKER_TRUE = np.array(
    [POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL],
    dtype=np.int_,
)
PERFECT_RANKER_SCORE = np.array(
    [HIGH_SCORE, SECOND_HIGH_SCORE, LOW_SCORE, LOWER_SCORE],
    dtype=np.float64,
)
PERFECT_BRIER_TRUE = np.array([POSITIVE_LABEL, NEGATIVE_LABEL], dtype=np.int_)
PERFECT_BRIER_PROB = np.array([PROBABILITY_MAX, PROBABILITY_MIN], dtype=np.float64)


def _bootstrap_pr_auc_interval(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return the production PR-AUC lower bound, point estimate, and upper bound."""
    lower, upper = bootstrap_ci(
        y_true,
        y_score,
        pr_auc,
        n_bootstrap=n_resamples,
        seed=seed,
    )
    return lower, pr_auc(y_true, y_score), upper


def _make_bootstrap_toy() -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Create a non-perfect binary toy ranking problem."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_true = np.array(
        [NEGATIVE_LABEL, POSITIVE_LABEL] * (BOOTSTRAP_N_SAMPLES // N_CLASSES),
        dtype=np.int_,
    )
    noise = rng.normal(scale=BOOTSTRAP_NOISE_SCALE, size=BOOTSTRAP_N_SAMPLES)
    y_score = np.clip(
        y_true.astype(np.float64) * BOOTSTRAP_SCORE_SHIFT + noise,
        PROBABILITY_MIN,
        PROBABILITY_MAX,
    )
    return y_true, y_score


def test_pr_auc_known_answer_does_equal_one_for_perfect_ranker() -> None:
    """Assert the production PR-AUC is one for a perfect ranker."""
    assert pr_auc(PERFECT_RANKER_TRUE, PERFECT_RANKER_SCORE) == PR_AUC_EXPECTED_PERFECT


def test_brier_score_known_answer_does_equal_zero_for_perfect_predictor() -> None:
    """Assert the production Brier score is zero for perfect probabilities."""
    assert brier_score(PERFECT_BRIER_TRUE, PERFECT_BRIER_PROB) == BRIER_EXPECTED_PERFECT


def test_netcal_ece_does_stay_small_for_confident_correct_predictions() -> None:
    """Assert the production ECE is small for confident correct predictions."""
    y_true = np.array(
        [POSITIVE_LABEL] * ECE_HALF_SAMPLES + [NEGATIVE_LABEL] * ECE_HALF_SAMPLES,
        dtype=np.int_,
    )
    y_prob = np.array(
        [ECE_CONFIDENT_POSITIVE] * ECE_HALF_SAMPLES
        + [ECE_CONFIDENT_NEGATIVE] * ECE_HALF_SAMPLES,
        dtype=np.float64,
    )

    assert ece(y_true, y_prob, ECE_N_BINS) < ECE_THRESHOLD


def test_bootstrap_does_return_finite_ordered_interval() -> None:
    """Assert the production bootstrap returns an ordered PR-AUC interval."""
    y_true, y_score = _make_bootstrap_toy()

    lower, mean, upper = _bootstrap_pr_auc_interval(
        y_true,
        y_score,
        BOOTSTRAP_N_RESAMPLES,
        BOOTSTRAP_SEED,
    )

    assert np.all(np.isfinite(np.array([lower, mean, upper], dtype=np.float64)))
    assert lower < mean < upper


def test_bootstrap_single_class_does_return_nan_interval() -> None:
    """Assert production bootstrap returns NaNs for single-class labels."""
    y_true = np.zeros(BOOTSTRAP_N_SAMPLES, dtype=np.int_)
    y_score = np.zeros(BOOTSTRAP_N_SAMPLES, dtype=np.float64)

    interval = _bootstrap_pr_auc_interval(
        y_true,
        y_score,
        BOOTSTRAP_N_RESAMPLES,
        BOOTSTRAP_SEED,
    )

    assert np.all(np.isnan(np.array(interval, dtype=np.float64)))


def test_paired_bootstrap_delta_ci_detects_positive_increment() -> None:
    """Assert production paired bootstrap detects a positive PR-AUC increment."""
    labels = np.array(
        [POSITIVE_LABEL, POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL, NEGATIVE_LABEL],
        dtype=np.int_,
    )
    baseline = np.full(labels.shape, PAIRED_BASE_SCORE, dtype=np.float64)
    augmented = np.array(
        [PAIRED_AUG_HIGH, 0.9, 0.85, 0.2, 0.1, PAIRED_AUG_LOW],
        dtype=np.float64,
    )

    result = paired_bootstrap_delta_ci(
        labels,
        baseline,
        augmented,
        pr_auc,
        n_bootstrap=BOOTSTRAP_N_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    assert result["delta"] > 0.0
    assert result["ci_low"] > 0.0
    assert result["n_valid"] > 0.0
