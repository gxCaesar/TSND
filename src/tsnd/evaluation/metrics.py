"""Evaluation metrics for TSND front forecasting."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger(__name__)

DEFAULT_ECE_BINS = 15
DEFAULT_BOOTSTRAP_RESAMPLES = 300
DEFAULT_CI_ALPHA = 0.05

MetricFn = Callable[[NDArray[np.int_], NDArray[np.float64]], float]
CellScores = Mapping[Any, Any]


def _validate_binary_inputs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    allow_empty: bool = True,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Validate binary labels and probability scores.

    Args:
        y_true: Binary labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).
        allow_empty: Whether zero-length arrays should be accepted.

    Returns:
        A tuple ``(labels, scores)`` with shapes (n_fronts,) and dtypes int and
        float64.

    Raises:
        ValueError: If shapes, labels, or score ranges are invalid.
    """
    y_raw = np.asarray(y_true)
    score_raw = np.asarray(y_score)
    if y_raw.ndim != 1:
        raise ValueError("y_true must be one-dimensional with shape (n_fronts,).")
    if score_raw.ndim != 1:
        raise ValueError("y_score must be one-dimensional with shape (n_fronts,).")
    if y_raw.shape[0] != score_raw.shape[0]:
        raise ValueError(
            "y_true and y_score must have the same length; "
            f"got {y_raw.shape[0]} and {score_raw.shape[0]}."
        )
    if y_raw.shape[0] == 0:
        if allow_empty:
            return y_raw.astype(int), score_raw.astype(np.float64)
        raise ValueError("Inputs must contain at least one front.")

    scores = score_raw.astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain only finite values.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("y_score must contain probabilities in [0, 1].")

    labels_float = y_raw.astype(np.float64)
    if not np.all(np.isfinite(labels_float)):
        raise ValueError("y_true must contain only finite values.")
    if not np.all(np.isin(labels_float, [0.0, 1.0])):
        raise ValueError("y_true must contain binary labels encoded as 0/1.")

    return labels_float.astype(int), scores


def _has_two_classes(y_true: NDArray[np.int_]) -> bool:
    """Return whether labels contain both classes.

    Args:
        y_true: Binary labels with shape (n_fronts,).

    Returns:
        ``True`` when both 0 and 1 are present.
    """
    return np.unique(y_true).size == 2


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute PR-AUC, the TSND primary metric.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).

    Returns:
        Average precision, or ``nan`` for empty or single-class cohorts.
    """
    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0 or not _has_two_classes(labels):
        LOGGER.warning("PR-AUC is undefined for empty or single-class labels.")
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute ROC-AUC.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).

    Returns:
        ROC-AUC, or ``nan`` for empty or single-class cohorts.
    """
    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0 or not _has_two_classes(labels):
        LOGGER.warning("ROC-AUC is undefined for empty or single-class labels.")
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def precision_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Compute precision among the top-k scored fronts.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).
        k: Number of highest-scored fronts to evaluate.

    Returns:
        Fraction of positives in the top-k slice, or ``nan`` when no fronts are
        available.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}.")
    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0:
        LOGGER.warning("precision@k is undefined for empty labels.")
        return float("nan")
    top_k = min(k, labels.size)
    order = np.argsort(-scores, kind="mergesort")
    return float(np.mean(labels[order[:top_k]]))


def brier_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Compute the Brier score for binary probabilities.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).

    Returns:
        Brier score loss, or ``nan`` for empty cohorts.
    """
    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0:
        LOGGER.warning("Brier score is undefined for empty labels.")
        return float("nan")
    from sklearn.metrics import brier_score_loss

    return float(brier_score_loss(labels, scores))


def ece(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = DEFAULT_ECE_BINS) -> float:
    """Compute expected calibration error with netcal.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).
        n_bins: Number of calibration bins.

    Returns:
        Expected calibration error, or ``nan`` for empty cohorts.

    Raises:
        ValueError: If ``n_bins`` is not positive.
        ImportError: If ``netcal`` is unavailable at runtime.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive; got {n_bins}.")
    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0:
        LOGGER.warning("ECE is undefined for empty labels.")
        return float("nan")

    from netcal.metrics import ECE

    probabilities = np.column_stack([1.0 - scores, scores])
    return float(ECE(n_bins).measure(probabilities, labels))


def _bootstrap_metric_value(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    metric_fn: MetricFn,
    child_seed: int,
) -> float | None:
    """Compute one bootstrap metric value.

    Args:
        labels: Binary labels with shape (n_fronts,).
        scores: Positive-class probabilities with shape (n_fronts,).
        metric_fn: Metric function accepting ``(y_true, y_score)`` arrays.
        child_seed: RNG seed for this resample.

    Returns:
        Metric value, or ``None`` when the resample is invalid.
    """
    rng = np.random.default_rng(child_seed)
    sample_idx = rng.integers(0, labels.size, labels.size)
    sample_y = labels[sample_idx]
    if not _has_two_classes(sample_y):
        return None
    try:
        value = float(metric_fn(sample_y, scores[sample_idx]))
    except (FloatingPointError, ValueError) as exc:
        LOGGER.debug("Skipping invalid bootstrap sample: %s", exc)
        return None
    if np.isfinite(value):
        return value
    return None


def bootstrap_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metric_fn: MetricFn,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_CI_ALPHA,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap confidence interval.

    Args:
        y_true: Binary uptake labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).
        metric_fn: Metric function accepting ``(y_true, y_score)`` arrays.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Two-sided interval tail mass.
        seed: Optional RNG seed. Pass the run seed from config for reproducible CIs.
        n_jobs: Optional joblib worker count. ``None`` keeps serial execution.

    Returns:
        ``(lower, upper)`` confidence interval. Returns ``(nan, nan)`` when the
        full cohort is empty or single-class, or when every resample is
        single-class.

    Raises:
        ValueError: If bootstrap parameters are invalid.
    """
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive; got {n_bootstrap}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be None or a non-zero integer.")

    labels, scores = _validate_binary_inputs(y_true, y_score)
    if labels.size == 0 or not _has_two_classes(labels):
        LOGGER.warning("Bootstrap CI is undefined for empty or single-class labels.")
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=n_bootstrap)
    if n_jobs is None:
        raw_values = [
            _bootstrap_metric_value(labels, scores, metric_fn, int(child_seed))
            for child_seed in child_seeds
        ]
    else:
        from joblib import Parallel, delayed

        raw_values = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_bootstrap_metric_value)(labels, scores, metric_fn, int(child_seed))
            for child_seed in child_seeds
        )
    values = [value for value in raw_values if value is not None]

    if not values:
        return float("nan"), float("nan")

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    lower, upper = np.percentile(np.asarray(values, dtype=np.float64), [lower_q, upper_q])
    return float(lower), float(upper)


def _bootstrap_delta_value(
    labels: NDArray[np.int_],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    metric_fn: MetricFn,
    child_seed: int,
) -> float | None:
    """Compute one paired bootstrap delta value, metric(B) - metric(A)."""
    rng = np.random.default_rng(child_seed)
    sample_idx = rng.integers(0, labels.size, labels.size)
    sample_y = labels[sample_idx]
    if not _has_two_classes(sample_y):
        return None
    try:
        value_a = float(metric_fn(sample_y, scores_a[sample_idx]))
        value_b = float(metric_fn(sample_y, scores_b[sample_idx]))
    except (FloatingPointError, ValueError) as exc:
        LOGGER.debug("Skipping invalid paired bootstrap sample: %s", exc)
        return None
    delta = value_b - value_a
    if np.isfinite(delta):
        return float(delta)
    return None


def paired_bootstrap_delta_ci(
    y_true: ArrayLike,
    y_score_a: ArrayLike,
    y_score_b: ArrayLike,
    metric_fn: MetricFn = pr_auc,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_CI_ALPHA,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, float]:
    """Estimate a paired bootstrap CI for ``metric(B) - metric(A)``.

    Args:
        y_true: Binary labels with shape ``(n_fronts,)``.
        y_score_a: Positive-class probabilities for the baseline model.
        y_score_b: Positive-class probabilities for the augmented model.
        metric_fn: Metric function accepting ``(y_true, y_score)`` arrays.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Two-sided interval tail mass.
        seed: Optional RNG seed.
        n_jobs: Optional joblib worker count.

    Returns:
        Dictionary with point delta, percentile CI, bootstrap sign p-value, and
        valid resample count. Invalid input returns NaN-valued statistics.
    """
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive; got {n_bootstrap}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be None or a non-zero integer.")

    labels, scores_a = _validate_binary_inputs(y_true, y_score_a)
    labels_b, scores_b = _validate_binary_inputs(y_true, y_score_b)
    if not np.array_equal(labels, labels_b):
        raise ValueError("Paired bootstrap labels differ between score arrays.")
    if labels.size == 0 or not _has_two_classes(labels):
        LOGGER.warning("Paired bootstrap is undefined for empty or single-class labels.")
        nan = float("nan")
        return {"delta": nan, "ci_low": nan, "ci_high": nan, "p_two_sided": nan, "n_valid": 0.0}

    delta = float(metric_fn(labels, scores_b) - metric_fn(labels, scores_a))
    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=n_bootstrap)
    if n_jobs is None:
        raw_values = [
            _bootstrap_delta_value(labels, scores_a, scores_b, metric_fn, int(child_seed))
            for child_seed in child_seeds
        ]
    else:
        from joblib import Parallel, delayed

        raw_values = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_bootstrap_delta_value)(labels, scores_a, scores_b, metric_fn, int(child_seed))
            for child_seed in child_seeds
        )
    values = np.asarray([value for value in raw_values if value is not None], dtype=np.float64)
    if values.size == 0:
        nan = float("nan")
        return {"delta": delta, "ci_low": nan, "ci_high": nan, "p_two_sided": nan, "n_valid": 0.0}

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(values, [lower_q, upper_q])
    p_nonpositive = float(np.mean(values <= 0.0))
    p_nonnegative = float(np.mean(values >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(p_nonpositive, p_nonnegative))
    return {
        "delta": delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": float(p_two_sided),
        "n_valid": float(values.size),
    }


def _extract_scalar_score(value: Mapping[Any, Any]) -> float | None:
    """Extract a scalar score from a metric-result mapping.

    Args:
        value: Mapping that may contain a score-like scalar.

    Returns:
        A float score when one is found, otherwise ``None``.
    """
    for key in ("score", "value", "pr_auc", "roc_auc", "precision", "metric", "mean"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float, np.integer, np.floating)):
            return float(candidate)
    return None


def _extend_cell_path(prefix: tuple[Any, ...], key: Any) -> tuple[Any, ...]:
    """Append a key to a nested cell path.

    Args:
        prefix: Existing path components.
        key: New dictionary key, possibly a tuple cell key.

    Returns:
        The extended path.
    """
    if isinstance(key, tuple):
        return prefix + key
    return prefix + (key,)


def _cell_key(path: Sequence[Any]) -> tuple[str, ...]:
    """Normalize a cell path to a comparable key.

    Args:
        path: Cell path, typically ``(domain, test_year)``.

    Returns:
        Stringified tuple key.
    """
    return tuple(str(part) for part in path)


def _flatten_cell_scores(scores: CellScores, prefix: tuple[Any, ...] = ()) -> dict[tuple[str, ...], float]:
    """Flatten nested domain/year score dictionaries.

    Args:
        scores: Score mapping. Supported shapes include
            ``{(domain, year): score}``, ``{domain: {year: score}}``, and
            ``{domain: {year: {"pr_auc": score}}}``.
        prefix: Internal recursion prefix.

    Returns:
        Mapping from normalized cell keys to scalar scores.
    """
    flattened: dict[tuple[str, ...], float] = {}
    for key, value in scores.items():
        path = _extend_cell_path(prefix, key)
        if isinstance(value, Mapping):
            scalar = _extract_scalar_score(value)
            if scalar is None:
                flattened.update(_flatten_cell_scores(value, path))
            else:
                flattened[_cell_key(path)] = scalar
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flattened[_cell_key(path)] = float(value)
        else:
            LOGGER.debug("Ignoring non-scalar cell score at %s: %r", path, value)
    return flattened


def paired_wilcoxon(scores_a: dict[Any, Any], scores_b: dict[Any, Any]) -> dict[str, Any]:
    """Run a paired Wilcoxon signed-rank test over domain/year cells.

    Args:
        scores_a: Scores for system A keyed by ``(domain, test_year)`` or nested
            as ``domain -> test_year -> score``.
        scores_b: Scores for system B with the same cell convention.

    Returns:
        Dictionary containing paired cell count, statistic, p-value, mean and
        median deltas, and the common cell keys used.
    """
    flat_a = _flatten_cell_scores(scores_a)
    flat_b = _flatten_cell_scores(scores_b)
    cells = sorted(set(flat_a) & set(flat_b))
    if not cells:
        LOGGER.warning("Paired Wilcoxon received no overlapping cells.")
        return {
            "n_pairs": 0,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "cells": [],
        }

    values_a = np.asarray([flat_a[cell] for cell in cells], dtype=np.float64)
    values_b = np.asarray([flat_b[cell] for cell in cells], dtype=np.float64)
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[finite]
    values_b = values_b[finite]
    used_cells = [cell for cell, keep in zip(cells, finite, strict=True) if keep]
    if values_a.size == 0:
        return {
            "n_pairs": 0,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "cells": [],
        }

    deltas = values_a - values_b
    if np.allclose(deltas, 0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        from scipy.stats import wilcoxon

        test = wilcoxon(values_a, values_b)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    return {
        "n_pairs": int(values_a.size),
        "statistic": statistic,
        "p_value": p_value,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "cells": [list(cell) for cell in used_cells],
    }


def lead_time_gain(results_by_horizon: dict[int, float]) -> dict[str, float]:
    """Summarize lead-time gain from horizon-indexed metric values.

    Args:
        results_by_horizon: Mapping from forecast horizon in years to a scalar
            score such as precision@k. Shape is conceptually (n_horizons,).

    Returns:
        Dictionary with the best horizon, best score, gain in years relative to
        the shortest evaluated horizon, and the maximum possible gain.
    """
    finite_items = sorted(
        (int(horizon), float(score))
        for horizon, score in results_by_horizon.items()
        if np.isfinite(float(score))
    )
    if not finite_items:
        return {
            "best_horizon": float("nan"),
            "best_score": float("nan"),
            "gain_years_vs_shortest": float("nan"),
            "max_possible_gain_years": float("nan"),
        }

    best_horizon, best_score = max(finite_items, key=lambda item: (item[1], item[0]))
    shortest_horizon = min(horizon for horizon, _ in finite_items)
    longest_horizon = max(horizon for horizon, _ in finite_items)
    return {
        "best_horizon": float(best_horizon),
        "best_score": float(best_score),
        "gain_years_vs_shortest": float(best_horizon - shortest_horizon),
        "max_possible_gain_years": float(longest_horizon - shortest_horizon),
    }
