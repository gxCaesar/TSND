"""Calibrated TSND forecasters.

This module ports the pilot LightGBM forecaster and adds an optional
discrete-time survival model. The runtime config is intentionally the source of
all modeling constants; a minimal ``forecaster`` section should define feature
groups, seeds, LightGBM parameters, calibration sizes, confidence-interval
settings, and survival settings when enabled.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tsnd.evaluation.metrics import (
    bootstrap_ci,
    brier_score,
    ece,
    paired_bootstrap_delta_ci,
    pr_auc,
    precision_at_k,
    roc_auc,
)

LOGGER = logging.getLogger(__name__)

FramePairProvider = Callable[[int], tuple[pd.DataFrame, pd.DataFrame]]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load``.

    Args:
        path: Config file path.

    Returns:
        Mutable config dictionary.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config root must be a mapping; got {type(loaded).__name__}.")
    return dict(loaded)


def _format_path(path: Sequence[str]) -> str:
    """Format a config path.

    Args:
        path: Config path components.

    Returns:
        Dot-delimited path string.
    """
    return ".".join(path)


def _lookup_config(config: object, path: Sequence[str]) -> Any:
    """Look up a nested config value in mappings or objects.

    Args:
        config: Mapping-style or object-style config.
        path: Nested path components.

    Returns:
        Config value.

    Raises:
        KeyError: If a path component is missing.
    """
    current: Any = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                raise KeyError(_format_path(path))
            current = current[key]
        else:
            if not hasattr(current, key):
                raise KeyError(_format_path(path))
            current = getattr(current, key)
    return current


def _first_config_value(config: object, paths: Sequence[Sequence[str]]) -> Any:
    """Return the first available config value.

    Args:
        config: Mapping-style or object-style config.
        paths: Candidate config paths.

    Returns:
        First value found.

    Raises:
        KeyError: If no candidate path exists.
    """
    for path in paths:
        try:
            return _lookup_config(config, path)
        except KeyError:
            continue
    joined = " or ".join(_format_path(path) for path in paths)
    raise KeyError(f"Missing required config value: {joined}")


def _as_bool(value: Any, name: str) -> bool:
    """Coerce a config value to bool without string guessing.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        Boolean value.

    Raises:
        TypeError: If the value is not a bool.
    """
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool; got {type(value).__name__}.")


def _as_float(value: Any, name: str) -> float:
    """Coerce a config value to float.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        Float value.

    Raises:
        TypeError: If the value is not numeric.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"{name} must be numeric; got {type(value).__name__}.")


def _as_int(value: Any, name: str) -> int:
    """Coerce a config value to int.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        Integer value.

    Raises:
        TypeError: If the value is not an integer.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")


def _as_str(value: Any, name: str) -> str:
    """Coerce a config value to string.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        String value.

    Raises:
        TypeError: If the value is not a string.
    """
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a string; got {type(value).__name__}.")


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Coerce a sequence config value to an integer tuple.

    Args:
        value: Sequence config value.
        name: Human-readable config path.

    Returns:
        Integer tuple.

    Raises:
        TypeError: If the value is not a non-string sequence.
        ValueError: If the sequence is empty.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers.")
    result = tuple(_as_int(item, name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _as_feature_groups(value: Any, name: str) -> dict[str, tuple[str, ...]]:
    """Coerce feature-group config to a string-column mapping.

    Args:
        value: Mapping from group names to column-name sequences.
        name: Human-readable config path.

    Returns:
        Normalized feature groups.

    Raises:
        TypeError: If the value has the wrong shape.
        ValueError: If no usable groups are defined.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping from group name to columns.")
    groups: dict[str, tuple[str, ...]] = {}
    for group_name, columns in value.items():
        if isinstance(columns, str) or not isinstance(columns, Sequence):
            raise TypeError(f"{name}.{group_name} must be a sequence of column names.")
        normalized_columns = tuple(str(column) for column in columns)
        if not normalized_columns:
            raise ValueError(f"{name}.{group_name} must contain at least one column.")
        groups[str(group_name)] = normalized_columns
    if not groups:
        raise ValueError(f"{name} must define at least one feature group.")
    return groups


def _as_paired_comparisons(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    """Coerce paired-comparison config to ``(baseline, augmented)`` pairs."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of comparison specs.")
    pairs: list[tuple[str, str]] = []
    for idx, item in enumerate(value):
        item_name = f"{name}[{idx}]"
        if isinstance(item, Mapping):
            try:
                baseline = item["baseline"]
                augmented = item["augmented"]
            except KeyError as exc:
                raise KeyError(f"{item_name} must define baseline and augmented.") from exc
            pairs.append(
                (
                    _as_str(baseline, f"{item_name}.baseline"),
                    _as_str(augmented, f"{item_name}.augmented"),
                )
            )
        elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
            pairs.append((_as_str(item[0], f"{item_name}.0"), _as_str(item[1], f"{item_name}.1")))
        else:
            raise TypeError(f"{item_name} must be a mapping or a two-item sequence.")
    return tuple(pairs)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    """Coerce a config value to a plain dictionary.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        Dictionary copy.

    Raises:
        TypeError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(value).__name__}.")
    return dict(value)


def _validate_probability_interval(value: float, name: str) -> None:
    """Validate a strict probability interval value.

    Args:
        value: Probability value.
        name: Human-readable config path.

    Raises:
        ValueError: If the value is outside (0, 1).
    """
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1); got {value}.")


@dataclass(frozen=True)
class ForecasterSettings:
    """Resolved forecaster settings.

    Attributes:
        feature_groups: Mapping from group name to feature columns.
        seeds: Random seeds for the multi-seed evaluation loop.
        label_column: Binary label column in train/test tables, shape
            (n_fronts,).
        calibration_fraction: Fraction of the training cohort reserved for
            isotonic calibration.
        min_calibration_samples: Minimum calibration-split size before isotonic
            calibration is attempted.
        min_cohort_samples: Minimum train/test cohort size accepted for model
            fitting.
        bootstrap_samples: Number of bootstrap resamples for PR-AUC CIs.
        bootstrap_n_jobs: Optional worker count for bootstrap CIs.
        ci_alpha: Two-sided interval tail mass.
        ece_bins: Number of ECE bins.
        precision_k: Top-k cutoff for the early-warning precision metric.
        lightgbm_params: Keyword arguments for ``lightgbm.LGBMClassifier``.
        use_survival_model: Whether to run pycox LogisticHazard.
        survival_horizons: Discrete horizons, typically (1, 2, 3, 4, 5).
        duration_column: Time-to-event column for survival mode.
        event_column: Binary event indicator column for survival mode.
        survival_hidden_layers: Hidden layer widths for LogisticHazard network.
        survival_dropout: Dropout probability for LogisticHazard network.
        survival_batch_norm: Whether to use batch normalization.
        survival_epochs: Training epochs for LogisticHazard.
        survival_batch_size: Batch size for LogisticHazard.
        survival_learning_rate: Learning rate for LogisticHazard optimizer.
        survival_early_stopping_patience: Early-stopping patience; non-positive
            disables the callback.
        survival_verbose: Whether pycox should log epoch progress.
        paired_comparisons: Optional baseline and augmented group pairs for paired
            bootstrap increment tests.
    """

    feature_groups: dict[str, tuple[str, ...]]
    seeds: tuple[int, ...]
    label_column: str
    calibration_fraction: float
    min_calibration_samples: int
    min_cohort_samples: int
    bootstrap_samples: int
    bootstrap_n_jobs: int | None
    ci_alpha: float
    ece_bins: int
    precision_k: int
    lightgbm_params: dict[str, Any]
    use_survival_model: bool
    survival_horizons: tuple[int, ...]
    duration_column: str | None
    event_column: str | None
    survival_hidden_layers: tuple[int, ...]
    survival_dropout: float | None
    survival_batch_norm: bool | None
    survival_epochs: int | None
    survival_batch_size: int | None
    survival_learning_rate: float | None
    survival_early_stopping_patience: int | None
    survival_verbose: bool | None
    paired_comparisons: tuple[tuple[str, str], ...]

    @classmethod
    def from_config(cls, config: object) -> ForecasterSettings:
        """Resolve settings from a runtime config.

        Args:
            config: Mapping-style or object-style config. Required keys are
                under ``forecaster`` except ``seed_list``, which is accepted for
                pilot compatibility.

        Returns:
            Resolved settings object.
        """
        feature_groups = _as_feature_groups(
            _first_config_value(config, (("forecaster", "feature_groups"),)),
            "forecaster.feature_groups",
        )
        seeds = _as_int_tuple(
            _first_config_value(config, (("forecaster", "seeds"), ("seed_list",))),
            "forecaster.seeds",
        )
        label_column = _as_str(
            _first_config_value(config, (("forecaster", "label_column"),)),
            "forecaster.label_column",
        )
        calibration_fraction = _as_float(
            _first_config_value(config, (("forecaster", "calibration_fraction"),)),
            "forecaster.calibration_fraction",
        )
        _validate_probability_interval(
            calibration_fraction,
            "forecaster.calibration_fraction",
        )
        min_calibration_samples = _as_int(
            _first_config_value(config, (("forecaster", "min_calibration_samples"),)),
            "forecaster.min_calibration_samples",
        )
        min_cohort_samples = _as_int(
            _first_config_value(config, (("forecaster", "min_cohort_samples"),)),
            "forecaster.min_cohort_samples",
        )
        bootstrap_samples = _as_int(
            _first_config_value(
                config,
                (("forecaster", "bootstrap_samples"), ("evaluation", "bootstrap_samples")),
            ),
            "forecaster.bootstrap_samples",
        )
        try:
            bootstrap_n_jobs_value = _lookup_config(config, ("compute", "n_process"))
        except KeyError:
            bootstrap_n_jobs = None
        else:
            bootstrap_n_jobs = _as_int(bootstrap_n_jobs_value, "compute.n_process")
        ci_alpha = _as_float(
            _first_config_value(config, (("forecaster", "ci_alpha"), ("evaluation", "ci_alpha"))),
            "forecaster.ci_alpha",
        )
        _validate_probability_interval(ci_alpha, "forecaster.ci_alpha")
        ece_bins = _as_int(
            _first_config_value(config, (("forecaster", "ece_bins"), ("evaluation", "ece_bins"))),
            "forecaster.ece_bins",
        )
        precision_k = _as_int(
            _first_config_value(
                config,
                (("forecaster", "precision_k"), ("evaluation", "precision_k")),
            ),
            "forecaster.precision_k",
        )
        lightgbm_params = _as_mapping(
            _first_config_value(config, (("forecaster", "lightgbm_params"),)),
            "forecaster.lightgbm_params",
        )
        use_survival_model = _as_bool(
            _first_config_value(config, (("forecaster", "use_survival_model"),)),
            "forecaster.use_survival_model",
        )
        paired_comparisons: tuple[tuple[str, str], ...]
        try:
            paired_value = _lookup_config(config, ("forecaster", "paired_comparisons"))
        except KeyError:
            paired_comparisons = ()
        else:
            paired_comparisons = _as_paired_comparisons(
                paired_value,
                "forecaster.paired_comparisons",
            )

        if min_calibration_samples < 0:
            raise ValueError("forecaster.min_calibration_samples must be non-negative.")
        if min_cohort_samples < 2:
            raise ValueError("forecaster.min_cohort_samples must be at least 2.")
        if bootstrap_samples <= 0:
            raise ValueError("forecaster.bootstrap_samples must be positive.")
        if bootstrap_n_jobs == 0:
            raise ValueError("compute.n_process must be non-zero when provided.")
        if ece_bins <= 0:
            raise ValueError("forecaster.ece_bins must be positive.")
        if precision_k <= 0:
            raise ValueError("forecaster.precision_k must be positive.")

        if not use_survival_model:
            return cls(
                feature_groups=feature_groups,
                seeds=seeds,
                label_column=label_column,
                calibration_fraction=calibration_fraction,
                min_calibration_samples=min_calibration_samples,
                min_cohort_samples=min_cohort_samples,
                bootstrap_samples=bootstrap_samples,
                bootstrap_n_jobs=bootstrap_n_jobs,
                ci_alpha=ci_alpha,
                ece_bins=ece_bins,
                precision_k=precision_k,
                lightgbm_params=lightgbm_params,
                use_survival_model=use_survival_model,
                survival_horizons=(),
                duration_column=None,
                event_column=None,
                survival_hidden_layers=(),
                survival_dropout=None,
                survival_batch_norm=None,
                survival_epochs=None,
                survival_batch_size=None,
                survival_learning_rate=None,
                survival_early_stopping_patience=None,
                survival_verbose=None,
                paired_comparisons=paired_comparisons,
            )

        survival_horizons = _as_int_tuple(
            _first_config_value(config, (("forecaster", "survival_horizons"),)),
            "forecaster.survival_horizons",
        )
        duration_column = _as_str(
            _first_config_value(config, (("forecaster", "duration_column"),)),
            "forecaster.duration_column",
        )
        event_column = _as_str(
            _first_config_value(config, (("forecaster", "event_column"),)),
            "forecaster.event_column",
        )
        survival_hidden_layers = _as_int_tuple(
            _first_config_value(config, (("forecaster", "survival_hidden_layers"),)),
            "forecaster.survival_hidden_layers",
        )
        survival_dropout = _as_float(
            _first_config_value(config, (("forecaster", "survival_dropout"),)),
            "forecaster.survival_dropout",
        )
        survival_batch_norm = _as_bool(
            _first_config_value(config, (("forecaster", "survival_batch_norm"),)),
            "forecaster.survival_batch_norm",
        )
        survival_epochs = _as_int(
            _first_config_value(config, (("forecaster", "survival_epochs"),)),
            "forecaster.survival_epochs",
        )
        survival_batch_size = _as_int(
            _first_config_value(config, (("forecaster", "survival_batch_size"),)),
            "forecaster.survival_batch_size",
        )
        survival_learning_rate = _as_float(
            _first_config_value(config, (("forecaster", "survival_learning_rate"),)),
            "forecaster.survival_learning_rate",
        )
        survival_early_stopping_patience = _as_int(
            _first_config_value(config, (("forecaster", "survival_early_stopping_patience"),)),
            "forecaster.survival_early_stopping_patience",
        )
        survival_verbose = _as_bool(
            _first_config_value(config, (("forecaster", "survival_verbose"),)),
            "forecaster.survival_verbose",
        )

        if any(horizon <= 0 for horizon in survival_horizons):
            raise ValueError("forecaster.survival_horizons must contain positive years.")
        if not 0.0 <= survival_dropout < 1.0:
            raise ValueError("forecaster.survival_dropout must be in [0, 1).")
        if survival_epochs <= 0:
            raise ValueError("forecaster.survival_epochs must be positive.")
        if survival_batch_size <= 0:
            raise ValueError("forecaster.survival_batch_size must be positive.")
        if survival_learning_rate <= 0.0:
            raise ValueError("forecaster.survival_learning_rate must be positive.")

        return cls(
            feature_groups=feature_groups,
            seeds=seeds,
            label_column=label_column,
            calibration_fraction=calibration_fraction,
            min_calibration_samples=min_calibration_samples,
            min_cohort_samples=min_cohort_samples,
            bootstrap_samples=bootstrap_samples,
            bootstrap_n_jobs=bootstrap_n_jobs,
            ci_alpha=ci_alpha,
            ece_bins=ece_bins,
            precision_k=precision_k,
            lightgbm_params=lightgbm_params,
            use_survival_model=use_survival_model,
            survival_horizons=survival_horizons,
            duration_column=duration_column,
            event_column=event_column,
            survival_hidden_layers=survival_hidden_layers,
            survival_dropout=survival_dropout,
            survival_batch_norm=survival_batch_norm,
            survival_epochs=survival_epochs,
            survival_batch_size=survival_batch_size,
            survival_learning_rate=survival_learning_rate,
            survival_early_stopping_patience=survival_early_stopping_patience,
            survival_verbose=survival_verbose,
            paired_comparisons=paired_comparisons,
        )


@dataclass(frozen=True)
class GroupFitResult:
    """Prediction result for one feature group.

    Attributes:
        status: ``"ok"`` or ``"skipped"``.
        reason: Skip reason when status is ``"skipped"``.
        y_true: Binary labels with shape (n_test_fronts,).
        y_score: Positive-class probabilities with shape (n_test_fronts,).
        metrics: Scalar metrics for the group.
    """

    status: str
    reason: str | None
    y_true: NDArray[np.int_]
    y_score: NDArray[np.float64]
    metrics: dict[str, float]

    def summary(self, ci: tuple[float, float]) -> dict[str, Any]:
        """Return a JSON-serializable summary.

        Args:
            ci: PR-AUC bootstrap confidence interval.

        Returns:
            Summary dictionary without raw prediction arrays.
        """
        payload: dict[str, Any] = {
            "status": self.status,
            "ci": [float(ci[0]), float(ci[1])],
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        payload.update({key: float(value) for key, value in self.metrics.items()})
        return payload


class Forecaster:
    """Run calibrated LightGBM and optional survival forecasters."""

    def __init__(self, config: object) -> None:
        """Create a forecaster from config.

        Args:
            config: Mapping-style or object-style config.
        """
        self.config = config
        self.settings = ForecasterSettings.from_config(config)

    def fit_evaluate(self, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        """Fit/evaluate all feature groups using the same tables for each seed.

        Args:
            train: Training feature table with shape (n_train_fronts, n_columns).
            test: Test feature table with shape (n_test_fronts, n_columns).

        Returns:
            Nested result dictionary with per-seed metrics and aggregate means.
        """
        return self.fit_evaluate_by_seed(lambda _seed: (train, test))

    def fit_evaluate_by_seed(self, table_provider: FramePairProvider) -> dict[str, Any]:
        """Fit/evaluate all feature groups with seed-specific tables.

        Args:
            table_provider: Callable returning ``(train, test)`` tables for a
                seed. Each table has shape (n_fronts, n_columns).

        Returns:
            Nested result dictionary with per-seed metrics, CIs, and aggregate
            means/stds.
        """
        return run_forecaster(table_provider, self.settings)


def _set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _missing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Find missing columns in a table.

    Args:
        frame: Feature table with shape (n_fronts, n_columns).
        columns: Required column names.

    Returns:
        Missing column names.
    """
    return [column for column in columns if column not in frame.columns]


def _label_array(frame: pd.DataFrame, label_column: str) -> NDArray[np.int_]:
    """Extract and validate a binary label column.

    Args:
        frame: Feature table with shape (n_fronts, n_columns).
        label_column: Binary label column name.

    Returns:
        Integer labels with shape (n_fronts,).

    Raises:
        ValueError: If labels are missing, non-finite, or non-binary.
    """
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}.")
    values = frame[label_column].to_numpy()
    if values.ndim != 1:
        raise ValueError(f"Label column {label_column} must be one-dimensional.")
    labels_float = values.astype(np.float64)
    if not np.all(np.isfinite(labels_float)):
        raise ValueError(f"Label column {label_column} contains non-finite values.")
    if not np.all(np.isin(labels_float, [0.0, 1.0])):
        raise ValueError(f"Label column {label_column} must contain 0/1 labels.")
    return np.asarray(labels_float, dtype=np.int_)


def _safe_label_summary(frame: pd.DataFrame, label_column: str) -> dict[str, int | None]:
    """Summarize labels without raising for missing columns.

    Args:
        frame: Feature table with shape (n_fronts, n_columns).
        label_column: Binary label column name.

    Returns:
        Count summary.
    """
    if label_column not in frame.columns:
        return {"positives": None, "negatives": None}
    try:
        labels = _label_array(frame, label_column)
    except ValueError as exc:
        LOGGER.warning("Could not summarize labels: %s", exc)
        return {"positives": None, "negatives": None}
    positives = int(np.sum(labels))
    return {"positives": positives, "negatives": int(labels.size - positives)}


def _feature_array(frame: pd.DataFrame, columns: Sequence[str]) -> NDArray[np.float64]:
    """Extract numeric feature columns.

    Args:
        frame: Feature table with shape (n_fronts, n_columns).
        columns: Feature column names.

    Returns:
        Feature matrix with shape (n_fronts, n_features).

    Raises:
        ValueError: If columns cannot be converted to float64.
    """
    try:
        return np.asarray(
            frame.loc[:, list(columns)].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(f"Feature columns must be numeric: {columns}.") from exc


def _skip_result(reason: str, y_true: NDArray[np.int_] | None = None) -> GroupFitResult:
    """Create a skipped group result.

    Args:
        reason: Human-readable skip reason.
        y_true: Optional test labels with shape (n_test_fronts,).

    Returns:
        Skipped result with ``nan`` metrics.
    """
    labels = np.asarray([], dtype=int) if y_true is None else y_true
    scores = np.full(labels.shape[0], float("nan"), dtype=np.float64)
    return GroupFitResult(
        status="skipped",
        reason=reason,
        y_true=labels,
        y_score=scores,
        metrics={
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "precision_at_k": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        },
    )


def _cohort_skip_reason(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    settings: ForecasterSettings,
) -> tuple[str | None, NDArray[np.int_] | None, NDArray[np.int_] | None]:
    """Detect degenerate train/test cohorts.

    Args:
        train: Training table with shape (n_train_fronts, n_columns).
        test: Test table with shape (n_test_fronts, n_columns).
        columns: Feature columns for the current group.
        settings: Resolved forecaster settings.

    Returns:
        ``(reason, train_labels, test_labels)``. ``reason`` is ``None`` when
        fitting may proceed.
    """
    missing = _missing_columns(train, columns) + _missing_columns(test, columns)
    if missing:
        return f"missing feature columns: {sorted(set(missing))}", None, None

    if len(train) < settings.min_cohort_samples:
        return f"train cohort has fewer than {settings.min_cohort_samples} samples", None, None
    if len(test) < settings.min_cohort_samples:
        return f"test cohort has fewer than {settings.min_cohort_samples} samples", None, None

    try:
        y_train = _label_array(train, settings.label_column)
        y_test = _label_array(test, settings.label_column)
    except ValueError as exc:
        return str(exc), None, None

    if np.unique(y_train).size < 2:
        return "train cohort has a single label class", y_train, y_test
    if np.unique(y_test).size < 2:
        return "test cohort has a single label class", y_train, y_test

    x_train = _feature_array(train, columns)
    x_test = _feature_array(test, columns)
    if np.all(np.isnan(x_train)) or np.all(np.isnan(x_test)):
        return "feature matrix is all-NaN for train or test cohort", y_train, y_test
    return None, y_train, y_test


def _split_fit_calibration(
    n_samples: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Split training rows into fit and calibration indices.

    Args:
        n_samples: Number of training rows.
        calibration_fraction: Fraction held out for calibration.
        seed: Random seed.

    Returns:
        ``(fit_indices, calibration_indices)`` with shapes (n_fit,) and
        (n_calibration,).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    calibration_size = int(np.floor(calibration_fraction * n_samples))
    fit_size = n_samples - calibration_size
    if fit_size <= 0:
        return indices, np.asarray([], dtype=int)
    return indices[:fit_size], indices[fit_size:]


def _positive_class_scores(model: Any, x_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract positive-class probabilities from a sklearn-like classifier.

    Args:
        model: Fitted classifier exposing ``classes_`` and ``predict_proba``.
        x_matrix: Feature matrix with shape (n_fronts, n_features).

    Returns:
        Positive-class probabilities with shape (n_fronts,).

    Raises:
        ValueError: If class 1 is unavailable.
    """
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Fitted classifier does not expose class 1: {classes}.")
    positive_idx = classes.index(1)
    probabilities = model.predict_proba(x_matrix)
    return np.asarray(probabilities[:, positive_idx], dtype=np.float64)


def _metric_summary(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    settings: ForecasterSettings,
) -> dict[str, float]:
    """Compute scalar metrics for a prediction vector.

    Args:
        y_true: Binary labels with shape (n_fronts,).
        y_score: Positive-class probabilities with shape (n_fronts,).
        settings: Resolved forecaster settings.

    Returns:
        Metric dictionary.
    """
    metrics = {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "precision_at_k": precision_at_k(y_true, y_score, settings.precision_k),
        "brier": brier_score(y_true, y_score),
    }
    try:
        metrics["ece"] = ece(y_true, y_score, settings.ece_bins)
    except ImportError as exc:
        LOGGER.warning("netcal is unavailable; ECE set to nan: %s", exc)
        metrics["ece"] = float("nan")
    return metrics


def _fit_eval_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    settings: ForecasterSettings,
) -> GroupFitResult:
    """Fit a calibrated LightGBM for one feature group.

    Args:
        train: Training table with shape (n_train_fronts, n_columns).
        test: Test table with shape (n_test_fronts, n_columns).
        columns: Feature columns for the current group.
        seed: Random seed.
        settings: Resolved forecaster settings.

    Returns:
        Group prediction result. Degenerate cohorts return ``status="skipped"``.
    """
    reason, y_train, y_test = _cohort_skip_reason(train, test, columns, settings)
    if reason is not None:
        LOGGER.warning("Skipping feature group with columns %s: %s", columns, reason)
        return _skip_result(reason, y_test)
    if y_train is None or y_test is None:
        return _skip_result("label validation failed")

    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    _set_seed(seed)
    x_train = _feature_array(train, columns)
    x_test = _feature_array(test, columns)
    fit_idx, cal_idx = _split_fit_calibration(
        len(x_train),
        settings.calibration_fraction,
        seed,
    )
    if np.unique(y_train[fit_idx]).size < 2:
        LOGGER.warning(
            "Seed %s fit split is single-class; fitting on full train cohort without calibration.",
            seed,
        )
        fit_idx = np.arange(len(x_train))
        cal_idx = np.asarray([], dtype=int)

    params = dict(settings.lightgbm_params)
    params["random_state"] = seed
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train[fit_idx], y_train[fit_idx])

    raw_test = _positive_class_scores(model, x_test)
    calibrated_test = raw_test
    if cal_idx.size >= settings.min_calibration_samples and np.unique(y_train[cal_idx]).size == 2:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(_positive_class_scores(model, x_train[cal_idx]), y_train[cal_idx])
            calibrated_test = np.asarray(calibrator.transform(raw_test), dtype=np.float64)
        except ValueError as exc:
            LOGGER.warning("Isotonic calibration failed; using raw probabilities: %s", exc)
    else:
        LOGGER.info(
            "Skipping isotonic calibration for seed %s; calibration size=%s classes=%s.",
            seed,
            cal_idx.size,
            np.unique(y_train[cal_idx]).tolist() if cal_idx.size else [],
        )

    calibrated_test = np.clip(calibrated_test, 0.0, 1.0)
    return GroupFitResult(
        status="ok",
        reason=None,
        y_true=y_test,
        y_score=calibrated_test,
        metrics=_metric_summary(y_test, calibrated_test, settings),
    )


def _finite_metric_values(
    per_seed: Sequence[dict[str, Any]],
    group_name: str,
    metric_name: str,
) -> list[float]:
    """Collect finite metric values across seeds.

    Args:
        per_seed: Per-seed result payloads.
        group_name: Feature group name.
        metric_name: Metric key.

    Returns:
        Finite metric values.
    """
    values: list[float] = []
    for seed_result in per_seed:
        group_result = seed_result.get("groups", {}).get(group_name, {})
        value = group_result.get(metric_name)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            values.append(float(value))
    return values


def _aggregate_groups(
    per_seed: Sequence[dict[str, Any]],
    settings: ForecasterSettings,
) -> dict[str, dict[str, float]]:
    """Aggregate feature-group metrics across seeds.

    Args:
        per_seed: Per-seed result payloads.
        settings: Resolved forecaster settings.

    Returns:
        Mapping from group to mean/std metrics.
    """
    aggregate: dict[str, dict[str, float]] = {}
    metric_names = ("pr_auc", "roc_auc", "precision_at_k", "brier", "ece")
    for group_name in settings.feature_groups:
        group_summary: dict[str, float] = {}
        for metric_name in metric_names:
            values = _finite_metric_values(per_seed, group_name, metric_name)
            group_summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else float("nan")
            group_summary[f"{metric_name}_std"] = float(np.std(values)) if values else float("nan")
            if metric_name == "pr_auc":
                group_summary["n_valid_seeds"] = float(len(values))
        aggregate[group_name] = group_summary
    return aggregate


def _comparison_key(baseline: str, augmented: str) -> str:
    """Return the stable JSON key for an augmented-minus-baseline comparison."""
    return f"{augmented}_minus_{baseline}"


def _paired_summary(
    baseline: GroupFitResult | None,
    augmented: GroupFitResult | None,
    seed: int,
    settings: ForecasterSettings,
) -> dict[str, Any]:
    """Compute one seed-level paired PR-AUC increment summary."""
    if baseline is None or augmented is None:
        return {"status": "skipped", "reason": "missing baseline or augmented group"}
    if baseline.status != "ok" or augmented.status != "ok":
        return {
            "status": "skipped",
            "reason": f"non-ok group status: baseline={baseline.status}, augmented={augmented.status}",
        }
    if not np.array_equal(baseline.y_true, augmented.y_true):
        return {"status": "skipped", "reason": "test labels differ between paired groups"}
    try:
        stats = paired_bootstrap_delta_ci(
            baseline.y_true,
            baseline.y_score,
            augmented.y_score,
            pr_auc,
            n_bootstrap=settings.bootstrap_samples,
            alpha=settings.ci_alpha,
            seed=seed,
            n_jobs=settings.bootstrap_n_jobs,
        )
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc)}
    return {
        "status": "ok",
        "metric": "pr_auc",
        "delta": stats["delta"],
        "ci": [stats["ci_low"], stats["ci_high"]],
        "p_two_sided": stats["p_two_sided"],
        "n_valid_bootstrap": stats["n_valid"],
    }


def _aggregate_paired(
    per_seed: Sequence[dict[str, Any]],
    settings: ForecasterSettings,
) -> dict[str, dict[str, Any]]:
    """Aggregate seed-level paired increment summaries."""
    aggregate: dict[str, dict[str, Any]] = {}
    for baseline, augmented in settings.paired_comparisons:
        key = _comparison_key(baseline, augmented)
        rows: list[dict[str, Any]] = []
        for seed_result in per_seed:
            row = seed_result.get("paired", {}).get(key, {})
            if row.get("status") == "ok" and isinstance(row.get("delta"), (int, float, np.floating)):
                if np.isfinite(float(row["delta"])):
                    rows.append(row)
        deltas = np.asarray([float(row["delta"]) for row in rows], dtype=np.float64)
        aggregate[key] = {
            "baseline": baseline,
            "augmented": augmented,
            "metric": "pr_auc",
            "delta_mean": float(np.mean(deltas)) if deltas.size else float("nan"),
            "delta_std": float(np.std(deltas)) if deltas.size else float("nan"),
            "n_valid_seeds": float(deltas.size),
            "all_positive": bool(deltas.size > 0 and np.all(deltas > 0.0)),
            "seed_ci_low_min": float(np.min([row["ci"][0] for row in rows])) if rows else float("nan"),
            "seed_ci_high_max": float(np.max([row["ci"][1] for row in rows])) if rows else float("nan"),
            "p_two_sided_max": float(np.max([row["p_two_sided"] for row in rows])) if rows else float("nan"),
        }
    return aggregate


def _survival_skip(reason: str) -> dict[str, Any]:
    """Create a skipped survival-model result.

    Args:
        reason: Human-readable skip reason.

    Returns:
        Survival result dictionary.
    """
    return {"status": "skipped", "reason": reason, "horizons": {}}


def _survival_columns_present(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    settings: ForecasterSettings,
) -> str | None:
    """Validate columns required by survival mode.

    Args:
        train: Training table with shape (n_train_fronts, n_columns).
        test: Test table with shape (n_test_fronts, n_columns).
        columns: Feature columns.
        settings: Resolved forecaster settings.

    Returns:
        Skip reason, or ``None`` if required columns exist.
    """
    required = list(columns)
    if settings.duration_column is not None:
        required.append(settings.duration_column)
    if settings.event_column is not None:
        required.append(settings.event_column)
    missing = _missing_columns(train, required) + _missing_columns(test, required)
    if missing:
        return f"missing survival columns: {sorted(set(missing))}"
    return None


def _duration_event_arrays(
    frame: pd.DataFrame,
    duration_column: str,
    event_column: str,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Extract survival duration/event arrays.

    Args:
        frame: Feature table with shape (n_fronts, n_columns).
        duration_column: Time-to-event column.
        event_column: Binary event indicator column.

    Returns:
        ``(durations, events)`` with shapes (n_fronts,) and (n_fronts,).

    Raises:
        ValueError: If values are invalid.
    """
    durations = frame[duration_column].to_numpy(dtype=np.float64)
    events_float = frame[event_column].to_numpy(dtype=np.float64)
    if durations.ndim != 1 or events_float.ndim != 1:
        raise ValueError("Survival duration and event columns must be one-dimensional.")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("Survival durations must be finite positive values.")
    if not np.all(np.isin(events_float, [0.0, 1.0])):
        raise ValueError("Survival events must be binary 0/1 values.")
    return durations, events_float.astype(int)


def _impute_for_survival(
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Median-impute feature matrices for neural survival training.

    Args:
        x_train: Training features with shape (n_train_fronts, n_features).
        x_test: Test features with shape (n_test_fronts, n_features).

    Returns:
        Float32 imputed matrices with the same shapes.
    """
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_imputed = np.where(np.isnan(x_train), medians, x_train).astype(np.float32)
    test_imputed = np.where(np.isnan(x_test), medians, x_test).astype(np.float32)
    return train_imputed, test_imputed


def _binary_event_by_horizon(
    durations: NDArray[np.float64],
    events: NDArray[np.int_],
    horizon: int,
) -> NDArray[np.int_]:
    """Convert survival labels to a horizon-specific binary target.

    Args:
        durations: Event/censoring times with shape (n_fronts,).
        events: Binary event indicators with shape (n_fronts,).
        horizon: Forecast horizon in years.

    Returns:
        Binary event-by-horizon labels with shape (n_fronts,).
    """
    return np.asarray(
        (events == 1) & (durations <= float(horizon)),
        dtype=np.int_,
    )


def _fit_eval_survival(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    settings: ForecasterSettings,
) -> dict[str, Any]:
    """Fit/evaluate pycox LogisticHazard for one feature group.

    Args:
        train: Training table with shape (n_train_fronts, n_columns).
        test: Test table with shape (n_test_fronts, n_columns).
        columns: Feature columns for the current group.
        seed: Random seed.
        settings: Resolved forecaster settings.

    Returns:
        Survival result dictionary keyed by horizon.
    """
    if settings.duration_column is None or settings.event_column is None:
        return _survival_skip("survival duration/event columns are not configured")

    missing_reason = _survival_columns_present(train, test, columns, settings)
    if missing_reason is not None:
        LOGGER.warning("Skipping survival model: %s", missing_reason)
        return _survival_skip(missing_reason)
    if len(train) < settings.min_cohort_samples or len(test) < settings.min_cohort_samples:
        return _survival_skip("train or test cohort is smaller than min_cohort_samples")

    try:
        import torch
        import torchtuples as tt
        from pycox.models import LogisticHazard
    except ImportError as exc:
        LOGGER.warning("Skipping survival model because pycox stack is unavailable: %s", exc)
        return _survival_skip(f"missing survival dependency: {exc}")

    if (
        settings.survival_dropout is None
        or settings.survival_batch_norm is None
        or settings.survival_epochs is None
        or settings.survival_batch_size is None
        or settings.survival_learning_rate is None
        or settings.survival_early_stopping_patience is None
        or settings.survival_verbose is None
    ):
        return _survival_skip("survival settings are incomplete")

    _set_seed(seed)
    torch.manual_seed(seed)
    try:
        durations_train, events_train = _duration_event_arrays(
            train,
            settings.duration_column,
            settings.event_column,
        )
        durations_test, events_test = _duration_event_arrays(
            test,
            settings.duration_column,
            settings.event_column,
        )
        x_train = _feature_array(train, columns)
        x_test = _feature_array(test, columns)
    except ValueError as exc:
        return _survival_skip(str(exc))

    if np.unique(events_train).size < 2:
        return _survival_skip("training survival events have a single class")

    x_train32, x_test32 = _impute_for_survival(x_train, x_test)
    fit_idx, cal_idx = _split_fit_calibration(
        x_train32.shape[0],
        settings.calibration_fraction,
        seed,
    )
    if np.unique(events_train[fit_idx]).size < 2:
        fit_idx = np.arange(x_train32.shape[0])
        cal_idx = np.asarray([], dtype=int)

    horizon_cuts = np.asarray(settings.survival_horizons, dtype=np.float32)
    labtrans = LogisticHazard.label_transform(horizon_cuts)
    y_fit = labtrans.fit_transform(durations_train[fit_idx], events_train[fit_idx])
    val_data = None
    if cal_idx.size >= settings.min_calibration_samples:
        y_cal = labtrans.transform(durations_train[cal_idx], events_train[cal_idx])
        val_data = (x_train32[cal_idx], y_cal)

    net = tt.practical.MLPVanilla(
        in_features=x_train32.shape[1],
        num_nodes=list(settings.survival_hidden_layers),
        out_features=labtrans.out_features,
        batch_norm=settings.survival_batch_norm,
        dropout=settings.survival_dropout,
    )
    optimizer = tt.optim.Adam(settings.survival_learning_rate)
    model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
    callbacks: list[Any] = []
    if settings.survival_early_stopping_patience > 0:
        callbacks.append(
            tt.callbacks.EarlyStopping(
                patience=settings.survival_early_stopping_patience,
            )
        )
    model.fit(
        x_train32[fit_idx],
        y_fit,
        batch_size=settings.survival_batch_size,
        epochs=settings.survival_epochs,
        callbacks=callbacks,
        verbose=settings.survival_verbose,
        val_data=val_data,
    )

    survival_df = model.predict_surv_df(x_test32)
    duration_index = np.asarray(survival_df.index, dtype=np.float64)
    horizons: dict[str, dict[str, Any]] = {}
    for horizon in settings.survival_horizons:
        row_idx = int(np.searchsorted(duration_index, float(horizon), side="right") - 1)
        if row_idx < 0:
            horizons[str(horizon)] = {
                "status": "skipped",
                "reason": "horizon precedes pycox duration index",
            }
            continue
        risk = np.asarray(1.0 - survival_df.iloc[row_idx].to_numpy(dtype=np.float64))
        risk = np.clip(risk, 0.0, 1.0)
        labels = _binary_event_by_horizon(durations_test, events_test, horizon)
        if np.unique(labels).size < 2:
            horizons[str(horizon)] = {
                "status": "skipped",
                "reason": "test labels are single-class at horizon",
                "pr_auc": float("nan"),
                "ci": [float("nan"), float("nan")],
            }
            continue
        metrics = _metric_summary(labels, risk, settings)
        ci = bootstrap_ci(
            labels,
            risk,
            pr_auc,
            n_bootstrap=settings.bootstrap_samples,
            alpha=settings.ci_alpha,
            seed=seed,
            n_jobs=settings.bootstrap_n_jobs,
        )
        horizons[str(horizon)] = {
            "status": "ok",
            **metrics,
            "ci": [float(ci[0]), float(ci[1])],
        }

    return {"status": "ok", "horizons": horizons}


def _aggregate_survival(
    per_seed: Sequence[dict[str, Any]],
    settings: ForecasterSettings,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate survival metrics across seeds.

    Args:
        per_seed: Per-seed result payloads.
        settings: Resolved forecaster settings.

    Returns:
        Nested mapping from group to horizon to mean/std metrics.
    """
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = ("pr_auc", "roc_auc", "precision_at_k", "brier", "ece")
    for group_name in settings.feature_groups:
        aggregate[group_name] = {}
        for horizon in settings.survival_horizons:
            horizon_key = str(horizon)
            summary: dict[str, float] = {}
            for metric_name in metric_names:
                values: list[float] = []
                for seed_result in per_seed:
                    value = (
                        seed_result.get("survival", {})
                        .get(group_name, {})
                        .get("horizons", {})
                        .get(horizon_key, {})
                        .get(metric_name)
                    )
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
                        float(value)
                    ):
                        values.append(float(value))
                summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else float("nan")
                summary[f"{metric_name}_std"] = float(np.std(values)) if values else float("nan")
            aggregate[group_name][horizon_key] = summary
    return aggregate


def run_forecaster(
    table_provider: FramePairProvider,
    settings: ForecasterSettings,
) -> dict[str, Any]:
    """Run the multi-seed, per-feature-group forecaster loop.

    Args:
        table_provider: Callable returning ``(train, test)`` tables for each
            seed. Each table has shape (n_fronts, n_columns).
        settings: Resolved forecaster settings.

    Returns:
        Nested result dictionary with CIs and seed aggregates.
    """
    output: dict[str, Any] = {"per_seed": [], "groups": {}}
    if settings.paired_comparisons:
        output["paired"] = {}
    if settings.use_survival_model:
        output["survival_groups"] = {}

    for seed in settings.seeds:
        train, test = table_provider(seed)
        label_summary = _safe_label_summary(test, settings.label_column)
        seed_result: dict[str, Any] = {
            "seed": int(seed),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_positives": label_summary["positives"],
            "test_negatives": label_summary["negatives"],
            "groups": {},
        }
        if settings.use_survival_model:
            seed_result["survival"] = {}
        if settings.paired_comparisons:
            seed_result["paired"] = {}

        group_results: dict[str, GroupFitResult] = {}
        for group_name, columns in settings.feature_groups.items():
            result = _fit_eval_lightgbm(train, test, columns, seed, settings)
            group_results[group_name] = result
            ci = (
                bootstrap_ci(
                    result.y_true,
                    result.y_score,
                    pr_auc,
                    n_bootstrap=settings.bootstrap_samples,
                    alpha=settings.ci_alpha,
                    seed=seed,
                    n_jobs=settings.bootstrap_n_jobs,
                )
                if result.status == "ok"
                else (float("nan"), float("nan"))
            )
            summary = result.summary(ci)
            seed_result["groups"][group_name] = summary
            seed_result[group_name] = summary

            if settings.use_survival_model:
                seed_result["survival"][group_name] = _fit_eval_survival(
                    train,
                    test,
                    columns,
                    seed,
                    settings,
                )

        for baseline, augmented in settings.paired_comparisons:
            key = _comparison_key(baseline, augmented)
            seed_result["paired"][key] = _paired_summary(
                group_results.get(baseline),
                group_results.get(augmented),
                seed,
                settings,
            )

        LOGGER.info(
            "seed=%s group_pr_auc=%s",
            seed,
            {
                group_name: seed_result["groups"][group_name].get("pr_auc")
                for group_name in settings.feature_groups
            },
        )
        output["per_seed"].append(seed_result)

    output["groups"] = _aggregate_groups(output["per_seed"], settings)
    if settings.paired_comparisons:
        output["paired"] = _aggregate_paired(output["per_seed"], settings)
    if settings.use_survival_model:
        output["survival_groups"] = _aggregate_survival(output["per_seed"], settings)
    return output


def run_from_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: object,
) -> dict[str, Any]:
    """Convenience wrapper for non-seed-specific feature tables.

    Args:
        train: Training feature table with shape (n_train_fronts, n_columns).
        test: Test feature table with shape (n_test_fronts, n_columns).
        config: Mapping-style or object-style config.

    Returns:
        Nested result dictionary with per-seed metrics and aggregate means.
    """
    return Forecaster(config).fit_evaluate(train, test)


def run_from_provider(
    table_provider: FramePairProvider,
    config: object,
) -> dict[str, Any]:
    """Convenience wrapper for seed-specific table providers.

    Args:
        table_provider: Callable returning ``(train, test)`` tables for each
            seed. Each table has shape (n_fronts, n_columns).
        config: Mapping-style or object-style config.

    Returns:
        Nested result dictionary with per-seed metrics and aggregate means.
    """
    return Forecaster(config).fit_evaluate_by_seed(table_provider)
