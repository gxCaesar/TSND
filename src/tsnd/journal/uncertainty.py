"""Fixed-prediction sensitivity to dependence between fronts sharing concepts.

One exponential multiplier is drawn for each concept. A pair receives the product
of its endpoint multipliers, shared across cohorts, feature groups, and fitted
model seeds. Percentiles describe sensitivity conditional on the fitted predictors
and retained corpus, rather than guaranteed coverage for a selected network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score

ID_COLUMNS = ["t", "kind", "key"]
CELL_COLUMNS = ["condition", "cohort", "analysis"]
PREDICTION_COLUMNS = [
    *ID_COLUMNS, "label", "seed", "group", "score", "condition", "cohort", "prox_defined",
]


class FixedScoreAveragePrecision:
    """Cache fixed score ranks and ties for sklearn-equivalent weighted binary AP.

    Args:
        y_true: One-dimensional binary labels for the fixed prediction rows.
        y_score: One-dimensional finite scores in the same row order.

    Raises:
        ValueError: If labels or scores are empty, malformed, or incompatible.
    """

    def __init__(self, y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> None:
        labels = np.asarray(y_true)
        scores = np.asarray(y_score, dtype=np.float64)
        if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
            raise ValueError("AP labels and scores must be nonempty aligned one-dimensional arrays")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("AP labels must be binary")
        if not np.isfinite(scores).all():
            raise ValueError("AP scores must be finite")
        # The reversed stable ascending order matches sklearn's binary curve helper.
        self._order = np.argsort(scores, kind="mergesort")[::-1]
        ordered_scores = scores[self._order]
        self._positive = (labels[self._order] == 1).astype(np.float64)
        self._threshold_ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]

    def evaluate(self, sample_weight: npt.ArrayLike) -> float:
        """Compute noninterpolated weighted AP without re-sorting fixed scores.

        Precision and recall are evaluated only after complete equal-score groups.
        Zero-weight rows contribute no mass, as in sklearn's weighted binary curve.

        Args:
            sample_weight: Finite nonnegative weights aligned to the original rows.

        Returns:
            The same binary average precision as ``average_precision_score`` up to
            floating-point summation precision. No positive mass gives zero AP.

        Raises:
            ValueError: If weight dimensions, values, or total mass are invalid.
        """
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or weights.shape != self._order.shape:
            raise ValueError("AP weights must be one-dimensional and aligned to the fixed score rows")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("AP weights must be finite and nonnegative")
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("AP weights must have finite positive total mass")
        ordered_weight = weights[self._order]
        true_positives = np.cumsum(ordered_weight * self._positive)[self._threshold_ends]
        if true_positives[-1] == 0:
            return 0.0
        false_positives = np.cumsum(ordered_weight * (1.0 - self._positive))[self._threshold_ends]
        denominator = true_positives + false_positives
        precision = np.divide(
            true_positives, denominator, out=np.zeros_like(true_positives), where=denominator != 0,
        )
        recall = true_positives / true_positives[-1]
        # sklearn reverses its binary curve and appends recall=0 before this sum.
        return float(-np.sum(np.diff(np.r_[recall[::-1], 0.0]) * precision[::-1]))


def canonical_front_ids(frame: pd.DataFrame, source: str = "front table") -> pd.DataFrame:
    """Validate front IDs and canonicalize unordered pair endpoints.

    Args:
        frame: Table with ``t``, ``kind``, and ``key`` columns.
        source: Name used in diagnostics.

    Returns:
        Copy with integer years and canonical pair keys; other columns are retained.

    Raises:
        ValueError: If identifiers are missing, malformed, or duplicated.
    """
    missing = [column for column in ID_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{source}: missing ID columns {missing}")
    out = frame.copy()
    if out[ID_COLUMNS].isna().any().any():
        raise ValueError(f"{source}: missing front IDs")
    years = pd.to_numeric(out["t"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(years).all() or (years <= 0).any() or (years != np.floor(years)).any():
        raise ValueError(f"{source}: years must be positive integers")
    out["t"] = years.astype(np.int64)
    if not out["kind"].isin(["term", "pair"]).all():
        raise ValueError(f"{source}: kind must be term or pair")
    keys: list[str] = []
    for kind, key in zip(out["kind"], out["key"], strict=True):
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError(f"{source}: keys must be nonempty strings without outer whitespace")
        if kind == "pair":
            units = key.split("|")
            if len(units) != 2 or not all(units) or units[0] == units[1]:
                raise ValueError(f"{source}: malformed pair key {key!r}")
            key = "|".join(sorted(units))
        keys.append(key)
    out["key"] = keys
    if out.duplicated(ID_COLUMNS).any():
        raise ValueError(f"{source}: duplicate front IDs after canonicalizing pair endpoints")
    return out


def _validate_labels(labels: pd.Series, source: str) -> None:
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be nonmissing binary values")


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate prediction schema and exact alignment within each analysis.

    All model groups and seeds in a condition/cohort/analysis must score the same
    front IDs, labels, and proximity-availability flags. NaN proximity is represented
    explicitly by ``prox_defined=False`` and must not remove a row.

    Args:
        predictions: Long-form positive-class predictions.

    Returns:
        Canonical prediction table with an ``analysis`` column.

    Raises:
        ValueError: If schema, values, IDs, or paired alignment are invalid.
    """
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"predictions: missing columns {missing}")
    if predictions.empty:
        raise ValueError("predictions: empty table")
    out = predictions.copy()
    if "analysis" not in out:
        out["analysis"] = "all_rows"
    meta = [*CELL_COLUMNS, "seed", "group"]
    if out[meta].isna().any().any():
        raise ValueError("predictions: missing group, seed, condition, cohort, or analysis")
    seeds = pd.to_numeric(out["seed"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(seeds).all() or (seeds != np.floor(seeds)).any():
        raise ValueError("predictions: seeds must be integers")
    out["seed"] = seeds.astype(np.int64)
    _validate_labels(out["label"], "predictions")
    out["label"] = out["label"].astype(np.int64)
    scores = pd.to_numeric(out["score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("predictions: scores must be finite probabilities")
    out["score"] = scores
    if out["prox_defined"].isna().any() or not out["prox_defined"].isin([True, False]).all():
        raise ValueError("predictions: prox_defined must contain nonmissing boolean values")
    out["prox_defined"] = out["prox_defined"].astype(bool)
    parts: list[pd.DataFrame] = []
    for _, block in out.groupby(meta, sort=True, dropna=False):
        parts.append(canonical_front_ids(block, "predictions"))
    out = pd.concat(parts, ignore_index=True)
    for _, cell in out.groupby(CELL_COLUMNS, sort=True):
        expected: pd.DataFrame | None = None
        expected_groups: set[str] | None = None
        for _, seeded in cell.groupby("seed", sort=True):
            groups = set(seeded["group"])
            if expected_groups is not None and groups != expected_groups:
                raise ValueError("predictions: missing model groups in a fitted seed")
            expected_groups = groups
            for _, block in seeded.groupby("group", sort=True):
                aligned = block.sort_values(ID_COLUMNS).reset_index(drop=True)
                signature = aligned[[*ID_COLUMNS, "label", "prox_defined"]]
                if expected is not None and not signature.equals(expected):
                    raise ValueError("predictions: missing IDs or mismatched labels/proximity across models")
                expected = signature
    return out


def node_product_weights(
    fronts: pd.DataFrame, node_values: Mapping[str, float],
) -> npt.NDArray[np.float64]:
    """Map shared concept multipliers to pair products and single-term weights.

    Args:
        fronts: Unique front IDs, with unordered pair keys.
        node_values: Positive concept multipliers, shared across comparisons.

    Returns:
        One weight per input front in its original row order.

    Raises:
        ValueError: If IDs or node weights are missing or invalid.
    """
    fronts = canonical_front_ids(fronts)
    weights: list[float] = []
    for row in fronts.itertuples(index=False):
        units = row.key.split("|") if row.kind == "pair" else [row.key]
        if any(unit not in node_values for unit in units):
            raise ValueError(f"missing concept multiplier for front {row.key!r}")
        values = np.asarray([node_values[unit] for unit in units], dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("concept multipliers must be finite and positive")
        weights.append(float(values.prod()))
    return np.asarray(weights, dtype=np.float64)


def _comparisons(
    groups: set[str], comparisons: Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if comparisons is None:
        if "S" in groups:
            selected = [("S", group) for group in sorted(groups - {"S"})]
            for pair in [("S+prox", "S+prox+rc"), ("S+prox+z", "S+prox+z+rc")]:
                if set(pair).issubset(groups):
                    selected.append(pair)
            return selected
        if groups == {"legacy_S", "legacy_mixed"}:
            return [("legacy_S", "legacy_mixed")]
        raise ValueError("prediction analysis has no recognized baseline group")
    explicit: list[tuple[str, str]] = []
    for baseline, augmented in comparisons:
        available = (baseline in groups, augmented in groups)
        if any(available) and not all(available):
            raise ValueError(f"missing comparison group: {baseline}, {augmented}")
        if all(available):
            explicit.append((baseline, augmented))
    if not explicit:
        raise ValueError("no requested comparisons apply to prediction analysis")
    return explicit


def node_multiplier_sensitivity(
    predictions: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 20261017,
    comparisons: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Measure paired AP sensitivity with globally shared exponential node weights.

    Args:
        predictions: Predictions for one or more conditions/cohorts/model seeds.
        n_resamples: Number of exponential multiplier draws.
        seed: Deterministic multiplier RNG seed.
        comparisons: Optional explicit ``(baseline, augmented)`` group pairs.

    Returns:
        Point effects, separate seed variation, percentile sensitivity intervals,
        and draw effects. AP is computed per fitted seed before averaging paired
        increments. Predictors are fixed during all multiplier draws.

    Raises:
        ValueError: If predictions, comparisons, or draw count are invalid.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    predictions = validate_predictions(predictions)
    nodes: set[str] = set()
    for kind, key in predictions[["kind", "key"]].drop_duplicates().itertuples(index=False):
        nodes.update(key.split("|") if kind == "pair" else [key])
    node_index = {node: index for index, node in enumerate(sorted(nodes))}
    prepared: list[dict[str, Any]] = []
    for identity, cell in predictions.groupby(CELL_COLUMNS, sort=True):
        first = next(iter(cell.groupby(["seed", "group"], sort=True)))[1]
        fronts = first.sort_values(ID_COLUMNS).reset_index(drop=True)
        y_true = fronts["label"].to_numpy(dtype=int)
        if np.unique(y_true).size < 2:
            raise ValueError("AP sensitivity requires both outcome classes in each analysis")
        left: list[int] = []
        right: list[int] = []
        for kind, key in fronts[["kind", "key"]].itertuples(index=False):
            units = key.split("|") if kind == "pair" else [key]
            left.append(node_index[units[0]])
            right.append(node_index[units[1]] if len(units) == 2 else -1)
        scores: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        for (model_seed, group), block in cell.groupby(["seed", "group"], sort=True):
            scores.setdefault(int(model_seed), {})[str(group)] = (
                block.sort_values(ID_COLUMNS)["score"].to_numpy(dtype=float)
            )
        ranked = {
            model_seed: {group: FixedScoreAveragePrecision(y_true, score) for group, score in values.items()}
            for model_seed, values in scores.items()
        }
        group_names = set(cell["group"])
        for baseline, augmented in _comparisons(group_names, comparisons):
            per_seed = [
                {
                    "seed": model_seed,
                    "baseline_ap": float(average_precision_score(y_true, values[baseline])),
                    "augmented_ap": float(average_precision_score(y_true, values[augmented])),
                }
                for model_seed, values in sorted(scores.items())
            ]
            deltas = [row["augmented_ap"] - row["baseline_ap"] for row in per_seed]
            prepared.append({
                "identity": dict(zip(CELL_COLUMNS, identity, strict=True)),
                "baseline": baseline, "augmented": augmented, "y_true": y_true, "scores": scores,
                "ranked": ranked,
                "left": np.asarray(left), "right": np.asarray(right), "per_seed": per_seed,
                "mean_delta_ap": float(np.mean(deltas)),
                "seed_delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                "n_fronts": len(fronts), "n_positive": int(y_true.sum()),
                "n_prox_defined": int(fronts["prox_defined"].sum()),
                "draws": np.empty(n_resamples, dtype=float),
            })
    rng = np.random.default_rng(seed)
    for draw in range(n_resamples):
        multipliers = rng.exponential(scale=1.0, size=len(nodes))
        ap_cache: dict[int, dict[int, dict[str, float]]] = {}
        for item in prepared:
            cache_key = id(item["ranked"])
            if cache_key not in ap_cache:
                right_indices = item["right"]
                weights = multipliers[item["left"]] * np.where(
                    right_indices >= 0, multipliers[right_indices], 1.0,
                )
                ap_cache[cache_key] = {
                    model_seed: {
                        group: metric.evaluate(weights)
                        for group, metric in values.items()
                    }
                    for model_seed, values in item["ranked"].items()
                }
            deltas = [
                values[item["augmented"]] - values[item["baseline"]]
                for values in ap_cache[cache_key].values()
            ]
            item["draws"][draw] = np.mean(deltas)
    results: list[dict[str, Any]] = []
    for item in prepared:
        results.append({
            **item["identity"],
            **{name: item[name] for name in (
                "baseline", "augmented", "mean_delta_ap", "seed_delta_sd", "per_seed", "n_fronts",
                "n_positive", "n_prox_defined",
            )},
            "dependence_sensitivity_interval": np.percentile(item["draws"], [2.5, 97.5]).tolist(),
            "draw_mean_paired_delta_ap": item["draws"].tolist(),
        })
    return {
        "method": "shared_concept_exponential_rate_1_pair_products",
        "interpretation": "dependence sensitivity conditional on fitted predictors and retained corpus",
        "coverage_claim": "no guaranteed coverage for nonlinear selected networks",
        "effect": "mean of paired AP increments across original fitted model seeds",
        "seed_delta_sd_ddof": 1,
        "n_resamples": n_resamples, "multiplier_seed": seed, "n_concepts": len(nodes), "results": results,
    }
