"""B7 Impact4Cast-style feature family for TSND front uptake.

Source paper: arXiv:2402.08640, Impact4Cast.
Source implementation: https://github.com/artificial-scientist-lab/Impact4Cast (MIT).

This module adapts a subset of the 141-feature family to the TSND front-uptake
label. The subset includes graph degree/PageRank controls, temporal growth,
front age, introducing-paper counts, and optional year-truncated citation
summaries. Author, venue, reference, and full concept-graph feature families
are absent from this subset. It is not a reproduction of the full upstream
model or its original citation-impact task.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tsnd.models.baselines.citation_history import (
    add_truncated_citation_hook,
    front_paper_indices_by_key,
    load_labels,
    load_year_graph,
    split_pair_key,
)
from tsnd.models.baselines.degree_growth import (
    aggregate_node_features,
    edge_features,
    graph_edge_weights,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINE_CONFIG_KEY = "impact4cast"
DEFAULT_INCLUDE_TRUNCATED_CITATIONS = False
DEFAULT_DATA_CONFIG_KEY = "data"

FEATURE_COLUMNS = [
    "kind_pair",
    "i4c_n_papers",
    "i4c_degree_min",
    "i4c_degree_max",
    "i4c_degree_mean",
    "i4c_weighted_degree_min",
    "i4c_weighted_degree_max",
    "i4c_weighted_degree_mean",
    "i4c_pagerank_min",
    "i4c_pagerank_max",
    "i4c_pagerank_mean",
    "i4c_degree_growth_min",
    "i4c_degree_growth_max",
    "i4c_degree_growth_mean",
    "i4c_degree_growth_ratio_mean",
    "i4c_edge_weight_t",
    "i4c_has_edge_t",
    "i4c_age_min",
    "i4c_age_max",
    "i4c_age_mean",
    "i4c_seen_before_min",
    "i4c_seen_before_max",
    "i4c_seen_before_mean",
]


def build_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build an Impact4Cast-style TSND feature subset.

    Args:
        cohort_year: Birth year ``t`` for front labels.
        config: Runtime configuration. Optional ``impact4cast`` keys include
            graph/PageRank settings and ``include_truncated_citations``.
        artifact_dir: Artifact root containing labels and yearly graph pickles.
        data_dir: Directory containing yearly works parquet files.
        train_years: Unused by the feature builder; present for the uniform
            baseline protocol.
        seed: Unused by deterministic B7 features.

    Returns:
        Per-front feature DataFrame keyed by ``key``.
    """
    del train_years, seed
    _validate_year(cohort_year)
    art_path = Path(artifact_dir)
    data_path = Path(data_dir)
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)

    labels = load_labels(art_path, cohort_year)
    graph_now = load_year_graph(art_path, cohort_year)
    graph_prev = _load_optional_graph(art_path, cohort_year - 1)
    current_node_features = _node_features(graph_now, baseline_cfg)
    previous_node_features = _node_features(graph_prev, baseline_cfg) if graph_prev is not None else {}
    paper_indices_by_key = front_paper_indices_by_key(labels, graph_now["paper_units"])
    edge_weights = graph_edge_weights(graph_now)
    first_seen = unit_first_seen_years(art_path, cohort_year, config)

    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        units = [key] if kind == "term" else list(split_pair_key(key))
        paper_indices = paper_indices_by_key[key]
        base_features = aggregate_node_features(
            units,
            current_node_features,
            previous_node_features,
            baseline_cfg,
        )
        age_features = aggregate_age_features(units, first_seen, cohort_year)
        edge_values = edge_features(key, kind, edge_weights)
        rows.append(
            {
                "key": key,
                "kind_pair": int(kind == "pair"),
                "i4c_n_papers": len(paper_indices),
                "_paper_indices": paper_indices,
                **_prefix_features(base_features, "i4c_"),
                **_prefix_features(edge_values, "i4c_"),
                **age_features,
            }
        )

    out = pd.DataFrame(rows)
    if bool(baseline_cfg.get("include_truncated_citations", DEFAULT_INCLUDE_TRUNCATED_CITATIONS)):
        out = add_truncated_citation_hook(
            out,
            cohort_year=cohort_year,
            config=baseline_cfg,
            data_dir=data_path,
        )
        out = out.rename(
            columns={
                "mean_cites_year_t": "i4c_mean_cites_year_t",
                "sum_cites_year_t": "i4c_sum_cites_year_t",
            }
        )
    out = out.drop(columns=["_paper_indices"])
    LOGGER.info("built B7 Impact4Cast-style subset for %d fronts at t=%d", len(out), cohort_year)
    return out


def unit_first_seen_years(
    artifact_dir: Path,
    cohort_year: int,
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Scan yearly graph vocabularies to find the first observed year per unit."""
    data_cfg = config.get(DEFAULT_DATA_CONFIG_KEY, {})
    if isinstance(data_cfg, Mapping) and "year_start" in data_cfg:
        year_start = int(data_cfg["year_start"])
    else:
        year_start = cohort_year
    _validate_year(year_start)

    first_seen: dict[str, int] = {}
    for year in range(year_start, cohort_year + 1):
        try:
            graph = load_year_graph(artifact_dir, year)
        except FileNotFoundError:
            LOGGER.warning("graph missing while computing first-seen years: %d", year)
            continue
        for unit in graph["vocab"]:
            first_seen.setdefault(str(unit), year)
    return first_seen


def aggregate_age_features(
    units: Sequence[str],
    first_seen: Mapping[str, int],
    cohort_year: int,
) -> dict[str, float]:
    """Aggregate first-seen ages for a term or pair front."""
    _validate_year(cohort_year)
    ages = np.asarray(
        [max(0, cohort_year - int(first_seen.get(unit, cohort_year))) for unit in units],
        dtype=np.float64,
    )
    seen_before = (ages > 0.0).astype(np.float64)
    return {
        "i4c_age_min": _array_min(ages),
        "i4c_age_max": _array_max(ages),
        "i4c_age_mean": _array_mean(ages),
        "i4c_seen_before_min": _array_min(seen_before),
        "i4c_seen_before_max": _array_max(seen_before),
        "i4c_seen_before_mean": _array_mean(seen_before),
    }


def _node_features(graph: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Compute the B7 graph-node subset using the B2 implementation."""
    from tsnd.models.baselines.degree_growth import compute_graph_node_features

    return compute_graph_node_features(graph, config)


def _load_optional_graph(artifact_dir: Path, cohort_year: int) -> dict[str, Any] | None:
    """Load a graph if it exists."""
    if cohort_year <= 0:
        return None
    try:
        return load_year_graph(artifact_dir, cohort_year)
    except FileNotFoundError:
        return None


def _prefix_features(features: Mapping[str, float], prefix: str) -> dict[str, float]:
    """Prefix feature names without changing values."""
    return {f"{prefix}{key}": float(value) for key, value in features.items()}


def _array_min(values: npt.NDArray[np.float64]) -> float:
    """Return minimum for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got {values.shape}")
    return float(np.min(values)) if values.size else float("nan")


def _array_max(values: npt.NDArray[np.float64]) -> float:
    """Return maximum for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got {values.shape}")
    return float(np.max(values)) if values.size else float("nan")


def _array_mean(values: npt.NDArray[np.float64]) -> float:
    """Return mean for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got {values.shape}")
    return float(np.mean(values)) if values.size else float("nan")


def _nested_config(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a nested config mapping or an empty mapping."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config[{key!r}] must be a mapping when provided")
    return value


def _validate_year(year: int) -> None:
    """Validate a positive integer year."""
    if not isinstance(year, int):
        raise TypeError(f"year must be int, got {type(year).__name__}")
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")
