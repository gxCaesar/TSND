"""B2 degree, PageRank, and graph-growth controls for TSND fronts.

Source paper: none; this is a trivial-strong structural control baseline.
This control is implemented directly for the TSND front-uptake task.
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
    edge_endpoint_to_unit,
    front_paper_indices_by_key,
    load_labels,
    load_year_graph,
    split_pair_key,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINE_CONFIG_KEY = "degree_growth"
DEFAULT_PAGERANK_ALPHA = 0.85
DEFAULT_PAGERANK_TOL = 1.0e-6
DEFAULT_PAGERANK_MAX_ITER = 100
DEFAULT_GROWTH_EPSILON = 1.0

FEATURE_COLUMNS = [
    "kind_pair",
    "n_papers",
    "degree_min",
    "degree_max",
    "degree_mean",
    "weighted_degree_min",
    "weighted_degree_max",
    "weighted_degree_mean",
    "pagerank_min",
    "pagerank_max",
    "pagerank_mean",
    "degree_growth_min",
    "degree_growth_max",
    "degree_growth_mean",
    "degree_growth_ratio_mean",
    "edge_weight_t",
    "has_edge_t",
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
    """Build B2 graph degree, PageRank, and growth features.

    Args:
        cohort_year: Birth year ``t`` for front labels.
        config: Runtime configuration. Optional ``degree_growth`` keys control
            PageRank and growth smoothing.
        artifact_dir: Directory containing ``labels/`` and ``terms/`` artifacts.
        data_dir: Unused by B2; present for the uniform baseline protocol.
        train_years: Unused by B2; present for the uniform baseline protocol.
        seed: Unused by deterministic B2 features.

    Returns:
        Per-front feature DataFrame keyed by ``key``.
    """
    del data_dir, train_years, seed
    _validate_year(cohort_year)
    art_path = Path(artifact_dir)
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)

    labels = load_labels(art_path, cohort_year)
    graph_now = load_year_graph(art_path, cohort_year)
    graph_prev = _load_optional_graph(art_path, cohort_year - 1)

    graph_features_now = compute_graph_node_features(graph_now, baseline_cfg)
    graph_features_prev = (
        compute_graph_node_features(graph_prev, baseline_cfg) if graph_prev is not None else {}
    )
    paper_indices_by_key = front_paper_indices_by_key(labels, graph_now["paper_units"])
    edge_weights = graph_edge_weights(graph_now)

    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        units = [key] if kind == "term" else list(split_pair_key(key))
        papers = paper_indices_by_key[key]
        rows.append(
            {
                "key": key,
                "kind_pair": int(kind == "pair"),
                "n_papers": len(papers),
                **aggregate_node_features(units, graph_features_now, graph_features_prev, baseline_cfg),
                **edge_features(key, kind, edge_weights),
            }
        )

    out = pd.DataFrame(rows)
    LOGGER.info("built B2 degree-growth features for %d fronts at t=%d", len(out), cohort_year)
    return out


def compute_graph_node_features(
    graph: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute node-level structural features for one yearly graph.

    Args:
        graph: Pilot-compatible graph dictionary.
        config: PageRank configuration.

    Returns:
        Mapping ``unit -> feature-name -> value``.
    """
    nx_graph = build_networkx_graph(graph)
    alpha = float(config.get("pagerank_alpha", DEFAULT_PAGERANK_ALPHA))
    tol = float(config.get("pagerank_tol", DEFAULT_PAGERANK_TOL))
    max_iter = int(config.get("pagerank_max_iter", DEFAULT_PAGERANK_MAX_ITER))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"pagerank_alpha must be in (0, 1), got {alpha}")
    if tol <= 0.0:
        raise ValueError(f"pagerank_tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"pagerank_max_iter must be positive, got {max_iter}")

    try:
        pagerank = nx_graph_pagerank(nx_graph, alpha=alpha, tol=tol, max_iter=max_iter)
    except RuntimeError as exc:
        LOGGER.warning("PageRank failed; using zeros: %s", exc)
        pagerank = {node: 0.0 for node in nx_graph.nodes}

    features: dict[str, dict[str, float]] = {}
    for node in nx_graph.nodes:
        degree = float(nx_graph.degree(node))
        weighted_degree = float(nx_graph.degree(node, weight="weight"))
        features[str(node)] = {
            "degree": degree,
            "weighted_degree": weighted_degree,
            "pagerank": float(pagerank.get(node, 0.0)),
        }
    return features


def build_networkx_graph(graph: Mapping[str, Any]) -> Any:
    """Create a NetworkX graph from a pilot-compatible yearly graph dict.

    Args:
        graph: Graph dict with ``vocab`` and weighted ``edges``.

    Returns:
        ``networkx.Graph`` with unit-string nodes and ``weight`` edge attrs.
    """
    import networkx as nx

    vocab = list(graph["vocab"])
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(str(unit) for unit in vocab)
    for edge_key, raw_weight in graph["edges"].items():
        if not isinstance(edge_key, tuple) or len(edge_key) != 2:
            raise ValueError(f"edge key must be a two-tuple, got {edge_key!r}")
        unit_u = edge_endpoint_to_unit(edge_key[0], vocab)
        unit_v = edge_endpoint_to_unit(edge_key[1], vocab)
        if unit_u == unit_v:
            continue
        weight = _safe_float(raw_weight, default=1.0)
        nx_graph.add_edge(unit_u, unit_v, weight=weight)
    return nx_graph


def nx_graph_pagerank(
    graph: Any,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> dict[str, float]:
    """Run NetworkX PageRank with explicit failure wrapping."""
    import networkx as nx

    try:
        values = nx.pagerank(graph, alpha=alpha, tol=tol, max_iter=max_iter, weight="weight")
    except nx.PowerIterationFailedConvergence as exc:
        raise RuntimeError("NetworkX PageRank did not converge") from exc
    return {str(node): float(value) for node, value in values.items()}


def aggregate_node_features(
    units: Sequence[str],
    current: Mapping[str, Mapping[str, float]],
    previous: Mapping[str, Mapping[str, float]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Aggregate node-level degree/PageRank features to one front row.

    Args:
        units: One unit for term fronts or two units for pair fronts.
        current: Current-year node features.
        previous: Previous-year node features.
        config: Growth-smoothing config.

    Returns:
        Front-level min, max, and mean features.
    """
    epsilon = float(config.get("growth_epsilon", DEFAULT_GROWTH_EPSILON))
    if epsilon <= 0.0:
        raise ValueError(f"growth_epsilon must be positive, got {epsilon}")

    degree_now = np.asarray([current.get(unit, {}).get("degree", 0.0) for unit in units], dtype=float)
    weighted_now = np.asarray(
        [current.get(unit, {}).get("weighted_degree", 0.0) for unit in units],
        dtype=float,
    )
    pagerank_now = np.asarray([current.get(unit, {}).get("pagerank", 0.0) for unit in units], dtype=float)
    degree_prev = np.asarray([previous.get(unit, {}).get("degree", 0.0) for unit in units], dtype=float)
    growth = degree_now - degree_prev
    growth_ratio = (degree_now + epsilon) / (degree_prev + epsilon)

    return {
        "degree_min": _array_min(degree_now),
        "degree_max": _array_max(degree_now),
        "degree_mean": _array_mean(degree_now),
        "weighted_degree_min": _array_min(weighted_now),
        "weighted_degree_max": _array_max(weighted_now),
        "weighted_degree_mean": _array_mean(weighted_now),
        "pagerank_min": _array_min(pagerank_now),
        "pagerank_max": _array_max(pagerank_now),
        "pagerank_mean": _array_mean(pagerank_now),
        "degree_growth_min": _array_min(growth),
        "degree_growth_max": _array_max(growth),
        "degree_growth_mean": _array_mean(growth),
        "degree_growth_ratio_mean": _array_mean(growth_ratio),
    }


def graph_edge_weights(graph: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """Return unit-string edge weights for a yearly graph."""
    vocab = list(graph["vocab"])
    weights: dict[tuple[str, str], float] = {}
    for edge_key, raw_weight in graph["edges"].items():
        if not isinstance(edge_key, tuple) or len(edge_key) != 2:
            raise ValueError(f"edge key must be a two-tuple, got {edge_key!r}")
        unit_u = edge_endpoint_to_unit(edge_key[0], vocab)
        unit_v = edge_endpoint_to_unit(edge_key[1], vocab)
        if unit_u == unit_v:
            continue
        key = (unit_u, unit_v) if unit_u <= unit_v else (unit_v, unit_u)
        weights[key] = _safe_float(raw_weight, default=1.0)
    return weights


def edge_features(
    key: str,
    kind: str,
    edge_weights: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    """Return current-year edge features for a term or pair front."""
    if kind == "term":
        return {"edge_weight_t": 0.0, "has_edge_t": 0.0}
    if kind != "pair":
        raise ValueError(f"unsupported front kind: {kind!r}")
    unit_u, unit_v = split_pair_key(key)
    edge_key = (unit_u, unit_v) if unit_u <= unit_v else (unit_v, unit_u)
    weight = float(edge_weights.get(edge_key, 0.0))
    return {"edge_weight_t": weight, "has_edge_t": float(weight > 0.0)}


def _load_optional_graph(artifact_dir: Path, cohort_year: int) -> dict[str, Any] | None:
    """Load a graph if the cohort year is valid and the file exists."""
    if cohort_year <= 0:
        return None
    try:
        return load_year_graph(artifact_dir, cohort_year)
    except FileNotFoundError:
        LOGGER.info("previous-year graph missing for t=%d", cohort_year)
        return None


def _nested_config(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a nested config mapping or an empty mapping."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config[{key!r}] must be a mapping when provided")
    return value


def _safe_float(value: Any, *, default: float) -> float:
    """Convert numeric edge weights to finite floats."""
    if isinstance(value, Mapping):
        raw_value = value.get("weight", default)
    else:
        raw_value = value
    if isinstance(raw_value, (int, float, np.integer, np.floating)) and np.isfinite(raw_value):
        return float(raw_value)
    return default


def _array_min(values: npt.NDArray[np.float64]) -> float:
    """Return a finite minimum for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {values.shape}")
    return float(np.min(values)) if values.size else float("nan")


def _array_max(values: npt.NDArray[np.float64]) -> float:
    """Return a finite maximum for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {values.shape}")
    return float(np.max(values)) if values.size else float("nan")


def _array_mean(values: npt.NDArray[np.float64]) -> float:
    """Return a finite mean for a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {values.shape}")
    return float(np.mean(values)) if values.size else float("nan")


def _validate_year(year: int) -> None:
    """Validate a positive integer year."""
    if not isinstance(year, int):
        raise TypeError(f"year must be int, got {type(year).__name__}")
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")
