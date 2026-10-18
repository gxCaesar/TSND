"""B8 structural/topological features plus two-stage LightGBM helpers.

Source paper: arXiv:2606.03864, breakthroughs forecasting.
Source implementation: https://github.com/wazaahhh/breakthroughs-forecasting (MIT).

This module adapts a NetworkX structural/topological subset to TSND fronts.
It is not an exact reproduction of the upstream 59-feature map or its original
link-formation evaluation. The subset and two-stage helper retain their
separate scientific scope.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tsnd.models.baselines.citation_history import (
    front_paper_indices_by_key,
    load_labels,
    load_year_graph,
    split_pair_key,
)
from tsnd.models.baselines.degree_growth import build_networkx_graph
from tsnd.models.baselines.impact4cast import unit_first_seen_years

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINE_CONFIG_KEY = "structural_lgbm"
DEFAULT_PAGERANK_ALPHA = 0.85
DEFAULT_PAGERANK_TOL = 1.0e-6
DEFAULT_PAGERANK_MAX_ITER = 100
DEFAULT_COMPUTE_SHORTEST_PATH = True
DEFAULT_MISSING_PATH_LENGTH = -1.0
DEFAULT_LGBM_STAGE1_FEATURE_FRACTION = 0.8
DEFAULT_LGBM_N_ESTIMATORS = 300
DEFAULT_LGBM_LEARNING_RATE = 0.05
DEFAULT_LGBM_NUM_LEAVES = 31
DEFAULT_LGBM_SUBSAMPLE = 0.8
DEFAULT_LGBM_COLSAMPLE_BYTREE = 0.8

FEATURE_COLUMNS = [
    "kind_pair",
    "struct_n_papers",
    "struct_has_edge_t",
    "struct_edge_weight_t",
    "struct_common_neighbors",
    "struct_union_neighbors",
    "struct_jaccard",
    "struct_overlap",
    "struct_adamic_adar",
    "struct_resource_allocation",
    "struct_preferential_attachment",
    "struct_salton",
    "struct_sorensen",
    "struct_hub_promoted",
    "struct_hub_depressed",
    "struct_lhn",
    "struct_same_component",
    "struct_shortest_path_length",
    "struct_degree_min",
    "struct_degree_max",
    "struct_degree_mean",
    "struct_degree_abs_diff",
    "struct_weighted_degree_min",
    "struct_weighted_degree_max",
    "struct_weighted_degree_mean",
    "struct_weighted_degree_abs_diff",
    "struct_pagerank_min",
    "struct_pagerank_max",
    "struct_pagerank_mean",
    "struct_pagerank_abs_diff",
    "struct_clustering_min",
    "struct_clustering_max",
    "struct_clustering_mean",
    "struct_core_min",
    "struct_core_max",
    "struct_core_mean",
    "struct_triangles_min",
    "struct_triangles_max",
    "struct_triangles_mean",
    "struct_degree_prev_min",
    "struct_degree_prev_max",
    "struct_degree_prev_mean",
    "struct_degree_delta_min",
    "struct_degree_delta_max",
    "struct_degree_delta_mean",
    "struct_pagerank_prev_min",
    "struct_pagerank_prev_max",
    "struct_pagerank_prev_mean",
    "struct_pagerank_delta_min",
    "struct_pagerank_delta_max",
    "struct_pagerank_delta_mean",
    "struct_age_min",
    "struct_age_max",
    "struct_age_mean",
    "struct_seen_before_min",
    "struct_seen_before_max",
    "struct_seen_before_mean",
]


@dataclass
class PairTopologyContext:
    """Cached graph state for per-pair topology features."""

    neighbors: Mapping[str, set[str]]
    edge_weights: Mapping[tuple[str, str], float]
    component_ids: Mapping[str, int]
    adamic_adar_weights: Mapping[str, float]
    resource_allocation_weights: Mapping[str, float]
    compute_shortest_path: bool
    missing_path_length: float
    shortest_path_cache: dict[tuple[str, str], float]


def build_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build B8 structural/topological per-front features.

    Args:
        cohort_year: Birth year ``t`` for front labels.
        config: Runtime configuration. Optional ``structural_lgbm`` keys
            control PageRank, shortest-path features, and LightGBM defaults.
        artifact_dir: Artifact root containing labels and yearly graph pickles.
        data_dir: Unused by B8 feature extraction; present for the uniform
            baseline protocol.
        train_years: Unused by deterministic B8 features.
        seed: Unused by feature extraction.

    Returns:
        Per-front feature DataFrame keyed by ``key``.
    """
    del data_dir, train_years, seed
    _validate_year(cohort_year)
    art_path = Path(artifact_dir)
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)

    labels = load_labels(art_path, cohort_year)
    graph_now_raw = load_year_graph(art_path, cohort_year)
    graph_prev_raw = _load_optional_graph(art_path, cohort_year - 1)
    graph_now = build_networkx_graph(graph_now_raw)
    graph_prev = build_networkx_graph(graph_prev_raw) if graph_prev_raw is not None else None
    node_now = compute_structural_node_features(graph_now, baseline_cfg)
    node_prev = compute_structural_node_features(graph_prev, baseline_cfg) if graph_prev is not None else {}
    paper_indices_by_key = front_paper_indices_by_key(labels, graph_now_raw["paper_units"])
    topology_context = build_pair_topology_context(graph_now, baseline_cfg)
    first_seen = unit_first_seen_years(art_path, cohort_year, config)

    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        units = [key] if kind == "term" else list(split_pair_key(key))
        paper_indices = paper_indices_by_key[key]
        rows.append(
            {
                "key": key,
                "kind_pair": int(kind == "pair"),
                "struct_n_papers": len(paper_indices),
                **pair_topology_features_from_context(key, kind, graph_now, topology_context),
                **aggregate_structural_features(units, node_now, node_prev),
                **structural_age_features(units, first_seen, cohort_year),
            }
        )

    out = pd.DataFrame(rows)
    LOGGER.info("built B8 structural features for %d fronts at t=%d", len(out), cohort_year)
    return out


def compute_structural_node_features(
    graph: Any | None,
    config: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute node-level structural features for a NetworkX graph."""
    if graph is None:
        return {}
    alpha = float(config.get("pagerank_alpha", DEFAULT_PAGERANK_ALPHA))
    tol = float(config.get("pagerank_tol", DEFAULT_PAGERANK_TOL))
    max_iter = int(config.get("pagerank_max_iter", DEFAULT_PAGERANK_MAX_ITER))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"pagerank_alpha must be in (0, 1), got {alpha}")
    if tol <= 0.0:
        raise ValueError(f"pagerank_tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"pagerank_max_iter must be positive, got {max_iter}")

    import networkx as nx

    try:
        pagerank = nx.pagerank(graph, alpha=alpha, tol=tol, max_iter=max_iter, weight="weight")
    except nx.PowerIterationFailedConvergence:
        LOGGER.warning("PageRank did not converge for structural baseline; using zeros")
        pagerank = {node: 0.0 for node in graph.nodes}
    clustering = nx.clustering(graph, weight="weight")
    triangles = nx.triangles(graph)
    try:
        core = nx.core_number(graph)
    except nx.NetworkXError:
        core = {node: 0 for node in graph.nodes}

    features: dict[str, dict[str, float]] = {}
    for node in graph.nodes:
        features[str(node)] = {
            "degree": float(graph.degree(node)),
            "weighted_degree": float(graph.degree(node, weight="weight")),
            "pagerank": float(pagerank.get(node, 0.0)),
            "clustering": float(clustering.get(node, 0.0)),
            "core": float(core.get(node, 0.0)),
            "triangles": float(triangles.get(node, 0.0)),
        }
    return features


def build_pair_topology_context(graph: Any, config: Mapping[str, Any]) -> PairTopologyContext:
    """Precompute graph state reused by all pair topology rows."""
    import networkx as nx

    neighbors = {
        str(node): set(str(neighbor) for neighbor in graph.neighbors(node))
        for node in graph.nodes
    }
    edge_weights: dict[tuple[str, str], float] = {}
    for unit_u, unit_v, data in graph.edges(data=True):
        key = _ordered_pair_key(str(unit_u), str(unit_v))
        edge_weights[key] = float(data.get("weight", 1.0))

    component_ids: dict[str, int] = {}
    for component_index, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_ids[str(node)] = component_index

    adamic_adar_weights: dict[str, float] = {}
    resource_allocation_weights: dict[str, float] = {}
    for node in graph.nodes:
        node_key = str(node)
        degree = float(graph.degree(node))
        adamic_adar_weights[node_key] = 1.0 / float(np.log(degree)) if degree > 1.0 else 0.0
        resource_allocation_weights[node_key] = 1.0 / degree if degree > 0.0 else 0.0

    return PairTopologyContext(
        neighbors=neighbors,
        edge_weights=edge_weights,
        component_ids=component_ids,
        adamic_adar_weights=adamic_adar_weights,
        resource_allocation_weights=resource_allocation_weights,
        compute_shortest_path=bool(config.get("compute_shortest_path", DEFAULT_COMPUTE_SHORTEST_PATH)),
        missing_path_length=float(config.get("missing_path_length", DEFAULT_MISSING_PATH_LENGTH)),
        shortest_path_cache={},
    )


def pair_topology_features(
    key: str,
    kind: str,
    graph: Any,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Compute pairwise topological link-prediction features."""
    return pair_topology_features_from_context(key, kind, graph, build_pair_topology_context(graph, config))


def pair_topology_features_from_context(
    key: str,
    kind: str,
    graph: Any,
    context: PairTopologyContext,
) -> dict[str, float]:
    """Compute pairwise topological features from cached graph state."""
    if kind == "term":
        return _nan_pair_topology()
    if kind != "pair":
        raise ValueError(f"unsupported front kind: {kind!r}")

    unit_u, unit_v = split_pair_key(key)
    neighbors_u = context.neighbors.get(unit_u)
    neighbors_v = context.neighbors.get(unit_v)
    if neighbors_u is None or neighbors_v is None:
        return _nan_pair_topology()

    degree_u = float(len(neighbors_u))
    degree_v = float(len(neighbors_v))
    common = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v
    common_count = float(len(common))
    union_count = float(len(union))
    min_degree = min(degree_u, degree_v)
    max_degree = max(degree_u, degree_v)
    edge_key = _ordered_pair_key(unit_u, unit_v)
    weighted_edge = float(context.edge_weights.get(edge_key, 0.0))

    return {
        "struct_has_edge_t": float(edge_key in context.edge_weights),
        "struct_edge_weight_t": weighted_edge,
        "struct_common_neighbors": common_count,
        "struct_union_neighbors": union_count,
        "struct_jaccard": _safe_divide(common_count, union_count),
        "struct_overlap": _safe_divide(common_count, min_degree),
        "struct_adamic_adar": _sum_weights(common, context.adamic_adar_weights),
        "struct_resource_allocation": _sum_weights(common, context.resource_allocation_weights),
        "struct_preferential_attachment": degree_u * degree_v,
        "struct_salton": _safe_divide(common_count, np.sqrt(degree_u * degree_v)),
        "struct_sorensen": _safe_divide(2.0 * common_count, degree_u + degree_v),
        "struct_hub_promoted": _safe_divide(common_count, min_degree),
        "struct_hub_depressed": _safe_divide(common_count, max_degree),
        "struct_lhn": _safe_divide(common_count, degree_u * degree_v),
        "struct_same_component": _same_component_from_context(unit_u, unit_v, context),
        "struct_shortest_path_length": _shortest_path_length_from_context(graph, unit_u, unit_v, context),
    }


def aggregate_structural_features(
    units: Sequence[str],
    current: Mapping[str, Mapping[str, float]],
    previous: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate node structural features and temporal deltas to a front."""
    degree_now = _values(units, current, "degree")
    weighted_now = _values(units, current, "weighted_degree")
    pagerank_now = _values(units, current, "pagerank")
    clustering_now = _values(units, current, "clustering")
    core_now = _values(units, current, "core")
    triangles_now = _values(units, current, "triangles")
    degree_prev = _values(units, previous, "degree")
    pagerank_prev = _values(units, previous, "pagerank")
    degree_delta = degree_now - degree_prev
    pagerank_delta = pagerank_now - pagerank_prev

    return {
        **_min_max_mean_abs("struct_degree", degree_now),
        **_min_max_mean_abs("struct_weighted_degree", weighted_now),
        **_min_max_mean_abs("struct_pagerank", pagerank_now),
        **_min_max_mean("struct_clustering", clustering_now),
        **_min_max_mean("struct_core", core_now),
        **_min_max_mean("struct_triangles", triangles_now),
        **_min_max_mean("struct_degree_prev", degree_prev),
        **_min_max_mean("struct_degree_delta", degree_delta),
        **_min_max_mean("struct_pagerank_prev", pagerank_prev),
        **_min_max_mean("struct_pagerank_delta", pagerank_delta),
    }


def structural_age_features(
    units: Sequence[str],
    first_seen: Mapping[str, int],
    cohort_year: int,
) -> dict[str, float]:
    """Compute age and prior-seen features for structural baseline."""
    ages = np.asarray(
        [max(0, cohort_year - int(first_seen.get(unit, cohort_year))) for unit in units],
        dtype=np.float64,
    )
    seen_before = (ages > 0.0).astype(np.float64)
    return {
        "struct_age_min": _array_min(ages),
        "struct_age_max": _array_max(ages),
        "struct_age_mean": _array_mean(ages),
        "struct_seen_before_min": _array_min(seen_before),
        "struct_seen_before_max": _array_max(seen_before),
        "struct_seen_before_mean": _array_mean(seen_before),
    }


def fit_two_stage_lightgbm(
    train_frame: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, Any]:
    """Fit a two-stage LightGBM stack for structural features.

    Stage 1 fits on a deterministic prefix of the feature list; Stage 2 fits on
    all features plus the Stage-1 probability. This mirrors the two-stage
    spirit of the cited structural baseline without hard-coding reported-number
    protocol decisions.

    Args:
        train_frame: Training data containing features and a binary label.
        label_column: Name of the binary label column.
        feature_columns: Ordered feature-column names.
        config: ``structural_lgbm`` model config.
        seed: Random seed.

    Returns:
        Tuple ``(stage1_model, stage2_model)`` of fitted LightGBM classifiers.
    """
    if label_column not in train_frame.columns:
        raise ValueError(f"label column missing from training frame: {label_column}")
    missing = [column for column in feature_columns if column not in train_frame.columns]
    if missing:
        raise ValueError(f"feature columns missing from training frame: {missing}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")

    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)
    fraction = float(baseline_cfg.get("stage1_feature_fraction", DEFAULT_LGBM_STAGE1_FEATURE_FRACTION))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"stage1_feature_fraction must be in (0, 1], got {fraction}")
    stage1_count = max(1, int(np.ceil(len(feature_columns) * fraction)))
    stage1_columns = list(feature_columns[:stage1_count])
    all_columns = list(feature_columns)

    y = train_frame[label_column].to_numpy(dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("two-stage LightGBM requires both classes in training labels")
    stage1 = _new_lgbm_classifier(baseline_cfg, seed=seed)
    stage1.fit(train_frame[stage1_columns].to_numpy(dtype=np.float64), y)

    stage1_probability = stage1.predict_proba(train_frame[stage1_columns].to_numpy(dtype=np.float64))[:, 1]
    stage2_input = train_frame[all_columns].copy()
    stage2_input["struct_stage1_probability"] = stage1_probability
    stage2 = _new_lgbm_classifier(baseline_cfg, seed=seed)
    stage2.fit(stage2_input.to_numpy(dtype=np.float64), y)
    return stage1, stage2


def _new_lgbm_classifier(config: Mapping[str, Any], *, seed: int) -> Any:
    """Create a LightGBM classifier from explicit config defaults."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=int(config.get("n_estimators", DEFAULT_LGBM_N_ESTIMATORS)),
        learning_rate=float(config.get("learning_rate", DEFAULT_LGBM_LEARNING_RATE)),
        num_leaves=int(config.get("num_leaves", DEFAULT_LGBM_NUM_LEAVES)),
        subsample=float(config.get("subsample", DEFAULT_LGBM_SUBSAMPLE)),
        colsample_bytree=float(config.get("colsample_bytree", DEFAULT_LGBM_COLSAMPLE_BYTREE)),
        random_state=seed,
        verbose=-1,
    )


def _values(
    units: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    name: str,
) -> npt.NDArray[np.float64]:
    """Return node feature values for one or two units."""
    return np.asarray([features.get(unit, {}).get(name, 0.0) for unit in units], dtype=np.float64)


def _min_max_mean(prefix: str, values: npt.NDArray[np.float64]) -> dict[str, float]:
    """Return min, max, and mean aggregations."""
    return {
        f"{prefix}_min": _array_min(values),
        f"{prefix}_max": _array_max(values),
        f"{prefix}_mean": _array_mean(values),
    }


def _min_max_mean_abs(prefix: str, values: npt.NDArray[np.float64]) -> dict[str, float]:
    """Return min, max, mean, and absolute endpoint difference."""
    out = _min_max_mean(prefix, values)
    if values.size == 1:
        out[f"{prefix}_abs_diff"] = 0.0
    elif values.size == 2:
        out[f"{prefix}_abs_diff"] = float(abs(values[0] - values[1]))
    else:
        raise ValueError(f"expected one or two endpoint values, got {values.size}")
    return out


def _nan_pair_topology() -> dict[str, float]:
    """Return NaN/zero pair topology values for term fronts or missing nodes."""
    return {
        "struct_has_edge_t": 0.0,
        "struct_edge_weight_t": 0.0,
        "struct_common_neighbors": float("nan"),
        "struct_union_neighbors": float("nan"),
        "struct_jaccard": float("nan"),
        "struct_overlap": float("nan"),
        "struct_adamic_adar": float("nan"),
        "struct_resource_allocation": float("nan"),
        "struct_preferential_attachment": float("nan"),
        "struct_salton": float("nan"),
        "struct_sorensen": float("nan"),
        "struct_hub_promoted": float("nan"),
        "struct_hub_depressed": float("nan"),
        "struct_lhn": float("nan"),
        "struct_same_component": float("nan"),
        "struct_shortest_path_length": float("nan"),
    }


def _adamic_adar(graph: Any, common_neighbors: set[str]) -> float:
    """Compute Adamic-Adar score from common neighbors."""
    total = 0.0
    for node in common_neighbors:
        degree = float(graph.degree(node))
        if degree > 1.0:
            total += 1.0 / float(np.log(degree))
    return total


def _resource_allocation(graph: Any, common_neighbors: set[str]) -> float:
    """Compute resource-allocation score from common neighbors."""
    total = 0.0
    for node in common_neighbors:
        degree = float(graph.degree(node))
        if degree > 0.0:
            total += 1.0 / degree
    return total


def _ordered_pair_key(unit_u: str, unit_v: str) -> tuple[str, str]:
    """Return the canonical ordered key for an undirected pair."""
    return (unit_u, unit_v) if unit_u <= unit_v else (unit_v, unit_u)


def _sum_weights(nodes: set[str], weights: Mapping[str, float]) -> float:
    """Sum cached per-node weights over common neighbors."""
    return float(sum(weights.get(node, 0.0) for node in nodes))


def _same_component(graph: Any, unit_u: str, unit_v: str) -> float:
    """Return whether two nodes are in the same connected component."""
    import networkx as nx

    return float(nx.has_path(graph, unit_u, unit_v))


def _same_component_from_context(
    unit_u: str,
    unit_v: str,
    context: PairTopologyContext,
) -> float:
    """Return cached connected-component membership for two present nodes."""
    return float(context.component_ids.get(unit_u) == context.component_ids.get(unit_v))


def _shortest_path_length(
    graph: Any,
    unit_u: str,
    unit_v: str,
    config: Mapping[str, Any],
) -> float:
    """Compute shortest-path length with an explicit missing-path sentinel."""
    import networkx as nx

    compute_path = bool(config.get("compute_shortest_path", DEFAULT_COMPUTE_SHORTEST_PATH))
    missing_value = float(config.get("missing_path_length", DEFAULT_MISSING_PATH_LENGTH))
    if not compute_path:
        return missing_value
    try:
        return float(nx.shortest_path_length(graph, unit_u, unit_v))
    except nx.NetworkXNoPath:
        return missing_value


def _shortest_path_length_from_context(
    graph: Any,
    unit_u: str,
    unit_v: str,
    context: PairTopologyContext,
) -> float:
    """Compute shortest-path length using cached direct-edge/component checks."""
    import networkx as nx

    if not context.compute_shortest_path:
        return context.missing_path_length
    if unit_u == unit_v:
        return 0.0

    edge_key = _ordered_pair_key(unit_u, unit_v)
    if edge_key in context.edge_weights:
        return 1.0
    if not bool(_same_component_from_context(unit_u, unit_v, context)):
        return context.missing_path_length
    if edge_key not in context.shortest_path_cache:
        try:
            value = float(nx.shortest_path_length(graph, unit_u, unit_v))
        except nx.NetworkXNoPath:
            value = context.missing_path_length
        context.shortest_path_cache[edge_key] = value
    return context.shortest_path_cache[edge_key]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with NaN for zero denominators."""
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _load_optional_graph(artifact_dir: Path, cohort_year: int) -> dict[str, Any] | None:
    """Load a graph if it exists."""
    if cohort_year <= 0:
        return None
    try:
        return load_year_graph(artifact_dir, cohort_year)
    except FileNotFoundError:
        return None


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
