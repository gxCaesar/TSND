"""Registry and merge helpers for TSND forecast baseline feature families.

The forecast stage starts from the TSND feature table returned by
``tsnd.models.ours.assemble_features`` and optionally left-joins selected baseline
feature blocks by ``key``. This module centralizes the baseline registry, thin
adapters for non-uniform baselines, and the feature-group definitions consumed by
``tsnd.training.forecaster``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from tsnd.models.baselines import (
    degree_growth,
    impact4cast,
    shibayama_yin,
    specter_nn,
    structural_lgbm,
    tfidf_novelty,
    uzzi,
)
from tsnd.models.baselines.citation_history import edge_endpoint_to_unit
from tsnd.models.ours import TSND_FEATURE_COLUMNS, assemble_features

LOGGER = logging.getLogger(__name__)

_BuildFeatures = Callable[..., pd.DataFrame]
_BaselineEntry = tuple[_BuildFeatures, list[str]]
_TSND_BASE_COLUMNS = frozenset({"key", "label", *TSND_FEATURE_COLUMNS})
_TERMS_DIR_OVERRIDE = "_baseline_registry_terms_dir"
_EMB_DIR_OVERRIDE = "_baseline_registry_emb_dir"

UNIFORM_BASELINES: dict[str, _BaselineEntry] = {
    "b2": (degree_growth.build_features, list(degree_growth.FEATURE_COLUMNS)),
    "b3": (tfidf_novelty.build_features, list(tfidf_novelty.FEATURE_COLUMNS)),
    "b5": (specter_nn.build_features, list(specter_nn.FEATURE_COLUMNS)),
    "b7": (impact4cast.build_features, list(impact4cast.FEATURE_COLUMNS)),
    "b8": (structural_lgbm.build_features, list(structural_lgbm.FEATURE_COLUMNS)),
}


def build_baseline_features(
    name: str,
    t: int,
    config: Mapping[str, Any],
    artifact_dir: Path,
    data_dir: Path,
    terms_dir: Path,
    emb_dir: Path,
    train_years: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Build one baseline feature block and keep only merge-safe columns.

    Args:
        name: Baseline id such as ``"b2"`` or ``"b6"``.
        t: Cohort year used to load ``labels/labels_{t}.parquet`` and graph
            artifacts.
        config: Runtime configuration mapping.
        artifact_dir: Artifact root containing ``labels/``, ``terms/``, and
            baseline-specific artifact directories.
        data_dir: Directory containing yearly ``works_{year}.parquet`` inputs.
        terms_dir: Directory containing yearly graph pickles ``year_{t}.pkl``.
        emb_dir: Directory containing TSND embeddings ``emb_{t}_s{seed}.npz``
            with matrix shape ``(|V_t|, dim)``.
        train_years: Training years used by fit-style baselines such as B3.
        seed: Random seed for seed-dependent baselines.

    Returns:
        A DataFrame with shape ``(n_rows, 1 + n_unique_columns)`` containing
        ``key`` plus columns from the selected baseline that are not already
        supplied by the TSND base table.
    """
    baseline_name = _validate_baseline_name(name)
    builder, feature_columns = _baseline_entry(baseline_name)
    builder_config = _config_with_adapter_dirs(config, terms_dir, emb_dir)
    raw = builder(
        int(t),
        builder_config,
        artifact_dir,
        data_dir,
        train_years=list(train_years),
        seed=int(seed),
    )
    return _select_feature_payload(
        raw,
        baseline_name=baseline_name,
        feature_columns=feature_columns,
        existing_columns=_TSND_BASE_COLUMNS,
    )


def assemble_forecast_table(
    base_table: pd.DataFrame,
    names: Sequence[str],
    t: int,
    config: Mapping[str, Any],
    artifact_dir: Path,
    data_dir: Path,
    terms_dir: Path,
    emb_dir: Path,
    train_years: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Left-merge selected baseline features into one wide forecast table.

    Args:
        base_table: TSND table from ``assemble_features`` with shape
            ``(n_fronts, n_tsnd_columns)`` and required columns ``key``,
            ``label``, ``kind_pair``, ``le``, ``rc``, ``prox``, ``deg``, and
            ``n_papers``.
        names: Baseline ids to merge. ``b1`` is always represented by the TSND
            ``kind_pair``, ``deg``, and ``n_papers`` columns and is ignored when
            present here.
        t: Cohort year used to build baseline features.
        config: Runtime configuration mapping.
        artifact_dir: Artifact root containing labels, terms, and embeddings.
        data_dir: Directory containing yearly ``works_{year}.parquet`` inputs.
        terms_dir: Directory containing yearly graph pickles.
        emb_dir: Directory containing TSND embeddings.
        train_years: Training years for baselines that fit train-only state.
        seed: Random seed.

    Returns:
        Wide table with shape ``(n_fronts, n_columns)``. Row count matches
        ``base_table`` exactly, and no ``*_x`` or ``*_y`` merge suffix columns
        are allowed.
    """
    _require_columns(base_table, ["key", "label", *TSND_FEATURE_COLUMNS], "base_table")
    original_rows = len(base_table)
    wide = base_table.copy()

    for baseline_name in _normalize_names(names):
        features = build_baseline_features(
            baseline_name,
            int(t),
            config,
            artifact_dir,
            data_dir,
            terms_dir,
            emb_dir,
            train_years,
            int(seed),
        )
        merge_columns = [
            "key",
            *[column for column in features.columns if column != "key" and column not in wide],
        ]
        dropped_columns = [column for column in features.columns if column != "key" and column in wide]
        if dropped_columns:
            LOGGER.info(
                "baseline %s: skipped existing columns during merge: %s",
                baseline_name,
                dropped_columns,
            )
        if len(merge_columns) == 1:
            LOGGER.info("baseline %s: no new columns to merge", baseline_name)
            continue
        wide = wide.merge(features.loc[:, merge_columns], on="key", how="left", validate="m:1")
        _assert_row_count(wide, original_rows, baseline_name)
        _assert_no_suffix_columns(wide)

    groups = forecast_feature_groups(names)
    missing = sorted(
        {column for columns in groups.values() for column in columns if column not in wide.columns}
    )
    if missing:
        raise AssertionError(f"forecast feature columns missing from wide table: {missing}")
    _assert_no_suffix_columns(wide)
    return wide


def forecast_feature_groups(names: Sequence[str]) -> dict[str, list[str]]:
    """Return forecaster feature groups for TSND, B1, and selected baselines.

    Args:
        names: Baseline ids to expose. ``b1`` is always present and is skipped
            when provided in this sequence.

    Returns:
        Mapping ``group_name -> feature columns``. Shared columns such as
        ``kind_pair`` and ``n_papers`` intentionally point to the TSND-supplied
        columns already present in the wide table.
    """
    groups = {
        "tsnd": list(TSND_FEATURE_COLUMNS),
        "b1": ["kind_pair", "deg", "n_papers"],
    }
    for baseline_name in _normalize_names(names):
        _, feature_columns = _baseline_entry(baseline_name)
        groups[baseline_name] = list(feature_columns)
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local smoke assembly and write its shape/groups to stdout.

    Args:
        argv: Optional command-line arguments. When omitted, ``sys.argv`` is
            used by ``argparse``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Smoke-test TSND baseline feature merging.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--artifact-dir", required=True, help="Artifact root, e.g. artifacts_debug.")
    parser.add_argument("-t", "--cohort-year", type=int, required=True, help="Cohort year.")
    parser.add_argument("--seed", type=int, required=True, help="Embedding/decomposition seed.")
    parser.add_argument("--names", default="b2,b3,b4,b6,b7,b8", help="Comma-separated baseline ids.")
    parser.add_argument("--data-dir", default=None, help="Override yearly works parquet directory.")
    parser.add_argument("--terms-dir", default=None, help="Override terms artifact directory.")
    parser.add_argument("--emb-dir", default=None, help="Override TSND embedding artifact directory.")
    parser.add_argument(
        "--train-years",
        default=None,
        help="Comma-separated train years for fit-style baselines.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config_path = Path(args.config)
    config = _load_yaml_config(config_path)
    artifact_dir = Path(args.artifact_dir)
    terms_dir = Path(args.terms_dir) if args.terms_dir else artifact_dir / "terms"
    emb_dir = Path(args.emb_dir) if args.emb_dir else artifact_dir / "embed"
    data_dir = Path(args.data_dir) if args.data_dir else _resolve_data_dir(config, config_path)
    names = _parse_csv(args.names)
    train_years = (
        _parse_years(args.train_years) if args.train_years else _default_train_years(config, args.cohort_year)
    )

    base_table = _load_base_table(
        artifact_dir=artifact_dir,
        terms_dir=terms_dir,
        cohort_year=args.cohort_year,
        seed=args.seed,
    )
    wide = assemble_forecast_table(
        base_table,
        names,
        args.cohort_year,
        config,
        artifact_dir,
        data_dir,
        terms_dir,
        emb_dir,
        train_years,
        args.seed,
    )
    payload = {
        "shape": [int(wide.shape[0]), int(wide.shape[1])],
        "feature_groups": forecast_feature_groups(names),
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _build_uzzi_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    del artifact_dir, data_dir, train_years, seed
    terms_dir = _terms_dir_from_config(config)
    graph_now = _load_year_graph(terms_dir, cohort_year)
    graph_prev = _load_year_graph(terms_dir, cohort_year - 1)
    new_pairs, observed_weight, degree = _new_pair_inputs(graph_now, graph_prev, cohort_year)
    return uzzi.uzzi_features(new_pairs, observed_weight, degree)


def _build_shibayama_yin_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    del artifact_dir, data_dir, train_years
    terms_dir = _terms_dir_from_config(config)
    emb_dir = _emb_dir_from_config(config)
    graph_now = _load_year_graph(terms_dir, cohort_year)
    graph_prev = _load_year_graph(terms_dir, cohort_year - 1)
    new_pairs, _, _ = _new_pair_inputs(graph_now, graph_prev, cohort_year)
    static_emb = _load_static_embeddings(emb_dir, cohort_year, seed)
    return shibayama_yin.shibayama_yin_features(new_pairs, static_emb)


ADAPTER_BASELINES: dict[str, _BaselineEntry] = {
    "b4": (_build_uzzi_features, ["uzzi_novelty"]),
    "b6": (_build_shibayama_yin_features, ["shibayama_novelty"]),
}


def _new_pair_inputs(
    graph_now: Mapping[str, Any],
    graph_prev: Mapping[str, Any],
    cohort_year: int,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float], dict[str, float]]:
    current_weights = _edge_weight_mapping(graph_now, cohort_year)
    previous_pairs = set(_edge_key_mapping(graph_prev).keys())
    new_pairs = sorted(pair for pair in current_weights if pair not in previous_pairs)
    observed_weight = {pair: current_weights[pair] for pair in new_pairs}
    degree = _weighted_degrees(current_weights)
    return new_pairs, observed_weight, degree


def _load_year_graph(terms_dir: Path, cohort_year: int) -> dict[str, Any]:
    if cohort_year <= 0:
        raise ValueError(f"cohort_year must be positive, got {cohort_year}")
    graph_path = terms_dir / f"year_{cohort_year}.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"yearly graph pickle not found: {graph_path}")
    if graph_path.suffix != ".pkl":
        raise ValueError(f"expected a .pkl graph artifact, got {graph_path}")
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, dict):
        raise TypeError(f"year graph must be a dict, got {type(graph).__name__}")
    _require_mapping_keys(graph, ["vocab", "edges"], f"year graph {graph_path}")
    return graph


def _edge_weight_mapping(graph: Mapping[str, Any], cohort_year: int) -> dict[tuple[str, str], float]:
    edges = graph["edges"]
    if not isinstance(edges, Mapping):
        LOGGER.warning(
            "year_%d graph edges do not store per-edge weights; using 1.0 for observed weights",
            cohort_year,
        )
        return {pair: 1.0 for pair in _edge_key_mapping(graph)}

    vocab = list(graph["vocab"])
    weights: dict[tuple[str, str], float] = {}
    used_fallback = False
    for edge_key, raw_weight in edges.items():
        pair = _edge_pair_from_key(edge_key, vocab)
        if pair is None:
            continue
        weight = _coerce_edge_weight(raw_weight)
        if weight is None:
            used_fallback = True
            weight = 1.0
        weights[pair] = float(weight)
    if used_fallback:
        LOGGER.warning(
            "year_%d graph has unweighted or malformed edge values; using 1.0 where needed",
            cohort_year,
        )
    return weights


def _edge_key_mapping(graph: Mapping[str, Any]) -> dict[tuple[str, str], None]:
    edges = graph["edges"]
    vocab = list(graph["vocab"])
    raw_keys = edges.keys() if isinstance(edges, Mapping) else edges
    pairs: dict[tuple[str, str], None] = {}
    for edge_key in raw_keys:
        pair = _edge_pair_from_key(edge_key, vocab)
        if pair is not None:
            pairs[pair] = None
    return pairs


def _edge_pair_from_key(edge_key: Any, vocab: Sequence[Any]) -> tuple[str, str] | None:
    if not isinstance(edge_key, tuple) or len(edge_key) != 2:
        raise ValueError(f"edge key must be a two-tuple, got {edge_key!r}")
    unit_u = edge_endpoint_to_unit(edge_key[0], vocab)
    unit_v = edge_endpoint_to_unit(edge_key[1], vocab)
    if unit_u == unit_v:
        return None
    return (unit_u, unit_v) if unit_u < unit_v else (unit_v, unit_u)


def _coerce_edge_weight(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("weight", "count", "n", "value"):
            if key in value:
                return _coerce_edge_weight(value[key])
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def _weighted_degrees(edge_weights: Mapping[tuple[str, str], float]) -> dict[str, float]:
    degrees: dict[str, float] = {}
    for (unit_u, unit_v), weight in edge_weights.items():
        degrees[unit_u] = degrees.get(unit_u, 0.0) + float(weight)
        degrees[unit_v] = degrees.get(unit_v, 0.0) + float(weight)
    return degrees


def _load_static_embeddings(
    emb_dir: Path,
    cohort_year: int,
    seed: int,
) -> dict[str, npt.NDArray[np.float32]]:
    emb_path = emb_dir / f"emb_{cohort_year}_s{seed}.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"embedding artifact not found: {emb_path}")
    if emb_path.suffix != ".npz":
        raise ValueError(f"expected a .npz embedding artifact, got {emb_path}")
    with np.load(emb_path, allow_pickle=True) as arrays:
        if "units" not in arrays or "mat" not in arrays:
            raise ValueError(f"embedding npz must contain 'units' and 'mat': {emb_path}")
        units = arrays["units"]
        mat = arrays["mat"]
    if units.ndim != 1:
        raise ValueError(f"embedding units must have shape (|V|,), got {units.shape}")
    if mat.ndim != 2:
        raise ValueError(f"embedding matrix must have shape (|V|, dim), got {mat.shape}")
    if mat.shape[0] != units.shape[0]:
        raise ValueError(f"embedding units/matrix row mismatch: {units.shape[0]} vs {mat.shape[0]}")
    matrix = mat.astype(np.float32, copy=False)
    return {str(unit): matrix[index] for index, unit in enumerate(units)}


def _select_feature_payload(
    frame: pd.DataFrame,
    *,
    baseline_name: str,
    feature_columns: Sequence[str],
    existing_columns: frozenset[str] | set[str],
) -> pd.DataFrame:
    _require_columns(frame, ["key"], f"baseline {baseline_name}")
    if frame.columns.duplicated().any():
        duplicates = sorted(str(column) for column in frame.columns[frame.columns.duplicated()])
        raise ValueError(f"baseline {baseline_name} returned duplicate columns: {duplicates}")
    if frame["key"].duplicated().any():
        raise ValueError(f"baseline {baseline_name} returned duplicate key rows")

    payload_columns = [
        column for column in _dedupe_preserve_order(feature_columns) if column not in existing_columns
    ]
    missing = [column for column in payload_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"baseline {baseline_name} returned no registered columns: {missing}")
    dropped = [column for column in feature_columns if column in existing_columns]
    if dropped:
        LOGGER.info("baseline %s: using TSND/base columns for shared features: %s", baseline_name, dropped)
    return frame.loc[:, ["key", *payload_columns]].copy()


def _validate_baseline_name(name: str) -> str:
    normalized = str(name).strip()
    if normalized == "b1":
        raise ValueError("b1 is supplied by the TSND base table and has no registry builder")
    if normalized not in _all_baselines():
        available = sorted(_all_baselines())
        raise ValueError(f"unknown baseline {name!r}; available baselines: {available}")
    return normalized


def _normalize_names(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name or name == "b1":
            continue
        validated = _validate_baseline_name(name)
        if validated not in seen:
            out.append(validated)
            seen.add(validated)
    return out


def _baseline_entry(name: str) -> _BaselineEntry:
    if name in UNIFORM_BASELINES:
        return UNIFORM_BASELINES[name]
    if name in ADAPTER_BASELINES:
        return ADAPTER_BASELINES[name]
    raise ValueError(f"unknown baseline {name!r}")


def _all_baselines() -> dict[str, _BaselineEntry]:
    return {**UNIFORM_BASELINES, **ADAPTER_BASELINES}


def _config_with_adapter_dirs(config: Mapping[str, Any], terms_dir: Path, emb_dir: Path) -> dict[str, Any]:
    out = dict(config)
    out[_TERMS_DIR_OVERRIDE] = str(terms_dir)
    out[_EMB_DIR_OVERRIDE] = str(emb_dir)
    return out


def _terms_dir_from_config(config: Mapping[str, Any]) -> Path:
    raw_path = config.get(_TERMS_DIR_OVERRIDE)
    if raw_path is None:
        artifacts_cfg = config.get("artifacts", {})
        if isinstance(artifacts_cfg, Mapping) and artifacts_cfg.get("out_dir"):
            return Path(str(artifacts_cfg["out_dir"])) / "terms"
        return Path("artifacts") / "terms"
    return Path(str(raw_path))


def _emb_dir_from_config(config: Mapping[str, Any]) -> Path:
    raw_path = config.get(_EMB_DIR_OVERRIDE)
    if raw_path is None:
        artifacts_cfg = config.get("artifacts", {})
        if isinstance(artifacts_cfg, Mapping) and artifacts_cfg.get("out_dir"):
            return Path(str(artifacts_cfg["out_dir"])) / "embed"
        return Path("artifacts") / "embed"
    return Path(str(raw_path))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _require_mapping_keys(mapping: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _assert_row_count(frame: pd.DataFrame, expected_rows: int, baseline_name: str) -> None:
    if len(frame) != expected_rows:
        raise AssertionError(
            f"baseline {baseline_name} merge changed row count: expected {expected_rows}, got {len(frame)}"
        )


def _assert_no_suffix_columns(frame: pd.DataFrame) -> None:
    suffix_columns = sorted(column for column in frame.columns if str(column).endswith(("_x", "_y")))
    if suffix_columns:
        raise AssertionError(f"merge suffix columns present: {suffix_columns}")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return loaded


def _load_base_table(
    *,
    artifact_dir: Path,
    terms_dir: Path,
    cohort_year: int,
    seed: int,
) -> pd.DataFrame:
    labels_path = artifact_dir / "labels" / f"labels_{cohort_year}.parquet"
    decomp_path = artifact_dir / "decompose" / f"decomp_{cohort_year}_s{seed}.pkl"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels parquet not found: {labels_path}")
    if not decomp_path.exists():
        raise FileNotFoundError(f"decomposition pickle not found: {decomp_path}")
    labels = pd.read_parquet(labels_path)
    with decomp_path.open("rb") as handle:
        decomposition = pickle.load(handle)
    if not isinstance(decomposition, dict):
        raise TypeError(f"decomposition must be a dict, got {type(decomposition).__name__}")
    return assemble_features(labels, decomposition, terms_dir, cohort_year)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_years(value: str) -> list[int]:
    return [int(item) for item in _parse_csv(value)]


def _resolve_data_dir(config: Mapping[str, Any], config_path: Path) -> Path:
    root = config_path.resolve().parent.parent
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, Mapping):
        return root / str(data_cfg.get("out_dir", "data"))
    return root / "data"


def _default_train_years(config: Mapping[str, Any], cohort_year: int) -> list[int]:
    raw_train_years = config.get("train_years")
    if isinstance(raw_train_years, Sequence) and not isinstance(raw_train_years, (str, bytes)):
        return sorted({int(year) for year in raw_train_years})

    data_cfg = config.get("data", {})
    if isinstance(data_cfg, Mapping) and data_cfg.get("year_start") is not None:
        year_start = int(data_cfg["year_start"])
        return list(range(year_start, int(cohort_year)))
    return []


if __name__ == "__main__":
    raise SystemExit(main())
