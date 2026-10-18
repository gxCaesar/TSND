"""B1 citation-history and degree controls for TSND fronts.

Source paper: none; this is a trivial-strong structural control baseline.
This control is implemented directly for the TSND front-uptake task.

NOTE: OpenAlex ``cited_by_count`` is an all-time count at pull time and leaks
future citations into year-t features. This baseline therefore uses only
year-t graph degree plus the number of introducing papers, matching the pilot
fix in ``pilot/src/features.py``. A documented opt-in hook is provided for
OpenAlex ``counts_by_year`` citation histories truncated to ``year <= t``.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = ["kind_pair", "deg", "n_papers"]
OPTIONAL_CITATION_COLUMNS = ["mean_cites_year_t", "sum_cites_year_t"]

LABEL_COLUMNS = ("key", "kind")
GRAPH_KEYS = ("vocab", "paper_units", "edges", "n_papers")
DEFAULT_BASELINE_CONFIG_KEY = "citation_history"
DEFAULT_LABELS_DIR = "labels"
DEFAULT_TERMS_DIR = "terms"
DEFAULT_WORKS_PATTERN = "works_{year}.parquet"


def build_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build B1 per-front features for a cohort year.

    Args:
        cohort_year: Birth year ``t`` for the front labels.
        config: Baseline/runtime configuration. Optional keys under
            ``citation_history`` are ``include_truncated_citations`` and
            ``works_pattern``.
        artifact_dir: Directory containing pilot-compatible ``labels/`` and
            ``terms/`` artifacts.
        data_dir: Directory containing ``works_{year}.parquet`` files.
        train_years: Unused by B1; present for the uniform baseline protocol.
        seed: Unused by B1; present for the uniform baseline protocol.

    Returns:
        DataFrame with columns ``key``, ``kind_pair``, ``deg``, ``n_papers``,
        and optional truncated-citation columns when explicitly configured.
    """
    del train_years, seed
    _validate_year(cohort_year)
    art_path = _as_path(artifact_dir, "artifact_dir")
    data_path = _as_path(data_dir, "data_dir")
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)

    labels = load_labels(art_path, cohort_year)
    graph = load_year_graph(art_path, cohort_year)
    degrees = graph_degrees(graph)
    inverted = invert_paper_units(graph["paper_units"])

    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        papers = front_paper_indices(key, kind, inverted)
        rows.append(
            {
                "key": key,
                "kind_pair": int(kind == "pair"),
                "deg": front_degree(key, kind, degrees),
                "n_papers": len(papers),
                "_paper_indices": papers,
            }
        )

    out = pd.DataFrame(rows)
    include_truncated = bool(baseline_cfg.get("include_truncated_citations", False))
    if include_truncated:
        out = add_truncated_citation_hook(
            out,
            cohort_year=cohort_year,
            config=baseline_cfg,
            data_dir=data_path,
        )
    out = out.drop(columns=["_paper_indices"])
    LOGGER.info("built B1 citation-history features for %d fronts at t=%d", len(out), cohort_year)
    return out


def add_truncated_citation_hook(
    features: pd.DataFrame,
    *,
    cohort_year: int,
    config: Mapping[str, Any],
    data_dir: Path,
) -> pd.DataFrame:
    """Add leakage-safe citation counts from OpenAlex ``counts_by_year``.

    This hook is intentionally opt-in because pilot artifacts did not require
    yearly citation histories. It expects the cohort works parquet to contain a
    ``counts_by_year`` column whose rows are lists of dictionaries with
    ``year`` and ``cited_by_count`` fields.

    Args:
        features: Per-front feature frame with an internal ``_paper_indices``
            column containing paper-row indices.
        cohort_year: Year ``t``. Only citation counts with ``year <= t`` are
            used.
        config: Hook config. ``works_pattern`` may override the file naming
            pattern and ``counts_by_year_column`` may override the column name.
        data_dir: Directory containing yearly works parquet files.

    Returns:
        Copy of ``features`` with ``mean_cites_year_t`` and
        ``sum_cites_year_t`` columns.
    """
    _validate_year(cohort_year)
    if "_paper_indices" not in features.columns:
        raise ValueError("features must include internal '_paper_indices' before citation hook")

    works_pattern = str(config.get("works_pattern", DEFAULT_WORKS_PATTERN))
    counts_column = str(config.get("counts_by_year_column", "counts_by_year"))
    works_path = data_dir / works_pattern.format(year=cohort_year)
    if not works_path.exists():
        raise FileNotFoundError(f"works parquet not found for citation hook: {works_path}")

    works = pd.read_parquet(works_path, columns=[counts_column])
    counts = np.asarray(
        [
            truncated_citation_count(value, cohort_year=cohort_year)
            for value in works[counts_column].tolist()
        ],
        dtype=np.float64,
    )

    out = features.copy()
    mean_values: list[float] = []
    sum_values: list[float] = []
    for paper_indices in out["_paper_indices"].tolist():
        indices = list(paper_indices)
        if not indices:
            mean_values.append(float("nan"))
            sum_values.append(float("nan"))
            continue
        selected = counts[np.asarray(indices, dtype=np.int64)]
        mean_values.append(float(np.mean(selected)))
        sum_values.append(float(np.sum(selected)))
    out["mean_cites_year_t"] = mean_values
    out["sum_cites_year_t"] = sum_values
    return out


def truncated_citation_count(counts_by_year: Any, *, cohort_year: int) -> float:
    """Return citations observed no later than ``cohort_year``.

    Args:
        counts_by_year: OpenAlex-style yearly citation history, usually a list
            of dictionaries with ``year`` and ``cited_by_count`` keys.
        cohort_year: Year ``t`` used as the causal cutoff.

    Returns:
        Sum of citation counts for records whose year is at most ``t``. Missing
        or malformed records contribute zero.
    """
    _validate_year(cohort_year)
    if not isinstance(counts_by_year, list):
        return 0.0

    total = 0.0
    for item in counts_by_year:
        if not isinstance(item, Mapping):
            continue
        year = item.get("year")
        count = item.get("cited_by_count")
        if not isinstance(year, (int, np.integer)) or int(year) > cohort_year:
            continue
        if isinstance(count, (int, float, np.integer, np.floating)) and np.isfinite(count):
            total += float(count)
    return total


def load_labels(artifact_dir: Path, cohort_year: int) -> pd.DataFrame:
    """Load front labels for one cohort year.

    Args:
        artifact_dir: Artifact root containing ``labels/labels_{t}.parquet``.
        cohort_year: Cohort year ``t``.

    Returns:
        DataFrame containing at least ``key`` and ``kind`` columns.
    """
    _validate_year(cohort_year)
    labels_path = artifact_dir / DEFAULT_LABELS_DIR / f"labels_{cohort_year}.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels parquet not found: {labels_path}")
    labels = pd.read_parquet(labels_path)
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"labels file {labels_path} is missing columns: {missing}")
    return labels


def load_year_graph(artifact_dir: Path, cohort_year: int) -> dict[str, Any]:
    """Load the pilot-compatible yearly graph pickle.

    Args:
        artifact_dir: Artifact root containing ``terms/year_{t}.pkl``.
        cohort_year: Cohort year ``t``.

    Returns:
        Dictionary with ``vocab``, ``paper_units``, ``edges``, and ``n_papers``.
    """
    _validate_year(cohort_year)
    graph_path = artifact_dir / DEFAULT_TERMS_DIR / f"year_{cohort_year}.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"yearly graph pickle not found: {graph_path}")
    _require_trusted_artifact_path(graph_path, ".pkl")
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, dict):
        raise TypeError(f"yearly graph must be a dict, got {type(graph).__name__}")
    missing = [key for key in GRAPH_KEYS if key not in graph]
    if missing:
        raise ValueError(f"yearly graph {graph_path} is missing keys: {missing}")
    return graph


def invert_paper_units(paper_units: Sequence[Sequence[str]]) -> dict[str, list[int]]:
    """Build an inverted index from unit string to paper-row indices.

    Args:
        paper_units: Sequence with length ``n_papers``; each element lists the
            units present in that paper.

    Returns:
        Mapping from unit string to sorted paper-row indices.
    """
    inverted: dict[str, list[int]] = {}
    for paper_index, units in enumerate(paper_units):
        for unit in units:
            inverted.setdefault(str(unit), []).append(paper_index)
    return inverted


def graph_degrees(graph: Mapping[str, Any]) -> dict[str, int]:
    """Compute unweighted unit degrees from the yearly co-occurrence graph.

    Args:
        graph: Pilot-compatible graph dictionary. ``edges`` may be keyed either
            by integer vocabulary indices or by unit strings.

    Returns:
        Mapping from unit string to unweighted degree in the year-t graph.
    """
    vocab = list(graph["vocab"])
    degrees: dict[str, int] = {str(unit): 0 for unit in vocab}
    for raw_u, raw_v in _edge_keys(graph["edges"]):
        unit_u = edge_endpoint_to_unit(raw_u, vocab)
        unit_v = edge_endpoint_to_unit(raw_v, vocab)
        if unit_u == unit_v:
            continue
        degrees[unit_u] = degrees.get(unit_u, 0) + 1
        degrees[unit_v] = degrees.get(unit_v, 0) + 1
    return degrees


def front_paper_indices(
    key: str,
    kind: str,
    inverted: Mapping[str, Sequence[int]],
) -> list[int]:
    """Return introducing paper indices for a front.

    Args:
        key: Front key. Pair fronts use ``"unit_a|unit_b"``.
        kind: Either ``"term"`` or ``"pair"``.
        inverted: Unit-to-paper inverted index.

    Returns:
        Paper-row indices introducing the term or pair front.
    """
    if kind == "term":
        return list(inverted.get(key, []))
    if kind == "pair":
        unit_u, unit_v = split_pair_key(key)
        return sorted(set(inverted.get(unit_u, [])) & set(inverted.get(unit_v, [])))
    raise ValueError(f"unsupported front kind: {kind!r}")


def front_paper_indices_by_key(
    labels: pd.DataFrame,
    paper_units: Sequence[Sequence[str]],
) -> dict[str, list[int]]:
    """Precompute introducing paper indices for all front labels.

    This is equivalent to calling :func:`front_paper_indices` for each row, but
    scans ``paper_units`` once instead of rebuilding endpoint posting sets for
    every pair front.

    Args:
        labels: Front labels with ``key`` and ``kind`` columns.
        paper_units: Sequence with length ``n_papers``; each element lists the
            units present in that paper.

    Returns:
        Mapping from front key to sorted paper-row indices.
    """
    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")

    # Build the posting index once, then materialise one set per unit so each pair
    # front is a single C-level set intersection. The previous implementation
    # scanned ``combinations(unique_units, 2)`` per paper, which is
    # O(sum of units_per_paper^2) and dominated build time on dense papers. This
    # form is O(num_fronts * min posting length) and returns identical lists:
    # a term keeps its raw paper-order postings (duplicates preserved) and a pair
    # keeps the ascending deduplicated overlap, matching ``front_paper_indices``.
    inverted = invert_paper_units(paper_units)
    inverted_sets: dict[str, set[int]] = {unit: set(rows) for unit, rows in inverted.items()}
    empty: set[int] = set()

    indices: dict[str, list[int]] = {}
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        if key in indices:
            continue
        if kind == "term":
            indices[key] = list(inverted.get(key, []))
        elif kind == "pair":
            unit_u, unit_v = split_pair_key(key)
            indices[key] = sorted(inverted_sets.get(unit_u, empty) & inverted_sets.get(unit_v, empty))
        else:
            raise ValueError(f"unsupported front kind: {kind!r}")
    return indices


def front_degree(key: str, kind: str, degrees: Mapping[str, int]) -> int:
    """Compute the pilot B1 degree control for a front.

    Args:
        key: Front key.
        kind: Either ``"term"`` or ``"pair"``.
        degrees: Unit-level degree mapping.

    Returns:
        Term degree for term fronts; minimum endpoint degree for pair fronts.
    """
    if kind == "term":
        return int(degrees.get(key, 0))
    if kind == "pair":
        unit_u, unit_v = split_pair_key(key)
        return int(min(degrees.get(unit_u, 0), degrees.get(unit_v, 0)))
    raise ValueError(f"unsupported front kind: {kind!r}")


def split_pair_key(key: str) -> tuple[str, str]:
    """Split a pair-front key into ordered endpoint unit strings.

    Args:
        key: Pair key formatted as ``"unit_a|unit_b"``.

    Returns:
        Two endpoint unit strings.
    """
    parts = key.split("|", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"pair key must be formatted as 'unit_a|unit_b', got {key!r}")
    unit_u, unit_v = parts
    return (unit_u, unit_v) if unit_u <= unit_v else (unit_v, unit_u)


def edge_endpoint_to_unit(endpoint: Any, vocab: Sequence[Any]) -> str:
    """Convert an edge endpoint from index-or-string form to a unit string.

    Args:
        endpoint: Integer vocabulary index or unit string.
        vocab: Vocabulary sequence from the yearly graph.

    Returns:
        Unit string.
    """
    if isinstance(endpoint, (int, np.integer)):
        index = int(endpoint)
        if index < 0 or index >= len(vocab):
            raise ValueError(f"edge endpoint index {index} outside vocab size {len(vocab)}")
        return str(vocab[index])
    return str(endpoint)


def _edge_keys(edges: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    """Return normalized two-endpoint edge keys from a graph edge mapping."""
    endpoints: list[tuple[Any, Any]] = []
    for edge_key in edges.keys():
        if not isinstance(edge_key, tuple) or len(edge_key) != 2:
            raise ValueError(f"edge key must be a two-tuple, got {edge_key!r}")
        endpoints.append((edge_key[0], edge_key[1]))
    return endpoints


def _nested_config(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a nested config mapping or an empty mapping."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config[{key!r}] must be a mapping when provided")
    return value


def _as_path(value: str | Path, name: str) -> Path:
    """Validate and normalize a filesystem path argument."""
    path = Path(value)
    if not str(path):
        raise ValueError(f"{name} must not be empty")
    return path


def _validate_year(year: int) -> None:
    """Validate a positive integer year."""
    if not isinstance(year, int):
        raise TypeError(f"year must be int, got {type(year).__name__}")
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")


def _require_trusted_artifact_path(path: Path, suffix: str) -> None:
    """Validate a local TSND artifact path before pickle-capable loading.

    Args:
        path: Artifact path.
        suffix: Required file suffix.

    Raises:
        ValueError: If the path suffix is not the expected artifact type.
    """
    if path.suffix != suffix:
        raise ValueError(f"expected a {suffix} TSND artifact, got {path}")
