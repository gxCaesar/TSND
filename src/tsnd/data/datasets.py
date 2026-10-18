"""Typed loaders for TSND data artifacts.

The pilot artifact formats are preserved:

* ``works_{year}.parquet``: work metadata rows.
* ``emb_{year}_s{seed}.npz``: ``units`` object array and ``mat`` float32 array
  with shape ``(n_units, dim)``.
* ``year_{year}.pkl``: yearly graph dict with keys ``vocab``, ``uidx``,
  ``unit_df``, ``paper_units``, ``edges``, and ``n_papers``.

Pickle and object-array npz files are only loaded when callers pass
``trusted=True``. This keeps the existing interface while preventing accidental
loads from untrusted input.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

WORK_COLUMNS = [
    "id",
    "year",
    "abstract",
    "concepts",
    "subfield_ids",
    "venue_id",
    "cited_by_count",
    "domain",
]

YEAR_GRAPH_KEYS = {"vocab", "uidx", "unit_df", "paper_units", "edges", "n_papers"}


class YearGraph(TypedDict):
    """Pilot yearly graph schema."""

    vocab: list[str]
    uidx: dict[str, int]
    unit_df: dict[str, int]
    paper_units: list[list[str]]
    edges: dict[tuple[int, int], int]
    n_papers: int


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Loaded yearly embedding artifact.

    Attributes:
        units: Unit strings with shape ``(n_units,)``.
        mat: Embedding matrix with shape ``(n_units, dim)`` and dtype
            ``float32``.
        uidx: Unit-to-row lookup for ``mat``.
    """

    units: list[str]
    mat: NDArray[np.float32]
    uidx: dict[str, int]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load``."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config must be a mapping: {config_path}")
    return loaded


def load_works_parquet(
    path: str | Path,
    columns: Sequence[str] | None = None,
    allow_legacy: bool = False,
) -> pd.DataFrame:
    """Load and validate a work parquet artifact.

    Args:
        path: Path to ``works_{year}.parquet``.
        columns: Optional subset of columns to read.
        allow_legacy: Allow pilot parquet without the new ``domain`` column.

    Returns:
        A dataframe with shape ``(n_papers, n_columns)``.
    """
    parquet_path = Path(path)
    read_columns = list(columns) if columns is not None else None
    frame = pd.read_parquet(parquet_path, columns=read_columns)
    required = [column for column in WORK_COLUMNS if allow_legacy or column != "domain"]
    if columns is not None:
        required = [column for column in required if column in columns]
    _require_columns(frame, required)
    if "year" in frame.columns:
        frame["year"] = frame["year"].astype(int)
    if "cited_by_count" in frame.columns:
        frame["cited_by_count"] = frame["cited_by_count"].fillna(0).astype(int)
    return frame


def load_works_year(
    data_dir: str | Path,
    year: int,
    columns: Sequence[str] | None = None,
    allow_legacy: bool = False,
) -> pd.DataFrame:
    """Load one ``works_{year}.parquet`` file from a data directory."""
    if year <= 0:
        raise ValueError("year must be positive")
    return load_works_parquet(
        Path(data_dir) / f"works_{year}.parquet",
        columns=columns,
        allow_legacy=allow_legacy,
    )


def load_embedding_npz(path: str | Path, trusted: bool = False) -> EmbeddingArtifact:
    """Load one embedding npz artifact.

    The stored ``units`` array has dtype ``object`` in the pilot format, so
    loading requires ``trusted=True``. The matrix is returned as float32 with
    shape ``(n_units, dim)``.
    """
    if not trusted:
        raise ValueError("loading object-array npz artifacts requires trusted=True")
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=True) as loaded:
        if "units" not in loaded or "mat" not in loaded:
            raise ValueError(f"embedding npz missing units or mat: {npz_path}")
        units = [str(unit) for unit in loaded["units"].tolist()]
        mat = np.asarray(loaded["mat"], dtype=np.float32)
    _validate_embedding(units, mat, npz_path)
    return EmbeddingArtifact(units=units, mat=mat, uidx={unit: idx for idx, unit in enumerate(units)})


def load_embedding_year(
    embedding_dir: str | Path,
    year: int,
    seed: int,
    trusted: bool = False,
) -> EmbeddingArtifact:
    """Load ``emb_{year}_s{seed}.npz`` from an embedding directory."""
    if year <= 0:
        raise ValueError("year must be positive")
    return load_embedding_npz(Path(embedding_dir) / f"emb_{year}_s{seed}.npz", trusted=trusted)


def _validate_embedding(units: Sequence[str], mat: NDArray[np.float32], path: Path) -> None:
    """Validate embedding array shape and dtype."""
    if mat.ndim != 2:
        raise ValueError(f"embedding mat must have shape (n_units, dim): {path}")
    if len(units) != mat.shape[0]:
        raise ValueError(f"embedding units length does not match mat rows: {path}")
    if mat.dtype != np.float32:
        raise TypeError(f"embedding mat must be float32: {path}")
    if len(set(units)) != len(units):
        raise ValueError(f"embedding units contain duplicates: {path}")


def load_year_graph(path: str | Path, trusted: bool = False) -> YearGraph:
    """Load and validate one yearly graph pkl artifact.

    Loading pickle requires ``trusted=True``. The returned dict preserves
    ``paper_units`` with shape ``(n_papers, variable_units_per_paper)`` and
    integer edge weights over vocab row ids.
    """
    if not trusted:
        raise ValueError("loading pickle graph artifacts requires trusted=True")
    graph_path = Path(path)
    with graph_path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"year graph must be a dict: {graph_path}")
    return validate_year_graph(loaded, graph_path)


def load_year_graph_from_dir(terms_dir: str | Path, year: int, trusted: bool = False) -> YearGraph:
    """Load ``year_{year}.pkl`` from a terms artifact directory."""
    if year <= 0:
        raise ValueError("year must be positive")
    return load_year_graph(Path(terms_dir) / f"year_{year}.pkl", trusted=trusted)


def validate_year_graph(raw: Mapping[str, Any], path: str | Path | None = None) -> YearGraph:
    """Validate a raw mapping against the preserved yearly graph schema."""
    label = str(path) if path is not None else "<memory>"
    missing = YEAR_GRAPH_KEYS.difference(raw)
    if missing:
        raise ValueError(f"year graph missing keys {sorted(missing)}: {label}")
    vocab = _str_list(raw["vocab"], f"{label}.vocab")
    uidx = _str_int_dict(raw["uidx"], f"{label}.uidx")
    unit_df = _str_int_dict(raw["unit_df"], f"{label}.unit_df")
    paper_units = _nested_str_list(raw["paper_units"], f"{label}.paper_units")
    edges = _edge_dict(raw["edges"], f"{label}.edges")
    n_papers = int(raw["n_papers"])
    if n_papers != len(paper_units):
        raise ValueError(f"n_papers does not match paper_units length: {label}")
    if len(vocab) != len(set(vocab)):
        raise ValueError(f"vocab contains duplicates: {label}")
    expected_uidx = {unit: idx for idx, unit in enumerate(vocab)}
    if uidx != expected_uidx:
        raise ValueError(f"uidx does not match vocab order: {label}")
    max_index = len(vocab)
    for left, right in edges:
        if left < 0 or right < 0 or left >= max_index or right >= max_index:
            raise ValueError(f"edge index out of range: {label}")
    return YearGraph(
        vocab=vocab,
        uidx=uidx,
        unit_df=unit_df,
        paper_units=paper_units,
        edges=edges,
        n_papers=n_papers,
    )


def _str_list(value: object, name: str) -> list[str]:
    """Validate a list of strings."""
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return [str(item) for item in value]


def _nested_str_list(value: object, name: str) -> list[list[str]]:
    """Validate a nested list of strings."""
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    nested: list[list[str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, list):
            raise TypeError(f"{name}[{idx}] must be a list")
        nested.append([str(unit) for unit in item])
    return nested


def _str_int_dict(value: object, name: str) -> dict[str, int]:
    """Validate a string-to-int dictionary."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    return {str(key): int(val) for key, val in value.items()}


def _edge_dict(value: object, name: str) -> dict[tuple[int, int], int]:
    """Validate an edge-weight dictionary keyed by integer pairs."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    edges: dict[tuple[int, int], int] = {}
    for key, weight in value.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"{name} keys must be integer pairs")
        left = int(key[0])
        right = int(key[1])
        if left == right:
            raise ValueError(f"{name} contains a self-edge")
        edge = (left, right) if left < right else (right, left)
        edges[edge] = int(weight)
    return edges


def load_graphs(terms_dir: str | Path, years: Sequence[int], trusted: bool = False) -> dict[int, YearGraph]:
    """Load multiple yearly graph artifacts keyed by year."""
    return {int(year): load_year_graph_from_dir(terms_dir, int(year), trusted=trusted) for year in years}


def load_embeddings(
    embedding_dir: str | Path,
    years: Sequence[int],
    seed: int,
    trusted: bool = False,
) -> dict[int, EmbeddingArtifact]:
    """Load multiple yearly embedding artifacts keyed by year."""
    return {int(year): load_embedding_year(embedding_dir, int(year), seed, trusted=trusted) for year in years}


def load_npz_array(path: str | Path, key: str, expected_ndim: int | None = None) -> NDArray[Any]:
    """Load a numeric array from a non-object npz artifact."""
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        if key not in loaded:
            raise ValueError(f"npz missing key {key!r}: {npz_path}")
        array = np.asarray(loaded[key])
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"array {key!r} must have ndim={expected_ndim}: {npz_path}")
    if array.dtype == object:
        raise TypeError(f"numeric npz loader refuses object dtype for {key!r}: {npz_path}")
    return array


def load_labels_parquet(path: str | Path) -> pd.DataFrame:
    """Load a front-label parquet artifact with shape ``(n_fronts, n_columns)``."""
    frame = pd.read_parquet(Path(path))
    _require_columns(frame, ["kind", "key", "t", "breadth", "label"])
    frame["t"] = frame["t"].astype(int)
    frame["label"] = frame["label"].astype(int)
    return frame


def load_feature_parquet(path: str | Path) -> pd.DataFrame:
    """Load a front-feature parquet artifact with shape ``(n_fronts, n_features)``."""
    frame = pd.read_parquet(Path(path))
    _require_columns(frame, ["key", "t", "label"])
    frame["t"] = frame["t"].astype(int)
    frame["label"] = frame["label"].astype(int)
    return frame


def available_years(directory: str | Path, prefix: str, suffix: str) -> list[int]:
    """Return years from files named ``{prefix}{year}{suffix}``."""
    root = Path(directory)
    years: list[int] = []
    for path in root.glob(f"{prefix}*{suffix}"):
        middle = path.name.removeprefix(prefix).removesuffix(suffix)
        try:
            years.append(int(middle))
        except ValueError:
            LOGGER.debug("skip non-year artifact %s", path)
    return sorted(set(years))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate required dataframe columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"dataframe missing required columns: {missing}")
