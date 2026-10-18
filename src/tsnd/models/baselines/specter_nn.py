"""B5 static SPECTER nearest-neighbor novelty per TSND front.

Source models: https://huggingface.co/allenai/specter2_base and
https://huggingface.co/allenai/specter2.

Paper-level novelty is computed once as chunked ``E_t @ pool.T`` and then
aggregated to fronts. This is a static embedding baseline with its own
model-availability boundary.
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
    front_paper_indices,
    invert_paper_units,
    load_labels,
    load_year_graph,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINE_CONFIG_KEY = "specter_nn"
DEFAULT_EMBEDDING_DIR = "specter"
DEFAULT_EMBEDDING_PATTERN = "paper_emb_{year}.npz"
DEFAULT_LEGACY_NPY_PATTERN = "paper_emb_{year}.npy"
DEFAULT_POOL_CAP = 50_000
DEFAULT_CHUNK_SIZE = 2_000
DEFAULT_NORMALIZE_EMBEDDINGS = True

FEATURE_COLUMNS = ["kind_pair", "specter_novelty", "specter_n_papers"]
FloatArray = npt.NDArray[np.float32]


def build_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build B5 SPECTER nearest-neighbor novelty features.

    Args:
        cohort_year: Birth year ``t`` for front labels.
        config: Runtime configuration. Optional ``specter_nn`` keys include
            ``embedding_dir``, ``embedding_pattern``, ``pool_cap``,
            ``chunk_size``, ``normalize_embeddings``, ``allow_legacy_npy``, and
            ``prior_years``.
        artifact_dir: Artifact root containing labels, terms, and embeddings.
        data_dir: Unused by B5; present for the uniform baseline protocol.
        train_years: Optional fallback for prior pool years. B5 itself is not
            fit on labels.
        seed: Random seed used only when subsampling the prior pool.

    Returns:
        Per-front DataFrame keyed by ``key`` with static embedding novelty.
    """
    del data_dir
    _validate_year(cohort_year)
    art_path = Path(artifact_dir)
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)
    embedding_dir = _embedding_dir(art_path, baseline_cfg)

    labels = load_labels(art_path, cohort_year)
    graph = load_year_graph(art_path, cohort_year)
    inverted = invert_paper_units(graph["paper_units"])
    year_embeddings = load_embedding_matrix(embedding_dir, cohort_year, baseline_cfg)
    _validate_embedding_rows(year_embeddings, expected_rows=int(graph["n_papers"]), year=cohort_year)

    prior_years = resolve_prior_years(cohort_year, config, baseline_cfg, train_years)
    pool = load_prior_pool(embedding_dir, prior_years, baseline_cfg, seed=seed)
    if pool is None:
        novelty = np.full(year_embeddings.shape[0], np.nan, dtype=np.float32)
    else:
        if year_embeddings.shape[1] != pool.shape[1]:
            raise ValueError(
                f"embedding dimensions differ: cohort dim={year_embeddings.shape[1]}, "
                f"pool dim={pool.shape[1]}"
            )
        novelty = paper_nn_novelty(year_embeddings, pool, baseline_cfg)

    rows: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        key = str(getattr(row, "key"))
        kind = str(getattr(row, "kind"))
        paper_indices = front_paper_indices(key, kind, inverted)
        rows.append(
            {
                "key": key,
                "kind_pair": int(kind == "pair"),
                "specter_novelty": aggregate_front_novelty(novelty, paper_indices),
                "specter_n_papers": len(paper_indices),
            }
        )

    out = pd.DataFrame(rows)
    LOGGER.info("built B5 SPECTER NN novelty for %d fronts at t=%d", len(out), cohort_year)
    return out


def load_embedding_matrix(
    embedding_dir: Path,
    year: int,
    config: Mapping[str, Any],
) -> FloatArray:
    """Load a pilot-compatible paper embedding matrix.

    The preferred artifact is ``.npz`` with arrays ``units`` of shape
    ``(n_papers,)`` and ``mat`` of shape ``(n_papers, dim)``. A legacy pilot
    ``.npy`` fallback is allowed only when ``allow_legacy_npy`` is true.

    Args:
        embedding_dir: Directory containing yearly embedding files.
        year: Year to load.
        config: Embedding config.

    Returns:
        Float32 matrix with shape ``(n_papers, dim)``.
    """
    _validate_year(year)
    pattern = str(config.get("embedding_pattern", DEFAULT_EMBEDDING_PATTERN))
    path = embedding_dir / pattern.format(year=year)
    if path.exists():
        _require_trusted_artifact_path(path, ".npz")
        # The TSND embedding contract stores units as an object array; only load
        # these trusted local artifacts from the configured artifact directory.
        with np.load(path, allow_pickle=True) as arrays:
            if "units" not in arrays or "mat" not in arrays:
                raise ValueError(f"embedding npz must contain 'units' and 'mat': {path}")
            units = arrays["units"]
            mat = arrays["mat"]
        if units.ndim != 1:
            raise ValueError(f"embedding units must have shape (n_papers,), got {units.shape}")
        return _validate_embedding_matrix(mat, path=path)

    allow_legacy = bool(config.get("allow_legacy_npy", False))
    if allow_legacy:
        legacy_pattern = str(config.get("legacy_npy_pattern", DEFAULT_LEGACY_NPY_PATTERN))
        legacy_path = embedding_dir / legacy_pattern.format(year=year)
        if legacy_path.exists():
            return _validate_embedding_matrix(np.load(legacy_path), path=legacy_path)
    raise FileNotFoundError(f"SPECTER embedding artifact not found for year {year}: {path}")


def load_prior_pool(
    embedding_dir: Path,
    prior_years: Sequence[int],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> FloatArray | None:
    """Load and optionally subsample the prior-literature embedding pool.

    Args:
        embedding_dir: Directory containing yearly embedding files.
        prior_years: Years strictly earlier than the cohort.
        config: Pool config.
        seed: Random seed for deterministic subsampling.

    Returns:
        Float32 matrix with shape ``(n_pool, dim)``, or ``None`` when no prior
        embeddings are available.
    """
    matrices: list[FloatArray] = []
    for year in prior_years:
        try:
            matrices.append(load_embedding_matrix(embedding_dir, int(year), config))
        except FileNotFoundError:
            LOGGER.warning("SPECTER prior embedding missing for year %d", year)
    if not matrices:
        return None

    pool = np.vstack(matrices).astype(np.float32, copy=False)
    pool_cap = int(config.get("pool_cap", DEFAULT_POOL_CAP))
    if pool_cap <= 0:
        raise ValueError(f"pool_cap must be positive, got {pool_cap}")
    if pool.shape[0] > pool_cap:
        rng = np.random.default_rng(seed)
        indices = rng.choice(pool.shape[0], size=pool_cap, replace=False)
        pool = pool[indices]
    if bool(config.get("normalize_embeddings", DEFAULT_NORMALIZE_EMBEDDINGS)):
        return normalize_rows(pool)
    return pool


def paper_nn_novelty(
    year_embeddings: FloatArray,
    prior_pool: FloatArray,
    config: Mapping[str, Any],
) -> FloatArray:
    """Compute vectorized paper-level nearest-neighbor novelty.

    Args:
        year_embeddings: Matrix ``E_t`` with shape ``(n_papers, dim)``.
        prior_pool: Matrix with shape ``(n_pool, dim)``.
        config: Contains ``chunk_size`` and ``normalize_embeddings``.

    Returns:
        Float32 array with shape ``(n_papers,)`` equal to ``1 - max cosine``.
    """
    _validate_matrix_pair(year_embeddings, prior_pool)
    chunk_size = int(config.get("chunk_size", DEFAULT_CHUNK_SIZE))
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    embeddings = (
        normalize_rows(year_embeddings)
        if bool(config.get("normalize_embeddings", DEFAULT_NORMALIZE_EMBEDDINGS))
        else year_embeddings.astype(np.float32, copy=False)
    )
    pool = prior_pool.astype(np.float32, copy=False)
    novelty = np.empty(embeddings.shape[0], dtype=np.float32)
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        similarities = embeddings[start:stop] @ pool.T
        novelty[start:stop] = 1.0 - similarities.max(axis=1)
    return novelty


def aggregate_front_novelty(paper_novelty: FloatArray, paper_indices: Sequence[int]) -> float:
    """Aggregate paper novelty to a front by mean over introducing papers."""
    if paper_novelty.ndim != 1:
        raise ValueError(f"paper_novelty must have shape (n_papers,), got {paper_novelty.shape}")
    if not paper_indices:
        return float("nan")
    indices = np.asarray(paper_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= paper_novelty.shape[0]):
        raise ValueError("paper index outside paper_novelty shape")
    return float(np.nanmean(paper_novelty[indices]))


def resolve_prior_years(
    cohort_year: int,
    config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    train_years: Sequence[int] | None,
) -> list[int]:
    """Resolve prior years for the nearest-neighbor pool."""
    raw_prior = baseline_config.get("prior_years")
    if raw_prior is not None:
        if not isinstance(raw_prior, Sequence) or isinstance(raw_prior, (str, bytes)):
            raise TypeError("specter_nn.prior_years must be a sequence of years")
        years = [int(year) for year in raw_prior]
    else:
        data_config = config.get("data", {})
        if isinstance(data_config, Mapping) and "year_start" in data_config:
            year_start = int(data_config["year_start"])
            years = list(range(year_start, cohort_year))
        elif train_years is not None:
            years = [int(year) for year in train_years if int(year) < cohort_year]
        else:
            raise ValueError("B5 requires specter_nn.prior_years, data.year_start, or train_years")
    years = sorted({year for year in years if year < cohort_year})
    for year in years:
        _validate_year(year)
    return years


def normalize_rows(matrix: FloatArray) -> FloatArray:
    """L2-normalize rows of a matrix with shape ``(n_rows, dim)``."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must have shape (n_rows, dim), got {matrix.shape}")
    values = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(values / norms, dtype=np.float32)


def _embedding_dir(artifact_dir: Path, config: Mapping[str, Any]) -> Path:
    """Resolve embedding directory from config."""
    raw_dir = config.get("embedding_dir")
    if raw_dir is None:
        return artifact_dir / DEFAULT_EMBEDDING_DIR
    path = Path(str(raw_dir))
    return path if path.is_absolute() else artifact_dir / path


def _validate_embedding_matrix(matrix: npt.NDArray[Any], *, path: Path) -> FloatArray:
    """Validate a loaded embedding matrix."""
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must have shape (n_papers, dim), got {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"embedding matrix must be nonempty: {path}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"embedding matrix must be numeric, got {matrix.dtype}")
    return matrix.astype(np.float32, copy=False)


def _validate_embedding_rows(matrix: FloatArray, *, expected_rows: int, year: int) -> None:
    """Validate that embeddings align with graph paper rows."""
    if expected_rows < 0:
        raise ValueError(f"expected_rows must be nonnegative, got {expected_rows}")
    if matrix.shape[0] < expected_rows:
        raise ValueError(
            f"embedding rows for year {year} ({matrix.shape[0]}) are fewer than graph n_papers "
            f"({expected_rows})"
        )


def _validate_matrix_pair(left: FloatArray, right: FloatArray) -> None:
    """Validate pairwise matrix multiplication shapes."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"expected 2-D matrices, got {left.shape} and {right.shape}")
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"matrix dimensions differ: {left.shape[1]} vs {right.shape[1]}")
    if right.shape[0] == 0:
        raise ValueError("prior_pool must contain at least one row")


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


def _require_trusted_artifact_path(path: Path, suffix: str) -> None:
    """Validate a local TSND artifact path before object-array loading.

    Args:
        path: Artifact path.
        suffix: Required file suffix.

    Raises:
        ValueError: If the path suffix is not the expected artifact type.
    """
    if path.suffix != suffix:
        raise ValueError(f"expected a {suffix} TSND artifact, got {path}")
