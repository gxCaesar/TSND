"""Stable-anchor temporal alignment for TSND embeddings.

This module ports the validated pilot alignment stage. Stable anchors are
selected by graph-neighborhood Jaccard, and scipy's orthogonal Procrustes maps
``E_t @ R`` into the ``E_{t-1}`` space.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Mapping, Sequence
from heapq import nlargest
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

LOG = logging.getLogger(__name__)

Config = Mapping[str, Any]
FloatMatrix = npt.NDArray[np.float32]
EdgeMap = Mapping[tuple[int, int], int | float]
PROCRUSTES_METHOD = "procrustes"
CADE_METHOD = "cade"
GRAPH_STABILITY_SELECTION = "graph_stability"
FREQUENCY_SELECTION = "frequency"
PILOT_MIN_ANCHORS = 3


class ConfigError(ValueError):
    """Raised when a TSND alignment configuration is missing or invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load``.

    Args:
        path: YAML config path.

    Returns:
        Parsed config mapping.

    Raises:
        ConfigError: If the config is empty or not a mapping.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"config must be a mapping: {cfg_path}")
    return loaded


def resolve_artifact_dir(cfg: Config, config_path: str | Path | None = None) -> Path:
    """Resolve the artifact directory used by pilot-compatible stages.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path used to infer the project root when
            ``_art_dir`` is absent.

    Returns:
        Artifact directory path.
    """
    raw_art_dir = cfg.get("_art_dir")
    if raw_art_dir is not None:
        return Path(str(raw_art_dir))
    root = _resolve_project_root(cfg, config_path)
    artifacts_cfg = cfg.get("artifacts")
    if isinstance(artifacts_cfg, Mapping) and artifacts_cfg.get("out_dir"):
        return root / str(artifacts_cfg["out_dir"])
    tag = str(cfg.get("run_tag", "main"))
    return root / f"artifacts_{tag}"


def load_emb(emb_dir: Path, year: int, seed: int) -> tuple[list[str], FloatMatrix, dict[str, int]]:
    """Load one pilot-compatible SGNS embedding artifact.

    Args:
        emb_dir: Directory containing ``emb_{year}_s{seed}.npz`` files.
        year: Embedding year.
        seed: SGNS seed.

    Returns:
        Tuple ``(units, mat, uidx)`` where ``mat`` has shape ``(n_units, dim)``
        and dtype float32.
    """
    _require_positive_int_value(year, "year")
    _require_non_negative_int_value(seed, "seed")
    emb_path = emb_dir / f"emb_{year}_s{seed}.npz"
    _require_trusted_artifact_path(emb_path, ".npz")
    # The validated pilot format stores units as an object array, so allow_pickle
    # is restricted to local TSND embedding artifacts with the expected suffix.
    data = np.load(emb_path, allow_pickle=True)
    units = [str(unit) for unit in data["units"].tolist()]
    mat = _as_float_matrix(data["mat"], "mat")
    if len(units) != mat.shape[0]:
        raise ValueError(f"units/mat row mismatch in {emb_path}: {len(units)} vs {mat.shape[0]}")
    uidx = {unit: idx for idx, unit in enumerate(units)}
    return units, mat, uidx


def top_k_neighbors(year_pkl: Path, k: int) -> dict[str, set[str]]:
    """Build top-k weighted co-occurrence neighborhoods for one year.

    Args:
        year_pkl: Path to ``year_{t}.pkl`` graph artifact.
        k: Number of neighbors per unit.

    Returns:
        Mapping from unit string to top-k neighbor unit set.
    """
    _require_positive_int_value(k, "k")
    graph = _load_year_graph(year_pkl)
    vocab = _require_str_list(graph, "vocab")
    edges = _require_edges(graph, "edges")
    adj: dict[int, list[tuple[int | float, int]]] = {}
    for (i, j), weight in edges.items():
        _validate_edge_indices(i, j, len(vocab), year_pkl)
        adj.setdefault(i, []).append((weight, j))
        adj.setdefault(j, []).append((weight, i))
    neighbors: dict[str, set[str]] = {}
    for idx, rows in adj.items():
        top_rows = nlargest(k, rows)
        neighbors[vocab[idx]] = {vocab[j] for _, j in top_rows}
    return neighbors


def jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two neighbor sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        Jaccard similarity in ``[0, 1]``. Empty-empty returns ``0.0`` to
        preserve the pilot alignment behavior.
    """
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_anchors(nbr_t: Mapping[str, set[str]], nbr_p: Mapping[str, set[str]], tau: float) -> list[str]:
    """Select stable anchors by graph-neighborhood Jaccard.

    Args:
        nbr_t: Current-year neighborhood mapping.
        nbr_p: Previous-year neighborhood mapping.
        tau: Jaccard threshold in ``[0, 1]``.

    Returns:
        Anchor unit list from ``V_t`` intersect ``V_{t-1}``.
    """
    _validate_tau(tau, "tau")
    shared = set(nbr_t) & set(nbr_p)
    return [concept for concept in shared if jaccard(nbr_t[concept], nbr_p[concept]) >= tau]


def select_anchors_frequency(graph_t: Mapping[str, Any], graph_p: Mapping[str, Any], top_k: int) -> list[str]:
    """Select anchors as the most frequent shared units, the HistWords default.

    This is the frequency-anchor baseline (ablation ABL-4) that the graph-stability
    anchor selection is compared against. Anchors are the ``top_k`` units present in
    both adjacent years with the highest year-$t$ document frequency.

    Args:
        graph_t: Current-year graph payload with ``vocab`` and ``unit_df``.
        graph_p: Previous-year graph payload with ``vocab``.
        top_k: Number of top-frequency shared units to use as anchors.

    Returns:
        Anchor unit list, deterministically ordered by descending frequency.
    """
    if top_k <= 0:
        raise ConfigError(f"frequency_anchor_top_k must be positive, got {top_k}")
    df_t = graph_t.get("unit_df", {})
    shared = set(graph_t.get("vocab", ())) & set(graph_p.get("vocab", ()))
    ranked = sorted(shared, key=lambda u: (df_t.get(u, 0), u), reverse=True)
    return ranked[:top_k]


def build_boundary(
    emb_dir: Path,
    terms_dir: Path,
    year: int,
    seed: int,
    k: int,
    tau_main: float,
    tau_sweep: Sequence[float],
    min_anchors: int,
    method: str,
    anchor_selection: str = GRAPH_STABILITY_SELECTION,
    freq_top_k: int = 5000,
) -> dict[str, Any] | None:
    """Build one boundary alignment artifact.

    Args:
        emb_dir: Directory containing yearly SGNS ``.npz`` embeddings.
        terms_dir: Directory containing yearly term graph ``.pkl`` files.
        year: Boundary year ``t`` for mapping ``E_t`` into ``E_{t-1}``.
        seed: SGNS seed.
        k: Top-k graph-neighbor count for anchor stability.
        tau_main: Main Jaccard threshold for the rotation.
        tau_sweep: Thresholds for anchor-count diagnostics.
        min_anchors: Minimum stable anchors required to estimate Procrustes.
        method: Alignment method, ``procrustes`` or ``cade``.
        anchor_selection: ``graph_stability`` or ``frequency`` anchor policy.
        freq_top_k: Number of shared high-frequency units for frequency anchors.

    Returns:
        Boundary dictionary or ``None`` if not enough anchors are available.
        The dictionary preserves graph keys ``vocab``, ``uidx``, ``unit_df``,
        ``paper_units``, ``edges``, ``n_papers`` and adds alignment fields,
        including ``R`` and ``aligned_mat`` with shape ``(n_units_t, dim)``.
    """
    method_norm = method.lower()
    if method_norm == PROCRUSTES_METHOD:
        return build_procrustes_boundary(
            emb_dir, terms_dir, year, seed, k, tau_main, tau_sweep, min_anchors,
            anchor_selection=anchor_selection, freq_top_k=freq_top_k,
        )
    if method_norm == CADE_METHOD:
        return build_cade_boundary(emb_dir, terms_dir, year, seed, k, tau_main, tau_sweep, min_anchors)
    raise ConfigError(f"unsupported align.method: {method}")


def build_procrustes_boundary(
    emb_dir: Path,
    terms_dir: Path,
    year: int,
    seed: int,
    k: int,
    tau_main: float,
    tau_sweep: Sequence[float],
    min_anchors: int,
    anchor_selection: str = GRAPH_STABILITY_SELECTION,
    freq_top_k: int = 5000,
) -> dict[str, Any] | None:
    """Build one validated stable-anchor Procrustes boundary.

    Args:
        emb_dir: Directory containing yearly SGNS ``.npz`` embeddings.
        terms_dir: Directory containing yearly term graph ``.pkl`` files.
        year: Boundary year ``t`` for mapping ``E_t`` into ``E_{t-1}``.
        seed: SGNS seed.
        k: Top-k graph-neighbor count for anchor stability.
        tau_main: Main Jaccard threshold for the rotation.
        tau_sweep: Thresholds for anchor-count diagnostics.
        min_anchors: Minimum stable anchors required to estimate Procrustes.
        anchor_selection: ``graph_stability`` or ``frequency`` anchor policy.
        freq_top_k: Number of shared high-frequency units for frequency anchors.

    Returns:
        Boundary dictionary or ``None`` if not enough anchors are available.
        ``aligned_mat`` has shape ``(n_units_t, dim)``.
    """
    from scipy.linalg import orthogonal_procrustes

    _validate_boundary_args(year, seed, k, tau_main, tau_sweep, min_anchors)
    units_t, mat_t, uidx_t = load_emb(emb_dir, year, seed)
    units_p, mat_p, uidx_p = load_emb(emb_dir, year - 1, seed)
    term_path_t = terms_dir / f"year_{year}.pkl"
    term_path_p = terms_dir / f"year_{year - 1}.pkl"
    graph_t = _load_year_graph(term_path_t)
    nbr_t = top_k_neighbors(term_path_t, k)
    nbr_p = top_k_neighbors(term_path_p, k)

    if anchor_selection == FREQUENCY_SELECTION:
        graph_p = _load_year_graph(term_path_p)
        freq_anchors = select_anchors_frequency(graph_t, graph_p, freq_top_k)
        anchors = [concept for concept in freq_anchors if concept in uidx_t and concept in uidx_p]
        counts = {tau_main: len(anchors)}
    else:
        counts = {tau: len(select_anchors(nbr_t, nbr_p, tau)) for tau in tau_sweep}
        anchors = [
            concept
            for concept in select_anchors(nbr_t, nbr_p, tau_main)
            if concept in uidx_t and concept in uidx_p
        ]
    if len(anchors) < min_anchors:
        LOG.warning(
            "boundary %d seed %d: only %d anchors at tau=%.4f, min_anchors=%d",
            year,
            seed,
            len(anchors),
            tau_main,
            min_anchors,
        )
        return None

    x = np.vstack([mat_t[uidx_t[concept]] for concept in anchors])
    y = np.vstack([mat_p[uidx_p[concept]] for concept in anchors])
    if x.shape != y.shape:
        raise ValueError(f"anchor matrix shape mismatch: {x.shape} vs {y.shape}")
    rotation, _ = orthogonal_procrustes(x, y)
    rotation32 = rotation.astype(np.float32)
    aligned_mat = (mat_t @ rotation32).astype(np.float32)
    residual = float(np.linalg.norm(x @ rotation32 - y) / np.sqrt(len(anchors)))

    result = _boundary_graph_payload(graph_t)
    result.update(
        {
            "t": year,
            "seed": seed,
            "tau_main": tau_main,
            "anchors": anchors,
            "n_anchors": len(anchors),
            "anchor_counts_by_tau": counts,
            "R": rotation32,
            "residual": residual,
            "method": PROCRUSTES_METHOD,
            "anchor_selection": anchor_selection,
            "embedding_units": units_t,
            "embedding_uidx": uidx_t,
            "mat": mat_t.astype(np.float32),
            "aligned_mat": aligned_mat,
            "prior_units": units_p,
            "prior_uidx": uidx_p,
        }
    )
    return result


def build_cade_boundary(
    emb_dir: Path,
    terms_dir: Path,
    year: int,
    seed: int,
    k: int,
    tau_main: float,
    tau_sweep: Sequence[float],
    min_anchors: int,
) -> dict[str, Any] | None:
    """CADE/compass fallback entry point.

    CADE is not included because the submitted evidence contains no reference
    implementation for compass vocabulary construction, compass pretraining,
    slice-specific fine-tuning, or serialization into the
    ``emb_{t}_s{seed}.npz`` row contract. The reported runs use Procrustes.
    An explicit CADE request logs this boundary and runs Procrustes.

    Args:
        emb_dir: Directory containing yearly SGNS ``.npz`` embeddings.
        terms_dir: Directory containing yearly term graph ``.pkl`` files.
        year: Boundary year ``t``.
        seed: SGNS seed.
        k: Top-k graph-neighbor count.
        tau_main: Main Jaccard threshold.
        tau_sweep: Thresholds for diagnostics.
        min_anchors: Minimum stable anchors required.

    Returns:
        Procrustes boundary dictionary, marked with ``method_requested='cade'``.
    """
    LOG.warning("align.method=cade requested, but CADE is not ported; falling back to procrustes")
    result = build_procrustes_boundary(emb_dir, terms_dir, year, seed, k, tau_main, tau_sweep, min_anchors)
    if result is not None:
        result["method_requested"] = CADE_METHOD
        result["method"] = PROCRUSTES_METHOD
    return result


def save_boundary(out_dir: Path, boundary: Mapping[str, Any]) -> Path:
    """Save one boundary pickle using the pilot filename schema.

    Args:
        out_dir: Output alignment directory.
        boundary: Boundary dictionary containing ``t`` and ``seed``.

    Returns:
        Saved path.
    """
    year = _require_positive_int_value(_coerce_int(boundary.get("t"), "boundary.t"), "boundary.t")
    seed = _require_non_negative_int_value(
        _coerce_int(boundary.get("seed"), "boundary.seed"),
        "boundary.seed",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bnd_{year}_s{seed}.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(dict(boundary), handle, protocol=4)
    return out_path


def run_alignment(cfg: Config, config_path: str | Path | None = None) -> None:
    """Run alignment for all configured boundaries and seeds.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path for artifact-root inference.
    """
    art_dir = resolve_artifact_dir(cfg, config_path)
    emb_dir = art_dir / "embed"
    terms_dir = art_dir / "terms"
    out_dir = art_dir / "align"
    align_cfg = _require_mapping(cfg, "align")
    tau_sweep = _tau_sweep(align_cfg)
    tau_main = _tau_main(align_cfg, tau_sweep)
    method = str(align_cfg.get("method", PROCRUSTES_METHOD))
    anchor_selection = str(align_cfg.get("anchor_selection", GRAPH_STABILITY_SELECTION))
    freq_top_k = int(align_cfg.get("frequency_anchor_top_k", 5000))
    min_anchors = _min_anchors(align_cfg)
    k_neighbors = _require_positive_int(align_cfg, "k_neighbors")
    y0, y1 = _year_span(cfg)
    for year in range(y0 + 1, y1 + 1):
        for seed in _seed_list(cfg):
            boundary = build_boundary(
                emb_dir=emb_dir,
                terms_dir=terms_dir,
                year=year,
                seed=seed,
                k=k_neighbors,
                tau_main=tau_main,
                tau_sweep=tau_sweep,
                min_anchors=min_anchors,
                method=method,
                anchor_selection=anchor_selection,
                freq_top_k=freq_top_k,
            )
            if boundary is None:
                continue
            out_path = save_boundary(out_dir, boundary)
            LOG.info(
                "boundary %d->%d seed %d: %d anchors (tau=%.4f), residual=%.4f, counts=%s -> %s",
                year,
                year - 1,
                seed,
                boundary["n_anchors"],
                tau_main,
                boundary["residual"],
                boundary["anchor_counts_by_tau"],
                out_path,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for pilot-compatible alignment.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_yaml_config(args.config)
    run_alignment(cfg, args.config)
    return 0


def _resolve_project_root(cfg: Config, config_path: str | Path | None) -> Path:
    """Resolve the project root for path derivation.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path.

    Returns:
        Project root path.
    """
    raw_root = cfg.get("_root")
    if raw_root is not None:
        return Path(str(raw_root))
    if config_path is not None:
        cfg_path = Path(config_path).resolve()
        if cfg_path.parent.name == "configs":
            return cfg_path.parent.parent
        return cfg_path.parent
    return Path.cwd()


def _load_year_graph(path: Path) -> dict[str, Any]:
    """Load a trusted yearly term graph pickle.

    Args:
        path: Path to a local TSND ``year_{t}.pkl`` artifact.

    Returns:
        Graph dictionary.
    """
    _require_trusted_artifact_path(path, ".pkl")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, dict):
        raise ValueError(f"year graph must be a dict: {path}")
    return graph


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


def _boundary_graph_payload(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the preserved term-graph payload for boundary pickles.

    Args:
        graph: Current-year graph mapping.

    Returns:
        Dictionary with keys ``vocab``, ``uidx``, ``unit_df``,
        ``paper_units``, ``edges``, and ``n_papers``.
    """
    required = ("vocab", "uidx", "unit_df", "paper_units", "edges", "n_papers")
    missing = [key for key in required if key not in graph]
    if missing:
        raise ValueError(f"year graph missing required keys: {missing}")
    return {key: graph[key] for key in required}


def _as_float_matrix(value: npt.ArrayLike, name: str) -> FloatMatrix:
    """Validate and coerce an array-like object to a finite float32 matrix.

    Args:
        value: Array-like value.
        name: Name used in error messages.

    Returns:
        Float32 matrix with shape ``(n_rows, dim)``.
    """
    mat = np.asarray(value, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {mat.shape}")
    if mat.shape[0] <= 0 or mat.shape[1] <= 0:
        raise ValueError(f"{name} must have positive shape, got {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} contains non-finite values")
    return mat


def _require_str_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    """Read a required list of strings.

    Args:
        cfg: Mapping.
        key: Key to read.

    Returns:
        List of strings.
    """
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    out = [str(item) for item in value]
    if any(not item for item in out):
        raise ValueError(f"{key} entries must be non-empty")
    return out


def _require_edges(cfg: Mapping[str, Any], key: str) -> EdgeMap:
    """Read and validate graph edge mapping.

    Args:
        cfg: Graph mapping.
        key: Edge key.

    Returns:
        Mapping from ``(i, j)`` to edge weight.
    """
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    for edge, weight in value.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"{key} keys must be (i, j) tuples")
        if not isinstance(edge[0], int) or not isinstance(edge[1], int):
            raise ValueError(f"{key} edge indices must be integers")
        if not isinstance(weight, (int, float)):
            raise ValueError(f"{key} weights must be numeric")
    return value


def _validate_edge_indices(i: int, j: int, vocab_size: int, path: Path) -> None:
    """Validate graph edge indices against vocabulary size.

    Args:
        i: First edge index.
        j: Second edge index.
        vocab_size: Vocabulary size.
        path: Source graph path.
    """
    if i < 0 or j < 0 or i >= vocab_size or j >= vocab_size:
        raise ValueError(f"edge index out of range in {path}: {(i, j)}")


def _validate_boundary_args(
    year: int,
    seed: int,
    k: int,
    tau_main: float,
    tau_sweep: Sequence[float],
    min_anchors: int,
) -> None:
    """Validate alignment boundary arguments.

    Args:
        year: Boundary year ``t``.
        seed: SGNS seed.
        k: Top-k graph-neighbor count.
        tau_main: Main Jaccard threshold.
        tau_sweep: Threshold diagnostics.
        min_anchors: Minimum anchors.
    """
    _require_positive_int_value(year, "year")
    _require_non_negative_int_value(seed, "seed")
    _require_positive_int_value(k, "k")
    _require_positive_int_value(min_anchors, "min_anchors")
    _validate_tau(tau_main, "tau_main")
    if not tau_sweep:
        raise ValueError("tau_sweep must be non-empty")
    for tau in tau_sweep:
        _validate_tau(tau, "tau_sweep item")


def _validate_tau(value: float, name: str) -> None:
    """Validate a Jaccard threshold.

    Args:
        value: Threshold value.
        name: Name used in error messages.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _tau_sweep(align_cfg: Mapping[str, Any]) -> list[float]:
    """Read and validate ``align.tau_sweep``.

    Args:
        align_cfg: Alignment config mapping.

    Returns:
        Tau sweep list.
    """
    raw = align_cfg.get("tau_sweep")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("align.tau_sweep must be a non-empty list")
    tau_sweep = [float(value) for value in raw]
    for tau in tau_sweep:
        _validate_tau(tau, "align.tau_sweep item")
    return tau_sweep


def _tau_main(align_cfg: Mapping[str, Any], tau_sweep: Sequence[float]) -> float:
    """Resolve the main alignment threshold.

    Args:
        align_cfg: Alignment config mapping.
        tau_sweep: Validated tau sweep.

    Returns:
        Main tau threshold.
    """
    raw = align_cfg.get("tau_main")
    if raw is not None:
        tau = float(raw)
        _validate_tau(tau, "align.tau_main")
        return tau
    sorted_sweep = sorted(tau_sweep)
    return float(sorted_sweep[len(sorted_sweep) // 2])


def _min_anchors(align_cfg: Mapping[str, Any]) -> int:
    """Resolve the minimum anchor count.

    Args:
        align_cfg: Alignment config mapping.

    Returns:
        Minimum stable-anchor count.
    """
    raw = align_cfg.get("min_anchors")
    if raw is None:
        return PILOT_MIN_ANCHORS
    return _require_positive_int_value(_coerce_int(raw, "align.min_anchors"), "align.min_anchors")


def _year_span(cfg: Config) -> tuple[int, int]:
    """Read and validate configured data year span.

    Args:
        cfg: TSND config mapping.

    Returns:
        Pair ``(year_start, year_end)``.
    """
    data_cfg = _require_mapping(cfg, "data")
    year_start = _require_positive_int(data_cfg, "year_start")
    year_end = _require_positive_int(data_cfg, "year_end")
    if year_end < year_start:
        raise ConfigError("data.year_end must be >= data.year_start")
    return year_start, year_end


def _seed_list(cfg: Config) -> list[int]:
    """Read and validate ``seed_list`` from config.

    Args:
        cfg: TSND config mapping.

    Returns:
        Non-empty seed list.
    """
    raw = cfg.get("seed_list")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("seed_list must be a non-empty list")
    seeds = [_coerce_int(seed, "seed_list item") for seed in raw]
    for seed in seeds:
        _require_non_negative_int_value(seed, "seed_list item")
    return seeds


def _require_mapping(cfg: Config, key: str) -> Mapping[str, Any]:
    """Read a required mapping config section.

    Args:
        cfg: Parent mapping.
        key: Section key.

    Returns:
        Required mapping section.
    """
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> int:
    """Read a required positive integer config value.

    Args:
        cfg: Config mapping.
        key: Config key.

    Returns:
        Positive integer.
    """
    return _require_positive_int_value(_coerce_int(cfg.get(key), key), key)


def _require_positive_int_value(value: int, name: str) -> int:
    """Validate a positive integer value.

    Args:
        value: Integer value.
        name: Name used in error messages.

    Returns:
        Positive integer.
    """
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def _require_non_negative_int_value(value: int, name: str) -> int:
    """Validate a non-negative integer value.

    Args:
        value: Integer value.
        name: Name used in error messages.

    Returns:
        Non-negative integer.
    """
    if value < 0:
        raise ConfigError(f"{name} must be non-negative")
    return value


def _coerce_int(value: Any, name: str) -> int:
    """Coerce a config value to int without accepting booleans.

    Args:
        value: Config value.
        name: Name used in error messages.

    Returns:
        Integer value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
