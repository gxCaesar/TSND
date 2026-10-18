"""TSND method orchestration: assemble the decomposed-novelty front feature table.

Given the per-year embeddings (.npz), co-occurrence graphs (.pkl) and stable-anchor
alignments (.pkl) produced by the data, embeddings and alignment stages, this computes
the three novelty components (``tsnd.models.decomposition``) for every front born at a
cohort year and returns the feature table the forecaster consumes. The on-disk artifact
formats are the validated pilot formats; loaders here are self-contained so this module
does not depend on the exact embeddings/alignment APIs.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tsnd.models.decomposition import (
    RecombinationResult,
    lexical_emergence,
    neighborhood_shift,
    recombination,
    weighted_degree,
)

logger = logging.getLogger(__name__)

TSND_FEATURE_COLUMNS = ["kind_pair", "le", "rc", "prox", "deg", "n_papers"]


def load_embeddings(
    emb_dir: Path,
    year: int,
    seed: int,
) -> tuple[list[str], npt.NDArray[np.float32], dict[str, int]]:
    """Load year-seed SGNS embeddings. Returns (units, matrix (|V|,dim), unit->row index)."""
    data = np.load(emb_dir / f"emb_{year}_s{seed}.npz", allow_pickle=True)
    units = [str(unit) for unit in data["units"].tolist()]
    uidx = {u: i for i, u in enumerate(units)}
    matrix = np.asarray(data["mat"], dtype=np.float32)
    return units, matrix, uidx


def load_graph(terms_dir: Path, year: int) -> dict[str, Any]:
    """Load the year-t co-occurrence graph artifact (validated pilot pkl schema)."""
    with open(terms_dir / f"year_{year}.pkl", "rb") as handle:
        return cast(dict[str, Any], pickle.load(handle))


def load_boundary(align_dir: Path, t: int, seed: int) -> dict[str, Any]:
    """Load the year-t alignment boundary artifact (contains the rotation R)."""
    with open(align_dir / f"bnd_{t}_s{seed}.pkl", "rb") as handle:
        return cast(dict[str, Any], pickle.load(handle))


def _new_pairs_and_features(
    g_t: dict[str, Any],
    g_p: dict[str, Any],
    prior_units: dict[str, int],
    prior_matrix_norm: npt.NDArray[np.float32],
    deg: dict[str, int],
) -> tuple[list[tuple[str, str]], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    vocab_t = g_t["vocab"]
    prior_edges = {(g_p["vocab"][i], g_p["vocab"][j]) for (i, j) in g_p["edges"]}
    new_pairs: list[tuple[str, str]] = []
    log_deg: list[float] = []
    iu: list[int] = []
    iv: list[int] = []
    prox_rows: list[int] = []
    for (i, j) in g_t["edges"]:
        u, v = vocab_t[i], vocab_t[j]
        key = (u, v) if u < v else (v, u)
        if key in prior_edges:
            continue
        new_pairs.append(key)
        log_deg.append(float(np.log(deg.get(u, 1) * deg.get(v, 1) + 1.0)))
        if u in prior_units and v in prior_units:
            iu.append(prior_units[u])
            iv.append(prior_units[v])
            prox_rows.append(len(new_pairs) - 1)
    prox = np.full(len(new_pairs), np.nan)
    if iu:
        prox[prox_rows] = np.einsum("ij,ij->i", prior_matrix_norm[iu], prior_matrix_norm[iv])
    return new_pairs, np.asarray(log_deg, dtype=np.float64), prox


def compute_decomposition(
    t: int, seed: int, emb_dir: Path, terms_dir: Path, align_dir: Path,
) -> dict[str, object]:
    """Compute LE, NS and RC for fronts born at cohort year t under one seed.

    Returns a dict with keys ``LE`` (unit -> float), ``NS`` (unit -> float), ``rc`` (the
    RecombinationResult) and bookkeeping counts. Raises FileNotFoundError if the
    alignment for this boundary is missing.
    """
    boundary = load_boundary(align_dir, t, seed)
    rotation = boundary["R"]
    units_t, mat_t, uidx_t = load_embeddings(emb_dir, t, seed)
    units_p, mat_p, uidx_p = load_embeddings(emb_dir, t - 1, seed)
    prior_set = set(units_p)
    aligned_t = mat_t @ rotation

    new_units = [u for u in units_t if u not in prior_set]
    le_vals = (
        lexical_emergence(aligned_t[[uidx_t[u] for u in new_units]], mat_p)
        if new_units
        else np.empty(0)
    )
    le = {u: float(v) for u, v in zip(new_units, le_vals)}

    surv = [u for u in units_t if u in prior_set]
    ns_vals = neighborhood_shift(
        aligned_t[[uidx_t[u] for u in surv]], mat_p[[uidx_p[u] for u in surv]],
    ) if surv else np.empty(0)
    ns = {u: float(v) for u, v in zip(surv, ns_vals)}

    g_t = load_graph(terms_dir, t)
    g_p = load_graph(terms_dir, t - 1)
    deg = weighted_degree(g_t["edges"], g_t["vocab"])
    prior_matrix_norm = mat_p / (np.linalg.norm(mat_p, axis=1, keepdims=True) + 1e-9)
    new_pairs, log_deg, prox = _new_pairs_and_features(g_t, g_p, uidx_p, prior_matrix_norm, deg)
    rc_result = recombination(new_pairs, log_deg, prox)

    logger.info("decomposition t=%d seed=%d: %d new terms, %d new pairs, %d surviving",
                t, seed, len(new_units), len(new_pairs), len(surv))
    return {"LE": le, "NS": ns, "rc": rc_result, "deg": deg,
            "n_new_terms": len(new_units), "n_new_pairs": len(new_pairs)}


def assemble_features(
    labels: pd.DataFrame, decomposition: dict[str, object], terms_dir: Path, t: int,
) -> pd.DataFrame:
    """Build the TSND front feature table to merge into the forecaster.

    Args:
        labels: cohort label table with columns kind, key, label.
        decomposition: output of ``compute_decomposition`` for the same cohort/seed.
        terms_dir: directory of year graphs (for the inverted index over introducing papers).
        t: cohort year.

    Returns:
        A DataFrame with columns ``TSND_FEATURE_COLUMNS`` plus key and label.
    """
    le = cast(dict[str, float], decomposition["LE"])
    rc_result = cast(RecombinationResult, decomposition["rc"])
    deg = cast(dict[str, int], decomposition["deg"])
    g_t = load_graph(terms_dir, t)
    inverted: dict[str, list[int]] = {}
    for row, units in enumerate(g_t["paper_units"]):
        for unit in units:
            inverted.setdefault(unit, []).append(row)
    # Precompute one introducing-paper set per unit. A pair's n_papers is then a
    # single intersection of cached sets, instead of rebuilding both operand sets
    # on every pair (the prior O(pairs x papers_per_unit) hot spot that dominated
    # assemble time at scale). Semantics are unchanged: a term keeps its raw
    # introducing-paper count and a pair keeps the deduplicated overlap count.
    inverted_sets: dict[str, set[int]] = {unit: set(rows) for unit, rows in inverted.items()}
    empty_set: set[int] = set()

    rows: list[dict[str, object]] = []
    for r in labels.itertuples(index=False):
        if r.kind == "term":
            n_papers = len(inverted.get(r.key, []))
            rows.append({"kind_pair": 0, "le": le.get(r.key, np.nan), "rc": np.nan,
                         "prox": np.nan, "deg": deg.get(r.key, 0), "n_papers": n_papers,
                         "key": r.key, "label": int(r.label)})
        else:
            u, v = r.key.split("|", 1)
            pair = (u, v) if u < v else (v, u)
            n_papers = len(inverted_sets.get(u, empty_set) & inverted_sets.get(v, empty_set))
            rows.append({"kind_pair": 1, "le": np.nan,
                         "rc": rc_result.rc.get(pair, np.nan),
                         "prox": rc_result.prox.get(pair, np.nan),
                         "deg": min(deg.get(u, 0), deg.get(v, 0)),
                         "n_papers": n_papers, "key": r.key, "label": int(r.label)})
    return pd.DataFrame(rows)
