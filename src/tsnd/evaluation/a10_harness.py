"""A10 component-response diagnostics on saved TSND artifacts.

Three synthetic interventions probe near-self relabeling, novel-pair response,
and same-term displacement against placebo noise bands. The legacy output key
``identifiability`` is the mean of the three intervention response accuracies.
It is not a proof of unique factor identification. The saved RC coefficients
are reused without refitting. Shared vectors can couple component responses.
"""
from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tsnd.models.decomposition import weighted_degree
from tsnd.models.ours import load_boundary, load_embeddings, load_graph

logger = logging.getLogger(__name__)


def _unit(vector: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return the L2-normalized vector, or the vector itself when its norm is zero."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _cfg(config: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    """Read ``config[section][key]`` with a default, tolerating missing sections."""
    sub = config.get(section, {}) if isinstance(config, Mapping) else {}
    if isinstance(sub, Mapping) and key in sub:
        return sub[key]
    return default


def run_identifiability(t: int, seed: int, art_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Score decomposition identifiability for one boundary year and seed.

    Args:
        t: cohort/boundary year. Reads the t and t-1 embeddings and the year-t graph.
        seed: embedding/decomposition seed.
        art_dir: artifact root (``artifacts_{run_tag}``) holding align/, decompose/, embed/, terms/.
        config: runtime config. Uses ``novelty.delta_relabel`` and ``a10.n_injections``.

    Returns:
        A dict with per-intervention accuracies (relabel_acc, pair_acc, drift_acc), the
        overall ``identifiability`` mean, the placebo bands, and bookkeeping. On too-few
        surviving units it returns a skipped record with nan scores.
    """
    rng = np.random.default_rng(seed)
    delta = float(_cfg(config, "novelty", "delta_relabel", 0.15))
    n_inj = int(_cfg(config, "a10", "n_injections", 100))

    rotation = load_boundary(art_dir / "align", t, seed)["R"]
    with open(art_dir / "decompose" / f"decomp_{t}_s{seed}.pkl", "rb") as handle:
        decomposition = pickle.load(handle)  # noqa: S301  trusted local artifact
    rc_result = decomposition["rc"]
    rc_coef = np.asarray(rc_result.coef, dtype=float)   # [log-deg-product, prox]
    rc_bias = float(rc_result.bias)

    units_t, mat_t, uidx_t = load_embeddings(art_dir / "embed", t, seed)
    units_p, mat_p, uidx_p = load_embeddings(art_dir / "embed", t - 1, seed)
    prior_set = set(units_p)
    mat_p_norm = mat_p / (np.linalg.norm(mat_p, axis=1, keepdims=True) + 1e-9)

    surviving = [u for u in units_t if u in prior_set]
    if len(surviving) < 30:
        logger.warning("a10 t=%d seed=%d skipped: only %d surviving units", t, seed, len(surviving))
        return {"t": int(t), "seed": int(seed), "skipped": True,
                "reason": f"too few surviving units ({len(surviving)})",
                "relabel_acc": float("nan"), "pair_acc": float("nan"),
                "drift_acc": float("nan"), "identifiability": float("nan")}

    graph_t = load_graph(art_dir / "terms", t)
    deg = weighted_degree(graph_t["edges"], graph_t["vocab"])

    def le_of(vec_t: npt.NDArray[Any]) -> float:
        aligned = _unit(vec_t @ rotation)
        return float(1.0 - (mat_p_norm @ aligned).max())

    def ns_of(unit: str, vec_t: npt.NDArray[Any]) -> float:
        aligned = _unit(vec_t @ rotation)
        return float(1.0 - aligned @ _unit(mat_p[uidx_p[unit]]))

    def prox(unit_u: str, unit_v: str) -> float:
        return float(mat_p_norm[uidx_p[unit_u]] @ mat_p_norm[uidx_p[unit_v]])

    def rc_of(unit_u: str, unit_v: str) -> float:
        features = np.array([np.log(deg.get(unit_u, 1) * deg.get(unit_v, 1) + 1.0), prox(unit_u, unit_v)])
        return float(features @ rc_coef + rc_bias)

    sample = list(rng.choice(surviving, size=min(len(surviving), 200), replace=False))
    base_scale = float(np.median([np.linalg.norm(mat_t[uidx_t[u]]) for u in sample])) + 1e-9
    placebo_ns = [ns_of(u, mat_t[uidx_t[u]] + rng.normal(0, 0.01 * base_scale, size=mat_t.shape[1]))
                  for u in sample]
    band_ns = float(np.percentile(placebo_ns, 95))

    # 1) RELABEL: re-embed close to self -> LE should flag (low); random far vectors stay novel.
    relabel_targets = list(rng.choice(surviving, size=min(n_inj, len(surviving)), replace=False))
    relabel_hits = sum(
        le_of(mat_t[uidx_t[u]] + rng.normal(0, 0.05 * base_scale, size=mat_t.shape[1])) < delta
        for u in relabel_targets
    )
    genuine_ok = sum(le_of(rng.normal(0, base_scale, size=mat_t.shape[1])) >= delta for _ in range(n_inj))
    relabel_acc = (relabel_hits + genuine_ok) / (2 * n_inj)

    # 2) NOVEL-PAIR: distant existing units -> high RC vs a fixed high-proximity placebo band.
    candidates = [u for u in surviving if u in uidx_p]
    if len(candidates) < 2:
        band_rc, pair_acc = float("nan"), float("nan")
    else:
        far_rcs, near_rcs = [], []
        for _ in range(n_inj):
            unit_u = candidates[int(rng.integers(len(candidates)))]
            order = np.argsort(mat_p_norm @ mat_p_norm[uidx_p[unit_u]])
            far_v = units_p[order[0]]
            near_v = units_p[order[-2]] if len(order) >= 2 else units_p[order[-1]]
            far_rcs.append(rc_of(unit_u, far_v))
            near_rcs.append(rc_of(unit_u, near_v))
        band_rc = float(np.percentile(near_rcs, 95))
        pair_acc = float(np.mean([value > band_rc for value in far_rcs]))

    # 3) DRIFT: a hard context shift -> NS should fire above the placebo band.
    drift_targets = list(rng.choice(surviving, size=min(n_inj, len(surviving)), replace=False))
    drift_hits = sum(
        ns_of(u, mat_t[uidx_t[u]] + rng.normal(0, 0.5 * base_scale, size=mat_t.shape[1])) > band_ns
        for u in drift_targets
    )
    drift_acc = drift_hits / n_inj

    identifiability = float(np.nanmean([relabel_acc, pair_acc, drift_acc]))
    logger.info("a10 t=%d seed=%d: relabel=%.2f pair=%.2f drift=%.2f -> identifiability=%.3f",
                t, seed, relabel_acc, pair_acc, drift_acc, identifiability)
    return {"t": int(t), "seed": int(seed), "relabel_acc": float(relabel_acc),
            "pair_acc": float(pair_acc), "drift_acc": float(drift_acc),
            "identifiability": identifiability, "band_ns": band_ns, "band_rc": band_rc,
            "delta": delta, "n_injections": n_inj}
