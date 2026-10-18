"""Recompute response-only, strict, and dominance component diagnostics.

The original response-only arithmetic and RNG draw order are preserved.
Strict and dominance criteria change the relabel and drift interventions.
The novel-pair criterion remains unchanged because swapping its partner does
not alter LE or NS. RC reads prior vectors and current degrees, so current-vector
relabeling and drift do not change RC. Shared current vectors couple LE and NS.
These scores measure intervention response and attribution, not unique factor
identification. The same perturbations are used for all three criteria.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from tsnd.models.decomposition import weighted_degree
from tsnd.models.ours import load_boundary, load_embeddings, load_graph


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def run_both(t: int, seed: int, art_dir: Path, delta: float = 0.15,
             n_inj: int = 100) -> dict:
    rng = np.random.default_rng(seed)

    rotation = load_boundary(art_dir / "align", t, seed)["R"]
    with open(art_dir / "decompose" / f"decomp_{t}_s{seed}.pkl", "rb") as handle:
        decomposition = pickle.load(handle)
    rc_result = decomposition["rc"]
    rc_coef = np.asarray(rc_result.coef, dtype=float)
    rc_bias = float(rc_result.bias)

    units_t, mat_t, uidx_t = load_embeddings(art_dir / "embed", t, seed)
    units_p, mat_p, uidx_p = load_embeddings(art_dir / "embed", t - 1, seed)
    prior_set = set(units_p)
    mat_p_norm = mat_p / (np.linalg.norm(mat_p, axis=1, keepdims=True) + 1e-9)

    surviving = [u for u in units_t if u in prior_set]
    if len(surviving) < 30:
        raise SystemExit(f"skipped: only {len(surviving)} surviving units")

    graph_t = load_graph(art_dir / "terms", t)
    deg = weighted_degree(graph_t["edges"], graph_t["vocab"])

    def le_of(vec_t: np.ndarray) -> float:
        aligned = _unit(vec_t @ rotation)
        return float(1.0 - (mat_p_norm @ aligned).max())

    def ns_of(unit: str, vec_t: np.ndarray) -> float:
        aligned = _unit(vec_t @ rotation)
        return float(1.0 - aligned @ _unit(mat_p[uidx_p[unit]]))

    def prox(unit_u: str, unit_v: str) -> float:
        return float(mat_p_norm[uidx_p[unit_u]] @ mat_p_norm[uidx_p[unit_v]])

    def rc_of(unit_u: str, unit_v: str) -> float:
        features = np.array([np.log(deg.get(unit_u, 1) * deg.get(unit_v, 1) + 1.0),
                             prox(unit_u, unit_v)])
        return float(features @ rc_coef + rc_bias)

    dim = mat_t.shape[1]

    # ---- placebo band (RNG draw #1 = choice, then one normal per sampled unit) --------
    sample = list(rng.choice(surviving, size=min(len(surviving), 200), replace=False))
    base_scale = float(np.median([np.linalg.norm(mat_t[uidx_t[u]]) for u in sample])) + 1e-9

    placebo_ns, placebo_dns, placebo_dle = [], [], []
    for u in sample:
        noise = rng.normal(0, 0.01 * base_scale, size=dim)   # same single draw as original
        vec0 = mat_t[uidx_t[u]]
        vecp = vec0 + noise
        ns_p = ns_of(u, vecp)
        placebo_ns.append(ns_p)                              # original semantics: NS level
        placebo_dns.append(abs(ns_p - ns_of(u, vec0)))       # strict: |delta NS|
        placebo_dle.append(abs(le_of(vecp) - le_of(vec0)))   # strict: |delta LE|
    band_ns = float(np.percentile(placebo_ns, 95))
    band_dns = float(np.percentile(placebo_dns, 95))
    band_dle = float(np.percentile(placebo_dle, 95))

    # ---- 1) RELABEL: target = LE, off-target = NS ------------------------------------
    relabel_targets = list(rng.choice(surviving, size=min(n_inj, len(surviving)),
                                      replace=False))
    relabel_hits = 0
    relabel_strict_hits = 0
    relabel_dom_hits = 0
    ns_moved_on_relabel = 0
    for u in relabel_targets:
        noise = rng.normal(0, 0.05 * base_scale, size=dim)   # same single draw as original
        vec0 = mat_t[uidx_t[u]]
        vecp = vec0 + noise
        hit = le_of(vecp) < delta
        relabel_hits += bool(hit)
        d_ns = abs(ns_of(u, vecp) - ns_of(u, vec0))
        d_le = abs(le_of(vecp) - le_of(vec0))
        ns_quiet = d_ns <= band_dns
        ns_moved_on_relabel += (not ns_quiet)
        relabel_strict_hits += bool(hit and ns_quiet)
        # dominance: target's band-normalized response must exceed every off-target's
        relabel_dom_hits += bool(hit and (d_le / band_dle) > (d_ns / band_dns))

    genuine_ok = 0
    for _ in range(n_inj):
        genuine_ok += bool(le_of(rng.normal(0, base_scale, size=dim)) >= delta)

    relabel_acc = (relabel_hits + genuine_ok) / (2 * n_inj)
    relabel_strict = (relabel_strict_hits + genuine_ok) / (2 * n_inj)
    relabel_dom = (relabel_dom_hits + genuine_ok) / (2 * n_inj)

    # ---- 2) NOVEL-PAIR: target = RC; LE/NS invariant by construction ------------------
    candidates = [u for u in surviving if u in uidx_p]
    far_rcs, near_rcs = [], []
    for _ in range(n_inj):
        unit_u = candidates[int(rng.integers(len(candidates)))]  # same single draw
        order = np.argsort(mat_p_norm @ mat_p_norm[uidx_p[unit_u]])
        far_v = units_p[order[0]]
        near_v = units_p[order[-2]] if len(order) >= 2 else units_p[order[-1]]
        far_rcs.append(rc_of(unit_u, far_v))
        near_rcs.append(rc_of(unit_u, near_v))
    band_rc = float(np.percentile(near_rcs, 95))
    pair_acc = float(np.mean([v > band_rc for v in far_rcs]))
    pair_strict = pair_acc   # LE/NS invariant to the partner swap, so quiescence is exact
    pair_dom = pair_acc      # off-target normalized response is exactly 0

    # ---- 3) DRIFT: target = NS, off-target = LE ---------------------------------------
    drift_targets = list(rng.choice(surviving, size=min(n_inj, len(surviving)),
                                    replace=False))
    drift_hits = 0
    drift_strict_hits = 0
    drift_dom_hits = 0
    le_moved_on_drift = 0
    for u in drift_targets:
        noise = rng.normal(0, 0.5 * base_scale, size=dim)    # same single draw as original
        vec0 = mat_t[uidx_t[u]]
        vecp = vec0 + noise
        hit = ns_of(u, vecp) > band_ns
        drift_hits += bool(hit)
        d_le = abs(le_of(vecp) - le_of(vec0))
        d_ns = abs(ns_of(u, vecp) - ns_of(u, vec0))
        le_quiet = d_le <= band_dle
        le_moved_on_drift += (not le_quiet)
        drift_strict_hits += bool(hit and le_quiet)
        drift_dom_hits += bool(hit and (d_ns / band_dns) > (d_le / band_dle))

    drift_acc = drift_hits / n_inj
    drift_strict = drift_strict_hits / n_inj
    drift_dom = drift_dom_hits / n_inj

    return {
        "t": int(t), "seed": int(seed),
        # published (response-only) criterion -- must match artifacts_*/axioms.json
        "relabel_acc": relabel_acc, "pair_acc": pair_acc, "drift_acc": drift_acc,
        "identifiability": float(np.nanmean([relabel_acc, pair_acc, drift_acc])),
        # strict conjunctive criterion
        "relabel_strict": relabel_strict, "pair_strict": pair_strict,
        "drift_strict": drift_strict,
        "identifiability_strict": float(np.nanmean([relabel_strict, pair_strict,
                                                    drift_strict])),
        "relabel_dom": relabel_dom, "pair_dom": pair_dom, "drift_dom": drift_dom,
        "identifiability_dominance": float(np.nanmean([relabel_dom, pair_dom, drift_dom])),
        # coupling diagnostics
        "ns_moved_on_relabel_frac": ns_moved_on_relabel / max(len(relabel_targets), 1),
        "le_moved_on_drift_frac": le_moved_on_drift / max(len(drift_targets), 1),
        "band_ns": band_ns, "band_rc": band_rc, "band_dns": band_dns,
        "band_dle": band_dle, "delta": delta, "n_injections": n_inj,
    }


if __name__ == "__main__":
    art = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts_mesh_onco")
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
    seeds = [int(s) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2"])]

    records = [run_both(year, s, art) for s in seeds]
    out = {
        "art_dir": str(art), "t": year, "seeds": seeds,
        "published_mean": float(np.mean([r["identifiability"] for r in records])),
        "published_std": float(np.std([r["identifiability"] for r in records])),
        "strict_mean": float(np.mean([r["identifiability_strict"] for r in records])),
        "strict_std": float(np.std([r["identifiability_strict"] for r in records])),
        "dominance_mean": float(np.mean([r["identifiability_dominance"] for r in records])),
        "dominance_std": float(np.std([r["identifiability_dominance"] for r in records])),
        "component_means_published": {
            k: float(np.mean([r[k] for r in records]))
            for k in ("relabel_acc", "pair_acc", "drift_acc")},
        "component_means_strict": {
            k: float(np.mean([r[k] for r in records]))
            for k in ("relabel_strict", "pair_strict", "drift_strict")},
        "component_means_dominance": {
            k: float(np.mean([r[k] for r in records]))
            for k in ("relabel_dom", "pair_dom", "drift_dom")},
        "coupling": {
            k: float(np.mean([r[k] for r in records]))
            for k in ("ns_moved_on_relabel_frac", "le_moved_on_drift_frac")},
        "per_seed": records,
    }
    print(json.dumps(out, indent=2))
