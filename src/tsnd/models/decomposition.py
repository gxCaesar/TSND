"""Typed temporal novelty components for concept terms and pairs.

Given year-t and year-(t-1) embeddings aligned by graph-stable Procrustes,
lexical emergence measures an entering term's distance from its nearest prior
vector. Neighborhood shift measures a surviving term's aligned displacement.
Recombination residualizes an eligible entering pair's prior semantic distance
against its current log distinct-partner degree product.

Component applicability is explicit and missing prior proximity is preserved.
The components do not form an independently identified additive partition of a
validated scalar. Component-response diagnostics can exhibit shared-vector
coupling. All functions operate on NumPy arrays without data acquisition.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

NumericArray = npt.NDArray[Any]


def unit_normalize(matrix: NumericArray) -> NumericArray:
    """L2-normalize rows of ``matrix`` of shape (n, dim). Zero rows are left unchanged."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms)


def lexical_emergence(
    new_vecs_aligned: NumericArray, prior_matrix: NumericArray,
) -> NumericArray:
    """Lexical emergence per new unit.

    Args:
        new_vecs_aligned: (n_new, dim) year-t vectors of first-appearing units, already
            mapped into the prior space (``E_t @ R``).
        prior_matrix: (n_prior, dim) year-(t-1) unit vectors.

    Returns:
        (n_new,) array of ``1 - max cosine similarity`` to any prior unit. Higher means
        further from everything that existed, so more genuinely new.
    """
    if new_vecs_aligned.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    q = unit_normalize(new_vecs_aligned)
    p = unit_normalize(prior_matrix)
    max_sim = (q @ p.T).max(axis=1)
    return np.asarray(1.0 - max_sim)


def neighborhood_shift(
    surv_vecs_aligned: NumericArray, surv_vecs_prior: NumericArray,
) -> NumericArray:
    """Neighborhood shift (drift) per surviving unit.

    Args:
        surv_vecs_aligned: (n_surv, dim) year-t vectors mapped into the prior space.
        surv_vecs_prior: (n_surv, dim) the same units' year-(t-1) vectors (aligned order).

    Returns:
        (n_surv,) array of cosine distance between a unit's aligned year-t vector and its
        year-(t-1) vector. Higher means the unit drifted more.
    """
    a = unit_normalize(surv_vecs_aligned)
    b = unit_normalize(surv_vecs_prior)
    return np.asarray(1.0 - (a * b).sum(axis=1))


@dataclass(frozen=True)
class RecombinationResult:
    """Output of the recombination component."""

    rc: dict[tuple[str, str], float]
    prox: dict[tuple[str, str], float]
    coef: tuple[float, float]   # linear RC coefficients on [log(deg-prod), prox]
    bias: float
    n_defined: int


def recombination(
    new_pairs: list[tuple[str, str]],
    log_deg_product: NumericArray,
    prox: NumericArray,
) -> RecombinationResult:
    """Degree-residualized prior semantic distance for new concept pairs.

    Args:
        new_pairs: sorted (u, v) keys of pairs first co-occurring at year t.
        log_deg_product: (n_pairs,) of ``log(deg_t(u) * deg_t(v) + 1)``.
        prox: (n_pairs,) of prior-year aligned cosine proximity, nan where a unit has no
            prior vector (such a pair takes its novelty from LE, not RC).

    Returns:
        A ``RecombinationResult``. RC is ``(1 - prox)`` residualized on the log
        degree-product via a year-t least-squares fit, so it is the semantic
        recombination distance with the degree trend removed. The linear coefficients
        ``coef`` and ``bias`` express RC as a function of [log-deg-product, prox] for the
        A10 audit: ``RC = coef[0] * log_deg + coef[1] * prox + bias``.
    """
    if not new_pairs:
        return RecombinationResult({}, {}, (0.0, -1.0), 1.0, 0)
    rawnov = 1.0 - prox                               # nan where prox undefined
    defined = ~np.isnan(rawnov)
    n_def = int(defined.sum())
    if n_def > 10:
        slope, intercept = np.polyfit(log_deg_product[defined], rawnov[defined], 1)
    else:
        slope = 0.0
        intercept = float(np.nanmean(rawnov)) if defined.any() else 0.0
    resid = rawnov - (slope * log_deg_product + intercept)
    rc: dict[tuple[str, str], float] = {}
    prox_out: dict[tuple[str, str], float] = {}
    for k in np.where(defined)[0]:
        rc[new_pairs[k]] = float(resid[k])
        prox_out[new_pairs[k]] = float(prox[k])
    # RC = -slope*logdeg - prox + (1 - intercept)
    coef = (float(-slope), -1.0)
    bias = float(1.0 - intercept)
    logger.info("recombination: %d new pairs, %d with prior proximity, slope=%.4f",
                len(new_pairs), n_def, slope)
    return RecombinationResult(rc, prox_out, coef, bias, n_def)


def weighted_degree(edges: dict[tuple[int, int], int], vocab: list[str]) -> dict[str, int]:
    """Distinct-partner degree per unit from a co-occurrence edge dict on vocab indices."""
    deg: dict[str, int] = {}
    for (i, j) in edges:
        deg[vocab[i]] = deg.get(vocab[i], 0) + 1
        deg[vocab[j]] = deg.get(vocab[j], 0) + 1
    return deg
