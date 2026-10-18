"""Uzzi atypical-combinations novelty (baseline B4).

Uzzi, Mukherjee, Stringer, Jones, "Atypical Combinations and Scientific Impact",
Science 2013. The original scores a paper by the z-scores of its cited journal pairs
against a degree-preserving randomization of the citation network. No maintained open
implementation was found in the pilot recon, so this is a faithful reimplementation
adapted to concept co-occurrence: for a concept pair, atypicality is how unexpected the
co-occurrence is under a degree-preserving (configuration-model) null. This is a purely
degree and frequency based combination metric, which is exactly the classical baseline
that our temporally-aligned semantic recombination component is meant to improve on.

Reference values in the original: example conventional z 21.55, atypical z -17.67 over
17.9M Web of Science papers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def configuration_model_expectation(
    deg_u: float, deg_v: float, two_m: float,
) -> float:
    """Expected co-occurrence weight of a pair under a degree-preserving null.

    Args:
        deg_u: weighted degree of unit u.
        deg_v: weighted degree of unit v.
        two_m: sum of all degrees (2 * number of edge endpoints).

    Returns:
        The Chung-Lu expected weight ``deg_u * deg_v / two_m``.
    """
    if two_m <= 0:
        return 0.0
    return float(deg_u * deg_v / two_m)


def atypicality_zscore(observed: float, expected: float) -> float:
    """z-score of an observed co-occurrence against a Poisson(expected) null.

    Higher (more positive) means more conventional; lower (more negative) means more
    atypical, so the novelty feature is the negated z-score.
    """
    if expected <= 0:
        return 0.0
    std = math_sqrt(expected)
    return float((observed - expected) / std) if std > 0 else 0.0


def math_sqrt(x: float) -> float:
    """Square root that returns 0 for non-positive input (degenerate guard)."""
    return float(np.sqrt(x)) if x > 0 else 0.0


def uzzi_features(
    new_pairs: list[tuple[str, str]],
    observed_weight: dict[tuple[str, str], float],
    degree: dict[str, float],
) -> pd.DataFrame:
    """Per-front Uzzi atypicality novelty for new concept pairs.

    Args:
        new_pairs: sorted (u, v) keys of pairs first co-occurring at year t.
        observed_weight: observed co-occurrence weight per pair at year t.
        degree: weighted degree per unit at year t.

    Returns:
        A DataFrame keyed by ``key`` ('u|v') with column ``uzzi_novelty`` = negated
        atypicality z-score (higher means more atypical, so more novel).
    """
    two_m = float(sum(degree.values()))
    rows = []
    for (u, v) in new_pairs:
        exp = configuration_model_expectation(degree.get(u, 1), degree.get(v, 1), two_m)
        obs = observed_weight.get((u, v), 1.0)
        z = atypicality_zscore(obs, exp)
        rows.append({"key": f"{u}|{v}", "uzzi_novelty": -z})
    logger.info("uzzi: scored %d new pairs (two_m=%.0f)", len(new_pairs), two_m)
    return pd.DataFrame(rows)
