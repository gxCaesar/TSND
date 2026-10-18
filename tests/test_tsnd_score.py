"""Deterministic axiom-property tests for the TSND pool-relative recombination metric.

These verify, without any GPU or external model, the properties the A1-A9 harness
checks, so the headline claim (TSND satisfies A9 no-future-leakage while static
pool metrics do not) rests on a reproducible unit test rather than a single run.
"""
from __future__ import annotations

import numpy as np

from tsnd.evaluation.tsnd_score import TemporalRecombinationMetric, build_year_map


class _ToyEmbedder:
    """Deterministic embedder: each distinct text maps to a fixed unit vector."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = {k: v / np.linalg.norm(v) for k, v in vectors.items()}
        self._dim = len(next(iter(vectors.values())))

    def __call__(self, texts):
        rows = []
        for text in texts:
            vec = self._vectors.get(text)
            if vec is None:  # unknown synthetic text -> deterministic hash vector
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                vec = rng.standard_normal(self._dim)
                vec = vec / np.linalg.norm(vec)
            rows.append(vec)
        return np.asarray(rows, dtype=np.float64)


def _setup():
    # focal F, prior base docs P1..P3 (years <= 2016), future docs Z1..Z2 (years > 2016).
    vectors = {
        "F": np.array([1.0, 0.0, 0.0]),
        "P1": np.array([0.9, 0.1, 0.0]),
        "P2": np.array([0.0, 1.0, 0.0]),
        "P3": np.array([0.0, 0.0, 1.0]),
        "Z1": np.array([0.95, 0.05, 0.0]),  # future doc very close to F (would lower novelty if counted)
        "Z2": np.array([0.99, 0.0, 0.1]),
    }
    embedder = _ToyEmbedder(vectors)
    base = ("P1", "P2", "P3")
    pool_years = (2012, 2013, 2014)
    future = ("Z1", "Z2")
    year_map = build_year_map(["F"], 2016, base, pool_years, future)
    metric = TemporalRecombinationMetric(embedder, year_map, k_neighbors=2)
    return metric, base, future


def test_a9_no_future_leakage_invariance():
    """Appending future-dated docs to the pool must not change the score (delta=0)."""
    metric, base, future = _setup()
    base_score = metric.score("F", base)
    with_future = metric.score("F", (*base, *future))
    assert abs(with_future - base_score) <= 1e-12, (base_score, with_future)


    def test_future_docs_would_change_a_nonfilter_metric():
        """Confirm a non-temporal metric would move.

        The future documents are near the focal document, so the invariance above is
        non-trivial rather than a consequence of irrelevant future documents.
        """
    metric, base, future = _setup()
    # mean distance to k nearest over base only vs base+future, IGNORING the year filter:
    emb = metric.embedder
    def nofilter_score(focal, pool):
        e = emb([focal, *pool])
        d = 1.0 - e[0] @ e[1:].T
        return float(np.sort(d)[:2].mean())
    assert nofilter_score("F", base) != nofilter_score("F", (*base, *future))


def test_a1_self_recognition_lowers_score():
    """Adding an exact copy of the focal (a present-dated doc) must lower novelty."""
    metric, base, _ = _setup()
    base_score = metric.score("F", base)
    with_copy = metric.score("F", (*base, "F"))  # "F" is in year_map at the focal year -> kept
    assert with_copy < base_score, (base_score, with_copy)


def test_a7_a8_temporal_monotonicity():
    """An older-only prior pool should look at least as novel as a newer-only one."""
    vectors = {
        "F": np.array([1.0, 0.0, 0.0]),
        "OLD1": np.array([0.0, 1.0, 0.0]),   # far from F
        "OLD2": np.array([0.0, 0.0, 1.0]),
        "NEW1": np.array([0.96, 0.28, 0.0]),  # close to F (recent work already near it)
        "NEW2": np.array([0.97, 0.0, 0.24]),
    }
    embedder = _ToyEmbedder(vectors)
    base = ("OLD1", "OLD2", "NEW1", "NEW2")
    pool_years = (2011, 2011, 2015, 2015)
    year_map = build_year_map(["F"], 2016, base, pool_years, ())
    metric = TemporalRecombinationMetric(embedder, year_map, k_neighbors=2)
    old_score = metric.score("F", ("OLD1", "OLD2"))
    new_score = metric.score("F", ("NEW1", "NEW2"))
    assert old_score > new_score, (old_score, new_score)
