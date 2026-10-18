"""Pool-relative, temporally-causal variant of the TSND recombination component.

The A1-A9 pool harness (``tsnd.evaluation.axioms``) scores a ``NoveltyMetric`` with
the signature ``score(focal: str, pool: Sequence[str]) -> float``. TSND is natively a
temporal graph decomposition, not a pool-relative score, so to evaluate it on the SAME
axiom footing as the B9 density baselines we expose a faithful pool-relative variant of
its recombination signal: the novelty of a focal abstract is its mean cosine distance to
its ``k`` nearest TEMPORALLY-PRIOR neighbours.

The decisive, TSND-defining property is the temporal restriction. The metric carries a
``text -> publication_year`` map and, before scoring, drops every pool document whose year
is greater than the focal year. Documents not in the map (synthetic paraphrases/fragments
injected by A2/A3) default to the focal year and stay in the pool. Consequently appending
future-dated documents to the pool cannot move the score, so this metric satisfies
A9 (no-future-leakage) by construction, while the static density baselines (RND, SemNovel,
Yin, FastTextLOF) recompute over the whole pool and fail it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], npt.NDArray[Any]]


class NoveltyMetric(Protocol):
    """Protocol for pool-relative novelty metrics."""

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return one focal-document novelty score."""


class TemporalRecombinationMetric:
    """TSND recombination novelty, evaluated pool-relative with a temporal cutoff.

    Args:
        embedder: maps a list of texts to an L2-normalized ``(n, dim)`` matrix (the same
            embedder the density baselines use, so the comparison is on identical footing).
        year_map: publication year per known text (focal abstracts, base-pool abstracts,
            and future-pool abstracts). Texts absent from the map are treated as same-year
            as the focal (kept in the pool); future-dated texts are dropped before scoring.
        k_neighbors: number of nearest prior neighbours averaged into the novelty score.
    """

    def __init__(self, embedder: Embedder, year_map: Mapping[str, int], k_neighbors: int = 10) -> None:
        self.embedder = embedder
        self.year_map = dict(year_map)
        self.k = k_neighbors

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Mean cosine distance from ``focal`` to its k nearest temporally-prior neighbours."""
        focal_year = self.year_map.get(focal)
        if focal_year is None:
            # The focal abstract is always registered at construction; a miss means an
            # unexpected caller. Fall back to the maximum known year so nothing is dropped.
            focal_year = max(self.year_map.values()) if self.year_map else 0
        prior = [doc for doc in pool if self.year_map.get(doc, focal_year) <= focal_year]
        if not prior:
            return 1.0
        emb = self.embedder([focal, *prior])
        sims = emb[0] @ emb[1:].T
        np.clip(sims, -1.0, 1.0, out=sims)
        dist = 1.0 - sims
        k = min(self.k, dist.shape[0])
        return float(np.sort(dist)[:k].mean())


class TemporalPoolFilterMetric:
    """Apply the D<=t pool cutoff to any pool-relative novelty metric.

    Wrapping a static density metric (RND, SemNovel, Yin, FastTextLOF) with this filter
    makes it satisfy A9 no-future-leakage by construction, since every pool document dated
    after the focal year is dropped before the base metric scores. This is the temporalized
    baseline that shows A9 is a data-access property, not a TSND-exclusive one. The wrapped
    metric still inherits its own behavior on the other axioms, so it can pass A9 yet keep
    failing the coverage and newer-slice axioms and offers no decomposition.
    """

    def __init__(self, base_metric: NoveltyMetric, year_map: Mapping[str, int]) -> None:
        self.base_metric = base_metric
        self.year_map = dict(year_map)

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Score after dropping pool documents dated after the focal year."""
        focal_year = self.year_map.get(focal)
        if focal_year is None:
            focal_year = max(self.year_map.values()) if self.year_map else 0
        prior = [doc for doc in pool if self.year_map.get(doc, focal_year) <= focal_year]
        return float(self.base_metric.score(focal, prior))


def build_year_map(
    focal_texts: Sequence[str],
    focal_year: int,
    base_pool: Sequence[str],
    pool_years: Sequence[int],
    future_docs: Sequence[str],
) -> dict[str, int]:
    """Assemble the text -> year map the temporal metric needs for the A9 cutoff.

    Base-pool abstracts keep their real publication year. Focal abstracts map to the focal
    year. Future-pool abstracts map to a sentinel above the focal year so they are dropped
    when appended to the pool during the A9 test. When a text appears in more than one role
    the earliest year wins, so a future document that also occurs earlier is never spuriously
    dropped.
    """
    year_map: dict[str, int] = {}

    def register(text: str, year: int) -> None:
        existing = year_map.get(text)
        if existing is None or year < existing:
            year_map[text] = year

    for text, year in zip(base_pool, pool_years):
        register(text, int(year))
    for text in focal_texts:
        register(text, int(focal_year))
    future_sentinel = int(focal_year) + 1000
    for text in future_docs:
        register(text, future_sentinel)
    return year_map
