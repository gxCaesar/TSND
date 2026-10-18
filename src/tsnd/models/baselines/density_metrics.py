"""Density-based novelty metrics (baseline B9) that implement the NoveltyMetric protocol.

These are the metrics evaluated by the Axiomatic Benchmark (Liu and Zhai,
arXiv:2604.15145). Reproducing them lets the axiom suite compare our decomposition
components against them and against LLM judges on identical footing.

  RND          Relative Neighbor Density: the proportion of a focal paper's neighbours
               whose local density is lower than the focal paper's own.
  SemNovel     sum of distances to the K nearest prior neighbours, K = max(10, 2% pool).
  Yin          distance of the focal paper from the rest of the corpus (q = 0).
  FastTextLOF  Local Outlier Factor on title embeddings.

Reference axiom pass rates (Liu and Zhai): RND 71.5 percent average (best single metric);
no metric passes both temporal axioms (Yin A7 94 / A8 11, SemNovel A7 0 / A8 100).

Each metric takes an injected embedder ``Callable[[list[str]], np.ndarray]`` returning a
normalized (n, dim) matrix, so the metrics are testable without a specific model. The
production embedder is BGE-M3 (see ``tsnd.models.embeddings.BgeM3Embedder``); FastTextLOF
uses fastText title embeddings.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], npt.NDArray[Any]]


def _pairwise_cosine_distance(matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Cosine distance matrix for L2-normalized rows of ``matrix`` of shape (n, dim)."""
    sims = matrix @ matrix.T
    np.clip(sims, -1.0, 1.0, out=sims)
    return np.asarray(1.0 - sims)


class RndMetric:
    """Relative Neighbor Density novelty (RND)."""

    def __init__(self, embedder: Embedder, k_neighbors: int = 10) -> None:
        self.embedder = embedder
        self.k = k_neighbors

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return relative-neighbor-density novelty for one focal text."""
        if not pool:
            return 1.0
        emb = self.embedder([focal, *pool])
        dist = _pairwise_cosine_distance(emb)
        n = emb.shape[0]
        k = min(self.k, n - 1)
        # local density = mean distance to k nearest neighbours (smaller = denser)
        knn_mean = np.sort(dist, axis=1)[:, 1 : k + 1].mean(axis=1)
        focal_density = knn_mean[0]
        # neighbours of the focal paper
        focal_neighbors = np.argsort(dist[0])[1 : k + 1]
        lower = np.mean(knn_mean[focal_neighbors] > focal_density)
        return float(lower)


class SemNovelMetric:
    """SemNovel novelty: sum of distances to the K nearest prior neighbours.

    Uses the raw embedding space by default. The original uses a t-SNE projection; set
    ``use_tsne=True`` to match it exactly (much slower, documented deviation otherwise).
    """

    def __init__(self, embedder: Embedder, use_tsne: bool = False) -> None:
        self.embedder = embedder
        self.use_tsne = use_tsne

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return summed nearest-neighbor distance for one focal text."""
        if not pool:
            return 1.0
        emb = self.embedder([focal, *pool])
        if self.use_tsne:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(5, emb.shape[0] // 4))
            emb = TSNE(n_components=2, perplexity=perplexity, init="pca").fit_transform(emb)
            emb = emb.astype(np.float64)
            dist = np.linalg.norm(emb[0] - emb[1:], axis=1)
        else:
            dist = _pairwise_cosine_distance(emb)[0, 1:]
        k = max(10, int(0.02 * len(pool)))
        k = min(k, dist.shape[0])
        return float(np.sort(dist)[:k].sum())


class YinMetric:
    """Yin novelty: distance of the focal paper from the rest of the corpus (q = 0)."""

    def __init__(self, embedder: Embedder, quantile: float = 0.0) -> None:
        self.embedder = embedder
        self.quantile = quantile

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return the configured focal-to-pool distance quantile."""
        if not pool:
            return 1.0
        emb = self.embedder([focal, *pool])
        dist = _pairwise_cosine_distance(emb)[0, 1:]
        if self.quantile <= 0.0:
            return float(dist.min())
        return float(np.quantile(dist, self.quantile))


class FastTextLofMetric:
    """Local Outlier Factor novelty on (title) embeddings."""

    def __init__(self, embedder: Embedder, k_neighbors: int = 20) -> None:
        self.embedder = embedder
        self.k = k_neighbors

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return local-outlier-factor novelty for one focal text."""
        if len(pool) < self.k + 1:
            return 1.0
        emb = self.embedder([focal, *pool])
        lof = LocalOutlierFactor(n_neighbors=min(self.k, emb.shape[0] - 1), metric="cosine")
        lof.fit(emb)
        # negative_outlier_factor_: more negative = more outlying = more novel
        return float(-lof.negative_outlier_factor_[0])
