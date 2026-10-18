"""Static word-embedding recombinant novelty (baseline B6).

Shibayama, Yin, Matsumoto, "Measuring novelty in science with word embedding",
PLOS ONE 2021 (DOI 10.1371/journal.pone.0254034, code github DeyunYinWIPO/Novelty, MIT);
Yin, Wu, Yokota, Matsumoto, Shibayama, "Identify novel elements of knowledge with word
embedding", PLOS ONE 2023 (DOI 10.1371/journal.pone.0284567).

The original measures recombinant novelty as the cosine distance between the embeddings
of a paper's cited references. Adapted to a concept pair born at year t, the recombinant
novelty is the cosine distance between the two units in a STATIC (global) embedding
space. The contrast with our recombination component is deliberate: B6 uses a single
static space, while our RC uses prior-year temporally-aligned proximity and removes the
degree trend. Comparing them isolates the value of temporal alignment.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


def recombinant_distance(
    pair: tuple[str, str], static_emb: dict[str, npt.NDArray[Any]],
) -> float:
    """Cosine distance between the two units of a pair in a static embedding space.

    Returns nan when either unit is absent from the static space (the front then has no
    static recombinant novelty and is left to other features).
    """
    u, v = pair
    if u not in static_emb or v not in static_emb:
        return float("nan")
    a = static_emb[u]
    b = static_emb[v]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    cos = float(a @ b / (na * nb))
    return 1.0 - cos


def shibayama_yin_features(
    new_pairs: list[tuple[str, str]],
    static_emb: dict[str, npt.NDArray[Any]],
) -> pd.DataFrame:
    """Per-front static recombinant novelty for new concept pairs.

    Args:
        new_pairs: sorted (u, v) keys of pairs first co-occurring at year t.
        static_emb: a single static embedding per unit (for example a global SGNS or
            SPECTER vector), not temporally aligned.

    Returns:
        A DataFrame keyed by ``key`` ('u|v') with column ``shibayama_novelty`` = the
        static recombinant cosine distance.
    """
    rows = [
        {"key": f"{u}|{v}", "shibayama_novelty": recombinant_distance((u, v), static_emb)}
        for (u, v) in new_pairs
    ]
    logger.info("shibayama-yin: scored %d new pairs (static space size %d)",
                len(new_pairs), len(static_emb))
    return pd.DataFrame(rows)
