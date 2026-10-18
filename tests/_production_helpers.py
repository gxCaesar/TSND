"""Small deterministic fixtures that exercise the shipped TSND implementation."""

from __future__ import annotations

import pickle
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def train_sgns_on_toy_corpus(
    corpus: Sequence[Sequence[str]],
    dim: int,
    seed: int,
) -> tuple[list[str], NDArray[np.float32]]:
    """Train the production yearly SGNS path on a tiny local graph artifact."""
    from tsnd.models.embeddings import YearlySgnsEmbedder

    normalized = [sorted(set(row)) for row in corpus]
    vocab = sorted({token for row in normalized for token in row})
    uidx = {token: index for index, token in enumerate(vocab)}
    unit_df: Counter[str] = Counter(token for row in normalized for token in row)
    edges: Counter[tuple[int, int]] = Counter()
    for row in normalized:
        indices = [uidx[token] for token in row]
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                edges[(min(left, right), max(left, right))] += 1
    graph = {
        "vocab": vocab,
        "uidx": uidx,
        "unit_df": dict(unit_df),
        "paper_units": normalized,
        "edges": dict(edges),
        "n_papers": len(normalized),
    }
    config = {
        "seed_list": [seed],
        "data": {"year_start": 2020, "year_end": 2020},
        "embed": {
            "sgns_dim": dim,
            "window": 2,
            "min_count": 1,
            "epochs": 8,
            "train_window_years": 1,
            "sg": 1,
            "negative": 5,
        },
    }
    with tempfile.TemporaryDirectory(prefix="tsnd-sgns-test-") as temporary:
        root = Path(temporary)
        terms_dir = root / "terms"
        terms_dir.mkdir()
        with (terms_dir / "year_2020.pkl").open("wb") as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        embedder = YearlySgnsEmbedder(config, terms_dir, root / "embeddings", workers=1)
        units, matrix, _ = embedder.train_year(2020, seed)
    return units, matrix
