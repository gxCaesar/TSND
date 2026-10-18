"""Tests for deterministic TSND seeding and SGNS reproducibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tests._production_helpers import train_sgns_on_toy_corpus
from tsnd.utils.seeding import set_all_seeds

SEED_VALUE = 42
N_RANDOM_DRAWS = 5
SGNS_DIM = 4
TOY_SGNS_CORPUS = (
    ("graph", "neural", "forecast", "novelty"),
    ("graph", "temporal", "alignment", "novelty"),
    ("citation", "forecast", "graph", "alignment"),
)


def _draw_numpy_floats(count: int) -> NDArray[np.float64]:
    """Draw NumPy random floats from the global RNG."""
    return np.random.random(count).astype(np.float64)


def test_set_all_seeds_does_reproduce_numpy_random_draws() -> None:
    """Assert repeated production seeding reproduces NumPy draws."""
    set_all_seeds(SEED_VALUE)
    first_draw = _draw_numpy_floats(N_RANDOM_DRAWS)
    set_all_seeds(SEED_VALUE)
    second_draw = _draw_numpy_floats(N_RANDOM_DRAWS)

    assert np.array_equal(first_draw, second_draw)


def test_sgns_reproducibility_does_return_identical_embeddings_for_same_seed() -> None:
    """Assert repeated production SGNS training returns equal matrices."""
    first_units, first_matrix = train_sgns_on_toy_corpus(TOY_SGNS_CORPUS, SGNS_DIM, SEED_VALUE)
    second_units, second_matrix = train_sgns_on_toy_corpus(TOY_SGNS_CORPUS, SGNS_DIM, SEED_VALUE)

    assert first_units == second_units
    assert np.array_equal(first_matrix, second_matrix)
