"""Tests for TSND model-facing matrix outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes

from tests._production_helpers import train_sgns_on_toy_corpus

PROCRUSTES_ROWS = 20
PROCRUSTES_DIM = 8
ROTATION_SEED = 2026
PROCRUSTES_MSE_THRESHOLD = 1e-6
SGNS_DIM = 4
SGNS_SEED = 42
TOY_SGNS_CORPUS = (
    ("graph", "neural", "forecast", "novelty"),
    ("graph", "temporal", "alignment", "novelty"),
    ("citation", "forecast", "graph", "alignment"),
)


def _random_rotation(dim: int, seed: int) -> NDArray[np.float64]:
    """Generate an orthogonal rotation matrix."""
    rng = np.random.default_rng(seed)
    raw_matrix = rng.normal(size=(dim, dim))
    rotation, _ = np.linalg.qr(raw_matrix)
    return rotation.astype(np.float64)


def test_procrustes_does_map_rotated_matrix_close_to_target() -> None:
    """Assert the configured SciPy solver maps a rotated matrix to its target."""
    rng = np.random.default_rng(ROTATION_SEED)
    source = rng.normal(size=(PROCRUSTES_ROWS, PROCRUSTES_DIM)).astype(np.float64)
    rotation = _random_rotation(PROCRUSTES_DIM, ROTATION_SEED)
    target = source @ rotation

    solved_rotation, _ = orthogonal_procrustes(source, target)
    mse = float(np.mean((source @ solved_rotation - target) ** 2))

    assert mse < PROCRUSTES_MSE_THRESHOLD


def test_sgns_wrapper_does_return_embedding_matrix_with_configured_dimension() -> None:
    """Assert production SGNS returns the configured embedding dimension."""
    units, matrix = train_sgns_on_toy_corpus(TOY_SGNS_CORPUS, SGNS_DIM, SGNS_SEED)

    assert units
    assert matrix.shape == (len(units), SGNS_DIM)
