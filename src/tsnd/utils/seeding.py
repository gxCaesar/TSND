"""Reproducible random seed setup for TSND."""

from __future__ import annotations

import importlib
import logging
import os
import random
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"
TORCH_MODULE_NAME = "torch"
MIN_SEED = 0


def _load_torch() -> Any | None:
    """Import PyTorch when it is available."""
    try:
        return importlib.import_module(TORCH_MODULE_NAME)
    except ImportError:
        return None


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    The helper sets `torch.backends.cudnn.deterministic = True` when PyTorch is
    available and calls `torch.use_deterministic_algorithms(True, warn_only=True)`
    when supported by the installed PyTorch version. Downstream embedding arrays
    such as `(n_papers, dim)` matrices should still be treated as config-bound
    artifacts, not regenerated implicitly.

    Args:
        seed: Non-negative integer seed used for all supported RNGs.

    Raises:
        ValueError: If `seed` is negative.
    """
    if seed < MIN_SEED:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ[PYTHONHASHSEED_ENV] = str(seed)

    torch_module = _load_torch()
    if torch_module is None:
        LOGGER.info("PyTorch is not available; skipped torch seed setup.")
        return

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.deterministic = True

    use_deterministic_algorithms = getattr(
        torch_module,
        "use_deterministic_algorithms",
        None,
    )
    if callable(use_deterministic_algorithms):
        use_deterministic_algorithms(True, warn_only=True)
