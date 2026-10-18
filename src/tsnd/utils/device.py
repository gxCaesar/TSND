"""Device selection helpers for PyTorch-backed TSND components."""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch

CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
CUDA_DEVICE_PREFIX = "cuda"
CPU_DEVICE_NAME = "cpu"
DISABLED_CUDA_VALUES = {"", "-1", "none", "None", "NoDevFiles", "void"}
MIN_GPU_INDEX = 0
TORCH_MODULE_NAME = "torch"

LOGGER = logging.getLogger(__name__)


def _load_torch() -> Any | None:
    """Import PyTorch when it is available."""
    try:
        return importlib.import_module(TORCH_MODULE_NAME)
    except ImportError:
        return None


def _parse_visible_devices(raw_devices: str | None) -> list[str] | None:
    """Parse `CUDA_VISIBLE_DEVICES` into visible device tokens.

    Args:
        raw_devices: Raw environment variable value.

    Returns:
        `None` when CUDA visibility is unrestricted, an empty list when CUDA is
        disabled, or the configured visible CUDA token list.
    """
    if raw_devices is None:
        return None

    stripped = raw_devices.strip()
    if stripped in DISABLED_CUDA_VALUES:
        return []

    return [token.strip() for token in stripped.split(",") if token.strip()]


def get_device(prefer_gpu_index: int = MIN_GPU_INDEX) -> torch.device:
    """Select a compute device for PyTorch-backed components.

    The helper reads `CUDA_VISIBLE_DEVICES` and selects the requested visible
    CUDA index only when PyTorch reports CUDA availability. If PyTorch is not
    installed, CUDA is disabled, or the requested device is not visible, it
    returns a CPU device fallback.

    Args:
        prefer_gpu_index: Visible CUDA device index to prefer after environment
            filtering.

    Returns:
        A PyTorch device object when PyTorch is available; otherwise a CPU
        fallback value typed as `torch.device` for callers that accept device-like
        inputs.

    Raises:
        ValueError: If `prefer_gpu_index` is negative.
    """
    if prefer_gpu_index < MIN_GPU_INDEX:
        raise ValueError(f"GPU index must be non-negative, got {prefer_gpu_index}.")

    torch_module = _load_torch()
    if torch_module is None:
        LOGGER.warning("PyTorch is not available; using CPU fallback.")
        return cast("torch.device", CPU_DEVICE_NAME)

    visible_devices = _parse_visible_devices(os.environ.get(CUDA_VISIBLE_DEVICES_ENV))
    if visible_devices == []:
        return cast("torch.device", torch_module.device(CPU_DEVICE_NAME))

    if visible_devices is not None and prefer_gpu_index >= len(visible_devices):
        LOGGER.warning(
            "Requested CUDA index %s is outside CUDA_VISIBLE_DEVICES=%s; using CPU.",
            prefer_gpu_index,
            ",".join(visible_devices),
        )
        return cast("torch.device", torch_module.device(CPU_DEVICE_NAME))

    if not torch_module.cuda.is_available():
        return cast("torch.device", torch_module.device(CPU_DEVICE_NAME))

    cuda_device_count = torch_module.cuda.device_count()
    if prefer_gpu_index >= cuda_device_count:
        LOGGER.warning(
            "Requested CUDA index %s but PyTorch reports %s visible devices; using CPU.",
            prefer_gpu_index,
            cuda_device_count,
        )
        return cast("torch.device", torch_module.device(CPU_DEVICE_NAME))

    return cast("torch.device", torch_module.device(f"{CUDA_DEVICE_PREFIX}:{prefer_gpu_index}"))
