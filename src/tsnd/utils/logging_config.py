"""Logging configuration helpers for TSND."""

from __future__ import annotations

import logging

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name, usually `__name__`.

    Returns:
        The requested `logging.Logger` instance.

    Raises:
        ValueError: If `name` is empty.
    """
    if not name.strip():
        raise ValueError("Logger name must be non-empty.")
    return logging.getLogger(name)


def configure_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure the process root logger.

    Args:
        level: Root logging level name, for example `INFO` or `DEBUG`.

    Raises:
        ValueError: If `level` is not a valid logging level name.
    """
    level_name = level.upper()
    numeric_level = logging.getLevelName(level_name)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}.")

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
