"""Leakage-safe rolling splits for TSND front prediction."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class RollingSplit:
    """Frozen train/validation/test front-year split."""

    train_years: tuple[int, ...]
    val_years: tuple[int, ...]
    test_years: tuple[int, ...]
    horizons: tuple[int, ...]
    data_year_end: int


def default_config() -> dict[str, Any]:
    """Return the default chronological dataset splits."""
    return {
        "data": {"year_end": 2024},
        "label": {"horizons": [2, 3, 5]},
        "splits": {
            "train_years": list(range(2010, 2018)),
            "val_years": list(range(2018, 2020)),
            "test_years": list(range(2020, 2023)),
        },
    }


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load`` and split defaults."""
    cfg = default_config()
    if path is None:
        return cfg
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config must be a mapping: {config_path}")
    return _deep_merge(cfg, loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mapping objects."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def rolling_split_from_config(config: Mapping[str, Any]) -> RollingSplit:
    """Construct the frozen rolling split from a config mapping."""
    split_cfg = _split_config(config)
    data_cfg = _mapping(config.get("data"), "config.data")
    label_cfg = _mapping(config.get("label"), "config.label")
    horizons = _horizons(label_cfg)
    data_year_end = int(data_cfg["year_end"])
    split = RollingSplit(
        train_years=_years_from_split(split_cfg, "train"),
        val_years=_years_from_split(split_cfg, "val"),
        test_years=_years_from_split(split_cfg, "test"),
        horizons=horizons,
        data_year_end=data_year_end,
    )
    validate_disjoint_ordered_splits(split)
    return split


def _split_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``splits`` or ``split`` config."""
    raw = config.get("splits", config.get("split", {}))
    return _mapping(raw, "config.splits")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Validate a mapping config section."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _horizons(label_cfg: Mapping[str, Any]) -> tuple[int, ...]:
    """Return configured positive forecast horizons."""
    if "horizons" in label_cfg:
        raw = label_cfg["horizons"]
    elif "horizon" in label_cfg:
        raw = [label_cfg["horizon"]]
    else:
        raise ValueError("config.label requires horizon or horizons")
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError("label horizons must be an iterable of integers")
    horizons = tuple(sorted({int(value) for value in raw}))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("label horizons must be positive")
    return horizons


def _years_from_split(split_cfg: Mapping[str, Any], name: SplitName) -> tuple[int, ...]:
    """Read split years from explicit lists or start/end ranges."""
    years_key = f"{name}_years"
    start_key = f"{name}_start"
    end_key = f"{name}_end"
    if years_key in split_cfg:
        raw = split_cfg[years_key]
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
            raise TypeError(f"splits.{years_key} must be an iterable of years")
        years = tuple(int(year) for year in raw)
    elif start_key in split_cfg and end_key in split_cfg:
        start = int(split_cfg[start_key])
        end = int(split_cfg[end_key])
        if end < start:
            raise ValueError(f"splits.{end_key} must be >= splits.{start_key}")
        years = tuple(range(start, end + 1))
    else:
        raise ValueError(f"config.splits requires {years_key} or {start_key}/{end_key}")
    return validate_years(years, f"splits.{name}")


def validate_years(years: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate a strictly increasing year sequence."""
    if not years:
        raise ValueError(f"{name} must not be empty")
    ordered = tuple(int(year) for year in years)
    if any(year <= 0 for year in ordered):
        raise ValueError(f"{name} contains non-positive years")
    if tuple(sorted(ordered)) != ordered or len(set(ordered)) != len(ordered):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return ordered


def validate_disjoint_ordered_splits(split: RollingSplit) -> None:
    """Ensure train, validation, and test years are disjoint and ordered."""
    groups = [split.train_years, split.val_years, split.test_years]
    all_years = [year for group in groups for year in group]
    if len(all_years) != len(set(all_years)):
        raise ValueError("train/val/test years must be disjoint")
    if max(split.train_years) >= min(split.val_years):
        raise ValueError("train years must end before validation years")
    if max(split.val_years) >= min(split.test_years):
        raise ValueError("validation years must end before test years")
    max_horizon = max(split.horizons)
    if min(split.train_years + split.val_years + split.test_years) + max_horizon > split.data_year_end:
        raise ValueError("no split year has a closed label window")


def label_years(front_year: int, horizon: int) -> tuple[int, ...]:
    """Return label years in the open-closed window ``(t, t+h]``."""
    if front_year <= 0 or horizon <= 0:
        raise ValueError("front_year and horizon must be positive")
    return tuple(range(front_year + 1, front_year + horizon + 1))


def assert_front_window(
    front_year: int,
    feature_years: Sequence[int],
    observed_label_years: Sequence[int],
    horizon: int,
) -> None:
    """Assert that one front's feature and label windows are leakage-safe.

    A front born at year ``t`` may use features from years ``<= t`` only, and
    labels from years ``(t, t+h]`` only.
    """
    if front_year <= 0:
        raise ValueError("front_year must be positive")
    if any(int(year) > front_year for year in feature_years):
        raise AssertionError(f"feature years exceed front year {front_year}: {list(feature_years)}")
    expected_label_years = set(label_years(front_year, horizon))
    actual_label_years = {int(year) for year in observed_label_years}
    if not actual_label_years:
        raise AssertionError("label years must not be empty")
    if not actual_label_years.issubset(expected_label_years):
        raise AssertionError(
            f"label years must be within ({front_year}, {front_year + horizon}]: {sorted(actual_label_years)}"
        )


def assert_closed_label_window(front_year: int, horizon: int, data_year_end: int) -> None:
    """Assert that ``t+h`` is available in the corpus."""
    if front_year + horizon > data_year_end:
        raise AssertionError(
            f"label window for t={front_year}, h={horizon} exceeds data_year_end={data_year_end}"
        )


def eligible_years(years: Sequence[int], horizon: int, data_year_end: int) -> tuple[int, ...]:
    """Return split years whose label windows close by ``data_year_end``."""
    validate_years(tuple(int(year) for year in years), "eligible_years")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return tuple(year for year in years if year + horizon <= data_year_end)


def split_years(split: RollingSplit, name: SplitName, horizon: int | None = None) -> tuple[int, ...]:
    """Return years for one split, optionally filtered to a closed horizon."""
    if name == "train":
        years = split.train_years
    elif name == "val":
        years = split.val_years
    elif name == "test":
        years = split.test_years
    else:
        raise ValueError(f"unknown split name: {name}")
    if horizon is None:
        return years
    return eligible_years(years, horizon, split.data_year_end)


def build_split_frame(split: RollingSplit) -> pd.DataFrame:
    """Build a split manifest with one row per split/year/horizon.

    The returned dataframe has shape ``(n_split_year_horizons, 6)`` and columns
    ``split``, ``t``, ``h``, ``feature_year_max``, ``label_year_start``, and
    ``label_year_end``.
    """
    rows: list[dict[str, int | str]] = []
    for split_name in ("train", "val", "test"):
        for horizon in split.horizons:
            for year in split_years(split, split_name, horizon):
                assert_closed_label_window(year, horizon, split.data_year_end)
                assert_front_window(
                    year,
                    range(min(split.train_years), year + 1),
                    label_years(year, horizon),
                    horizon,
                )
                rows.append(
                    {
                        "split": split_name,
                        "t": year,
                        "h": horizon,
                        "feature_year_max": year,
                        "label_year_start": year + 1,
                        "label_year_end": year + horizon,
                    }
                )
    return pd.DataFrame(rows)


def assert_front_table_leakage_safe(
    frame: pd.DataFrame,
    horizon: int,
    front_year_col: str = "t",
    feature_year_max_col: str = "feature_year_max",
    label_year_start_col: str = "label_year_start",
    label_year_end_col: str = "label_year_end",
) -> None:
    """Validate front-level feature/label windows in a dataframe.

    Args:
        frame: Front table with shape ``(n_fronts, n_columns)``.
        horizon: Forecast horizon ``h``.
        front_year_col: Column containing front birth year ``t``.
        feature_year_max_col: Maximum feature year used by the row.
        label_year_start_col: First label year used by the row.
        label_year_end_col: Last label year used by the row.
    """
    required = [front_year_col, feature_year_max_col, label_year_start_col, label_year_end_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"front table missing required columns: {missing}")
    for row in frame.itertuples(index=False):
        row_map = row._asdict()
        front_year = int(row_map[front_year_col])
        feature_max = int(row_map[feature_year_max_col])
        label_start = int(row_map[label_year_start_col])
        label_end = int(row_map[label_year_end_col])
        assert_front_window(
            front_year,
            range(feature_max, feature_max + 1),
            range(label_start, label_end + 1),
            horizon,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for split manifest validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_yaml_config(args.config)
    split = rolling_split_from_config(config)
    frame = build_split_frame(split)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        LOGGER.info("wrote split manifest with %d rows -> %s", len(frame), out_path)
    else:
        LOGGER.info("validated split manifest with %d rows", len(frame))


if __name__ == "__main__":
    main()
