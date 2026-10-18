"""Matched corpus views within one fully acquired, dated snapshot."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def select_corpus_view(
    frame: pd.DataFrame,
    mode: str,
    *,
    cap: int = 15_000,
    seed: int = 0,
    selection_year_end: int = 2016,
) -> pd.DataFrame:
    """Select full, API-order capped, or uniformly random capped yearly inputs.

    Rows after ``selection_year_end`` always remain complete. Acquisition rank
    comes from the same full collection; no current recrawl restores old records.
    """
    if mode not in ("full", "api_order", "random"):
        raise ValueError("mode must be full, api_order, or random.")
    if cap <= 0 or seed < 0:
        raise ValueError("cap must be positive and seed nonnegative.")
    required = {"id", "year", "acquisition_rank"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Corpus view missing columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Corpus IDs, years, and acquisition ranks must be present.")
    if not pd.api.types.is_integer_dtype(frame["acquisition_rank"]) or frame["acquisition_rank"].lt(0).any():
        raise ValueError("Acquisition ranks must be nonnegative integers.")
    if not pd.api.types.is_integer_dtype(frame["year"]):
        raise ValueError("Publication years must be integers.")
    if frame["id"].duplicated().any():
        raise ValueError("A full snapshot must have unique work IDs.")
    if frame.duplicated(["year", "acquisition_rank"]).any():
        raise ValueError("Acquisition ranks must be unique within each year.")
    ordered = frame.sort_values(["year", "acquisition_rank"], kind="stable")
    selected: list[pd.DataFrame] = []
    for year, yearly in ordered.groupby("year", sort=True):
        if mode == "full" or int(year) > selection_year_end or len(yearly) <= cap:
            selected.append(yearly)
        elif mode == "api_order":
            selected.append(yearly.iloc[:cap])
        else:
            # Independent deterministic draws by year, unaffected by other years' sizes.
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(year)]))
            indices = np.sort(rng.choice(len(yearly), size=cap, replace=False))
            selected.append(yearly.iloc[indices])
    if not selected:
        return ordered.reset_index(drop=True)
    return pd.concat(selected, ignore_index=True)


def materialize_corpus_views(
    data_dir: Path,
    output_dir: Path,
    *,
    years: Sequence[int] = tuple(range(2010, 2020)),
    cap: int = 15_000,
    sampling_seeds: Sequence[int] = tuple(range(10)),
    selection_year_end: int = 2016,
) -> pd.DataFrame:
    """Write same-snapshot yearly views and a compact membership-count table.

    Source files must be a completed full acquisition with acquisition_rank.
    This function performs no downloads and never changes source parquets.
    Outcome construction must always read the full source, including outcomes
    for training cohorts whose forward windows overlap feature-selection years.
    """
    if len(set(years)) != len(years) or len(set(sampling_seeds)) != len(sampling_seeds):
        raise ValueError("Year and sampling-seed lists must not contain duplicates.")
    if not years:
        raise ValueError("At least one source year is required.")
    specs = [("full", "full", 0), ("api_order", "api_order", 0)]
    specs.extend((f"random_s{seed}", "random", seed) for seed in sampling_seeds)
    destinations = [output_dir / name / f"works_{year}.parquet" for name, _, _ in specs for year in years]
    if any(path.exists() for path in destinations) or (output_dir / "views.csv").exists():
        raise FileExistsError("Corpus view outputs already exist; use a fresh output directory.")
    rows: list[dict[str, Any]] = []
    for year in sorted(years):
        source = data_dir / f"works_{year}.parquet"
        frame = pd.read_parquet(source)
        if not frame["year"].eq(year).all():
            raise ValueError(f"Source {source} contains a different publication year.")
        for name, mode, seed in specs:
            view = select_corpus_view(frame, mode, cap=cap, seed=seed, selection_year_end=selection_year_end)
            target = output_dir / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            view.to_parquet(target, index=False)
            rows.append({"view": name, "year": year, "n_full": len(frame), "n_retained": len(view),
                         "sampling_seed": seed if mode == "random" else None,
                         "selection_applied": (mode != "full" and year <= selection_year_end
                                               and len(frame) > cap)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "views.csv", index=False)
    return summary
