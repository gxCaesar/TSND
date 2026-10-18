"""Leakage-safe uptake-breadth labels for research fronts.

A front born at year t (a new term or a new concept pair, relative to year t-1) is
labelled by how broadly it diffuses in the window (t, t+h]:

    B_{t+h}(front) = geometric mean over channels of the count of distinct
                     {venues, institutions, subfields} among papers that USE the front
                     in years (t, t+h].

The label is 1 if B is in the top ``positive_quantile`` of the cohort-year-t stratum,
with a positive-breadth requirement so the mass of zero-uptake fronts is never labelled
positive. Uptake is strictly an outcome: it reads only years strictly after t, while
features read only years up to t. The institutions channel needs OpenAlex authorships
(the full run pulls them); the pilot used two channels (venues, subfields). The active
channel set is configurable.

This module is seed-independent (labels come from raw usage, not embeddings).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

CHANNELS = ("venues", "institutions", "subfields")


@dataclass(frozen=True)
class YearUsage:
    """Per-year inverted index and paper metadata for computing breadth.

    Attributes:
        inverted: unit -> list of paper row indices in that year.
        venue_of: per-paper venue id (or None).
        subfields_of: per-paper list of subfield ids.
        institutions_of: per-paper list of institution ids (empty when not pulled).
    """

    inverted: dict[str, list[int]]
    venue_of: Sequence[str | None]
    subfields_of: Sequence[Sequence[str]]
    institutions_of: Sequence[Sequence[str]]


def build_inverted_index(paper_units: Sequence[Sequence[str]]) -> dict[str, list[int]]:
    """Map each vocabulary unit to the row indices of the papers that contain it."""
    inverted: dict[str, list[int]] = {}
    for row, units in enumerate(paper_units):
        for unit in units:
            inverted.setdefault(unit, []).append(row)
    return inverted


def _channel_counts(
    paper_rows: Iterable[int], usage: YearUsage, channels: Sequence[str],
) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {c: set() for c in channels}
    for row in paper_rows:
        if "venues" in seen:
            v = usage.venue_of[row]
            if v:
                seen["venues"].add(v)
        if "subfields" in seen:
            seen["subfields"].update(usage.subfields_of[row])
        if "institutions" in seen:
            seen["institutions"].update(usage.institutions_of[row])
    return seen


def front_breadth(
    paper_rows_by_year: Sequence[Iterable[int]],
    usages: Sequence[YearUsage],
    channels: Sequence[str],
) -> float:
    """Geometric mean over channels of distinct-entity counts pooled across the window.

    Args:
        paper_rows_by_year: per-window-year iterable of paper rows that use the front.
        usages: per-window-year ``YearUsage`` (same order).
        channels: active breadth channels (subset of ``CHANNELS``).

    Returns:
        The geometric-mean breadth, or 0.0 if the front is used nowhere.
    """
    pooled: dict[str, set[str]] = {c: set() for c in channels}
    for rows, usage in zip(paper_rows_by_year, usages):
        counts = _channel_counts(rows, usage, channels)
        for c in channels:
            pooled[c] |= counts[c]
    sizes = [len(pooled[c]) for c in channels]
    if all(s == 0 for s in sizes):
        return 0.0
    product = 1.0
    for s in sizes:
        product *= max(s, 1)
    return float(product ** (1.0 / len(channels)))


def usage_rows_for_front(
    kind: str, key: str, usages: Sequence[YearUsage],
) -> list[list[int]]:
    """Rows that use a front in each window year. Term: contains the unit. Pair: both."""
    rows_by_year: list[list[int]] = []
    for usage in usages:
        if kind == "term":
            rows_by_year.append(list(usage.inverted.get(key, [])))
        else:
            u, v = key.split("|", 1)
            rows_by_year.append(list(set(usage.inverted.get(u, [])) & set(usage.inverted.get(v, []))))
    return rows_by_year


def label_cohort(
    fronts: pd.DataFrame,
    usages: Sequence[YearUsage],
    channels: Sequence[str],
    positive_quantile: float,
) -> pd.DataFrame:
    """Compute breadth and the top-fraction label for one cohort.

    Args:
        fronts: rows with columns ``kind`` ('term'|'pair'), ``key``, ``t``.
        usages: ``YearUsage`` for window years (t, t+h].
        channels: active breadth channels.
        positive_quantile: top fraction (for example 0.90) labelled positive.

    Returns:
        ``fronts`` with added ``breadth``, ``label`` (rank-based top fraction with a
        positive-breadth requirement, robust to mass ties at zero) and ``thr``.
    """
    if fronts.empty:
        return fronts.assign(breadth=[], label=[], thr=[])
    breadths = [
        front_breadth(usage_rows_for_front(r.kind, r.key, usages), usages, channels)
        for r in fronts.itertuples(index=False)
    ]
    out = fronts.copy()
    out["breadth"] = breadths
    k = max(1, math.ceil((1.0 - positive_quantile) * len(out)))
    eligible = out[out["breadth"] > 0]
    if eligible.empty:
        out["label"] = 0
        out["thr"] = 0.0
        return out
    cutoff = float(eligible["breadth"].nlargest(k).min())
    out["label"] = ((out["breadth"] >= cutoff) & (out["breadth"] > 0)).astype(int)
    out["thr"] = cutoff
    logger.info("cohort t=%s: %d fronts, %d positive (cutoff=%.3f, channels=%s)",
                out["t"].iloc[0] if "t" in out else "?", len(out),
                int(out["label"].sum()), cutoff, ",".join(channels))
    return out


def build_labels(
    fronts: pd.DataFrame,
    usages: Sequence[YearUsage],
    channels: Sequence[str] = ("venues", "subfields"),
    positive_quantile: float = 0.90,
) -> pd.DataFrame:
    """Convenience wrapper over ``label_cohort`` (the public label-building entry point).

    Artifact loading (turning year graphs and works parquet into ``YearUsage``) is done by
    the caller (see ``tsnd.cli``); this keeps the labelling logic IO-free and testable.
    """
    return label_cohort(fronts, usages, channels, positive_quantile)


def assert_no_leakage(feature_years: Sequence[int], label_years: Sequence[int], t: int, h: int) -> None:
    """Raise if any feature year is after t or any label year is outside (t, t+h]."""
    bad_feat = [y for y in feature_years if y > t]
    if bad_feat:
        raise ValueError(f"leakage: feature years {bad_feat} are after cohort year {t}")
    bad_lab = [y for y in label_years if y <= t or y > t + h]
    if bad_lab:
        raise ValueError(f"leakage: label years {bad_lab} outside ({t}, {t + h}]")
