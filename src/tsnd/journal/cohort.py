"""Separate raw observation histories from thresholded graph entry."""
from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from tsnd.data.preprocessing import DEFAULT_MESH_CHECK_TAGS, as_list, build_year

Front = tuple[str, str]
Graph = Mapping[str, Any]


def raw_mesh_graph(frame: pd.DataFrame) -> dict[str, Any]:
    """Build an uncapped graph of the retained papers with the production stoplist.

    DF=1 and bounds derived from the input disable vocabulary and per-paper
    truncation. This does not restore papers omitted during corpus acquisition.
    """
    mesh_rows = frame["mesh"].map(as_list)
    all_units = {str(unit) for units in mesh_rows for unit in units}
    terms = {
        "unit_source": "mesh",
        "min_df_per_year": 1,
        "top_terms": max(1, len(all_units)),
        "max_units_per_paper": max(1, max(map(len, mesh_rows), default=0)),
        "unknown_df_rank": 1_000_000_000,
        "mesh_stoplist": DEFAULT_MESH_CHECK_TAGS,
    }
    return build_year(frame, {"terms": terms})


def front_counts(graph: Graph) -> dict[Front, int]:
    """Return document counts keyed by typed, canonical front IDs."""
    vocab = graph["vocab"]
    counts = {("term", unit): int(count) for unit, count in graph["unit_df"].items()}
    for (i, j), count in graph["edges"].items():
        u, v = sorted((vocab[i], vocab[j]))
        counts[("pair", f"{u}|{v}")] = int(count)
    return counts


def load_graph_history(terms_dir: Path, start_year: int, end_year: int) -> dict[int, Graph]:
    """Read trusted project graph artifacts for every year of a closed interval."""
    result: dict[int, Graph] = {}
    for year in range(start_year, end_year + 1):
        with (terms_dir / f"year_{year}.pkl").open("rb") as handle:
            result[year] = pickle.load(handle)  # noqa: S301 - trusted project artifacts
    return result


def _history_fields(years: list[int], year: int, prefix: str) -> dict[str, Any]:
    last = years[-1] if years else None
    gap = year - last - 1 if last is not None else None
    if last is None:
        classification = "first_observed"
    elif gap == 0:
        classification = "present_previous"
    else:
        classification = "reentry_gap_1" if gap == 1 else "reentry_gap_ge2"
    return {
        f"{prefix}_class": classification,
        f"{prefix}_first_observed": last is None,
        f"{prefix}_first_year": years[0] if years else year,
        f"{prefix}_last_prior_year": last,
        f"{prefix}_gap_years": gap,
        f"{prefix}_prior_observed_years": len(years),
    }


def _hidden_cause(front: Front, raw: Graph, graph: Graph, min_df: int) -> str:
    units = front[1].split("|", 1) if front[0] == "pair" else [front[1]]
    if any(raw["unit_df"].get(unit, 0) < min_df for unit in units):
        return "df_threshold"
    if all(unit in graph["uidx"] for unit in units):
        return "paper_unit_cap"  # Both endpoints survived; their raw co-occurrence did not.
    return "vocabulary_or_paper_cap"


def _join_labels(classified: pd.DataFrame, labels: pd.DataFrame | None, year: int) -> pd.DataFrame:
    if labels is None:
        result = classified.copy()
        for name in ("label", "breadth", "thr"):
            result[name] = pd.NA
    else:
        labels = labels.copy()
        if "t" in labels and not labels["t"].eq(year).all():
            raise ValueError(f"Label years do not match cohort {year}.")
        pair_mask = labels["kind"].eq("pair")
        labels.loc[pair_mask, "key"] = labels.loc[pair_mask, "key"].map(
            lambda key: "|".join(sorted(str(key).split("|", 1)))
        )
        columns = [name for name in ("kind", "key", "label", "breadth", "thr") if name in labels]
        result = classified.merge(labels[columns], on=["kind", "key"], how="left", validate="1:1")
        for name in ("label", "breadth", "thr"):
            if name not in result:
                result[name] = pd.NA
    result["label_available"] = result["label"].notna()
    result["label"] = result["label"].astype("Int64")
    return result


def classify_cohort(
    raw_graphs: Mapping[int, Graph],
    graphs: Mapping[int, Graph],
    year: int,
    labels: pd.DataFrame | None = None,
    *,
    start_year: int = 2010,
    min_df: int = 20,
    raw_counts: Mapping[int, Mapping[Front, int]] | None = None,
    graph_counts: Mapping[int, Mapping[Front, int]] | None = None,
) -> pd.DataFrame:
    """Classify all adjacent-year candidates without assigning outcome labels.

    Raw and graph histories are separate dimensions. A graph first observation
    can have earlier raw observations hidden by thresholds or per-paper caps.
    ``*_last_prior_year`` excludes the focal year; gap counts absent full years.
    """
    if year <= start_year or min_df <= 0:
        raise ValueError("A cohort needs at least one prior year and a positive min_df.")
    years = list(range(start_year, year + 1))
    for y in years:
        if graphs[y]["n_papers"] != raw_graphs[y]["n_papers"]:
            raise ValueError(f"Raw and graph corpus sizes differ for year {y}.")
    if raw_counts is None:
        raw_counts = {y: front_counts(raw_graphs[y]) for y in years}
    if graph_counts is None:
        graph_counts = {y: front_counts(graphs[y]) for y in years}
    candidates = sorted(set(graph_counts[year]) - set(graph_counts[year - 1]))
    rows: list[dict[str, Any]] = []
    for front in candidates:
        if front not in raw_counts[year]:
            raise ValueError(f"Graph candidate {front} is absent from raw cohort {year}.")
        raw_years = [y for y in years[:-1] if front in raw_counts[y]]
        graph_years = [y for y in years[:-1] if front in graph_counts[y]]
        if not set(graph_years).issubset(raw_years):
            raise ValueError(f"Graph history for {front} is incompatible with the raw corpus.")
        hidden = [y for y in raw_years if front not in graph_counts[y]]
        causes = {_hidden_cause(front, raw_graphs[y], graphs[y], min_df) for y in hidden}
        previous_cause = (
            _hidden_cause(front, raw_graphs[year - 1], graphs[year - 1], min_df)
            if year - 1 in hidden else "none"
        )
        rows.append({
            "kind": front[0], "key": front[1], "t": year,
            **_history_fields(raw_years, year, "raw"),
            **_history_fields(graph_years, year, "graph"),
            "raw_current_count": raw_counts[year][front],
            "graph_current_count": graph_counts[year][front],
            "raw_prior_count": sum(raw_counts[y].get(front, 0) for y in years[:-1]),
            "graph_prior_count": sum(graph_counts[y].get(front, 0) for y in years[:-1]),
            "hidden_raw_history_years": ",".join(map(str, hidden)),
            "hidden_raw_history_causes": "|".join(sorted(causes)),
            "previous_graph_absence_cause": previous_cause,
            "graph_first_with_raw_history": not graph_years and bool(raw_years),
        })
    if not rows:
        # Derive the same schema from an empty selection of a valid template row.
        template = {
            "kind": "", "key": "", "t": year,
            **_history_fields([], year, "raw"), **_history_fields([], year, "graph"),
            "raw_current_count": 0, "graph_current_count": 0,
            "raw_prior_count": 0, "graph_prior_count": 0,
            "hidden_raw_history_years": "", "hidden_raw_history_causes": "",
            "previous_graph_absence_cause": "none", "graph_first_with_raw_history": False,
        }
        classified = pd.DataFrame([template]).iloc[:0].copy()
    else:
        classified = pd.DataFrame(rows)
    for prefix in ("raw", "graph"):
        for suffix in ("first_year", "last_prior_year", "gap_years", "prior_observed_years"):
            name = f"{prefix}_{suffix}"
            classified[name] = classified[name].astype("Int64")
    return _join_labels(classified, labels, year)


def summarize_cohort(
    classified: pd.DataFrame, baseline: pd.DataFrame, condition: str, year: int,
) -> list[dict[str, Any]]:
    """Count complete candidate sets, including entries and exits against DF20."""
    rows: list[dict[str, Any]] = []
    for kind in ("term", "pair"):
        sub = classified[classified["kind"].eq(kind)]
        current_keys = set(sub["key"])
        baseline_keys = set(baseline.loc[baseline["kind"].eq(kind), "key"])
        row: dict[str, Any] = {
            "condition": condition, "year": year, "kind": kind,
            "n_candidates": len(sub), "n_df20_candidates": len(baseline_keys),
            "n_shared_with_df20": len(current_keys & baseline_keys),
            "n_entries_vs_df20": len(current_keys - baseline_keys),
            "n_exits_vs_df20": len(baseline_keys - current_keys),
            "n_label_available": int(sub["label_available"].sum()),
            "n_label_missing": int((~sub["label_available"]).sum()),
            "n_positive": int(sub["label"].sum()),
            "n_raw_first_observed": int(sub["raw_first_observed"].sum()),
            "n_graph_first_observed": int(sub["graph_first_observed"].sum()),
            "n_graph_first_with_raw_history": int(sub["graph_first_with_raw_history"].sum()),
        }
        for prefix in ("raw", "graph"):
            for value in ("first_observed", "present_previous", "reentry_gap_1", "reentry_gap_ge2"):
                row[f"n_{prefix}_{value}"] = int(sub[f"{prefix}_class"].eq(value).sum())
        for cause in ("df_threshold", "paper_unit_cap", "vocabulary_or_paper_cap"):
            row[f"n_previous_absence_{cause}"] = int(sub["previous_graph_absence_cause"].eq(cause).sum())
        rows.append(row)
    return rows
