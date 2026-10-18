"""Full-reference outcomes shared across same-snapshot observation views."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tsnd.evaluation.labels import YearUsage, front_breadth, label_cohort, usage_rows_for_front
from tsnd.journal.uncertainty import ID_COLUMNS, canonical_front_ids


def assign_fixed_outcomes(
    candidates: pd.DataFrame,
    usages: Sequence[YearUsage],
    channels: Sequence[str],
    cutoff: float,
) -> pd.DataFrame:
    """Apply one frozen cutoff using only the supplied full-reference outcome usage.

    No quantile is estimated here. In particular, candidate filtering or sampling
    cannot turn a different cohort's quantile into a replacement outcome label.
    """
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("The fixed reference cutoff must be finite and nonnegative.")
    result = canonical_front_ids(candidates, "view candidates").sort_values(ID_COLUMNS).reset_index(drop=True)
    result["breadth"] = [
        front_breadth(usage_rows_for_front(row.kind, row.key, usages), usages, channels)
        for row in result.itertuples(index=False)
    ]
    result["thr"] = cutoff
    result["label"] = ((result["breadth"] >= cutoff) & result["breadth"].gt(0)).astype("int64")
    return result


def build_fixed_reference_outcomes(
    config: Mapping[str, Any],
    data_dir: Path,
    full_terms_dir: Path,
    view_terms_dirs: Mapping[str, Path],
    output_dir: Path,
    *,
    years: Sequence[int] = tuple(range(2011, 2017)),
) -> dict[str, Any]:
    """Estimate full-DF20 adjacent cutoffs once and label every view on that reference.

    Full-reference usage is cached for all forward windows, including training
    outcomes before 2017. All label files use canonical ID order so common-ID
    comparisons share the same seeded fit/calibration partition.
    """
    from tsnd.cli import _build_usage, _fronts_born

    if "full" not in view_terms_dirs or view_terms_dirs["full"].resolve() != full_terms_dir.resolve():
        raise ValueError("The full view must identify the DF20 reference graphs.")
    if int(config["terms"]["min_df_per_year"]) != 20:
        raise ValueError("Reference cutoffs require the registered full DF20 condition.")
    if not years or len(set(years)) != len(years):
        raise ValueError("Reference cohort years must be nonempty and unique.")
    horizon = int(config["label"]["horizon"])
    channels = tuple(channel for channel in config["label"]["breadth_channels"] if channel != "institutions")
    quantile = float(config["label"]["positive_quantile"])
    if horizon != 3 or channels != ("venues", "subfields"):
        raise ValueError("The registered reference uses h=3 and the original venues/subfields breadth.")
    output_dir.mkdir(parents=True, exist_ok=False)
    usage_years = sorted({future for year in years for future in range(year + 1, year + horizon + 1)})
    usages = {year: _build_usage(full_terms_dir, data_dir, year) for year in usage_years}
    (output_dir / "reference_labels").mkdir()
    for view in view_terms_dirs:
        (output_dir / "views" / view).mkdir(parents=True)
    cutoffs, counts = [], []
    for year in sorted(years):
        window = [usages[future] for future in range(year + 1, year + horizon + 1)]
        reference_candidates = _fronts_born(full_terms_dir, year)
        if reference_candidates.empty:
            raise ValueError(f"Full reference cohort {year} is empty; no scientific cutoff is defined.")
        reference = label_cohort(reference_candidates, window, channels, quantile)
        if not reference["breadth"].gt(0).any():
            raise ValueError(
                f"Full reference cohort {year} has no positive breadth; inspect before assigning views."
            )
        cutoff = float(reference["thr"].iloc[0])
        if reference["thr"].nunique() != 1:
            raise ValueError("The reference cohort does not have one shared annual cutoff.")
        candidates = {view: _fronts_born(terms, year) for view, terms in view_terms_dirs.items()}
        for view, frame in candidates.items():
            if frame.empty:
                candidates[view] = pd.DataFrame(columns=ID_COLUMNS)
        union = pd.concat(list(candidates.values()), ignore_index=True).drop_duplicates(ID_COLUMNS)
        labeled_union = assign_fixed_outcomes(union, window, channels, cutoff)
        by_id = labeled_union.set_index(ID_COLUMNS)
        reference = canonical_front_ids(reference).sort_values(ID_COLUMNS).reset_index(drop=True)
        reference_index = pd.MultiIndex.from_frame(reference[ID_COLUMNS])
        aligned_reference = by_id.loc[reference_index].reset_index()
        columns = [*ID_COLUMNS, "breadth", "label", "thr"]
        if not aligned_reference[columns].equals(reference[columns]):
            raise ValueError(
                f"Fixed outcome assignment changed the original full-reference labels in {year}."
            )
        reference.to_parquet(output_dir / "reference_labels" / f"labels_{year}.parquet", index=False)
        reference_ids = set(reference_index)
        cutoffs.append(
            {
                "t": year,
                "cutoff": cutoff,
                "reference_candidates": len(reference),
                "reference_positives": int(reference["label"].sum()),
                "outcome_years": list(range(year + 1, year + horizon + 1)),
                "cutoff_estimations": 1,
            }
        )
        for view, frame in candidates.items():
            ids = pd.MultiIndex.from_frame(canonical_front_ids(frame)[ID_COLUMNS])
            labels = by_id.loc[ids].reset_index().sort_values(ID_COLUMNS).reset_index(drop=True)
            labels.to_parquet(output_dir / "views" / view / f"labels_{year}.parquet", index=False)
            for kind in ("term", "pair"):
                subset = labels[labels["kind"].eq(kind)]
                subset_ids = set(pd.MultiIndex.from_frame(subset[ID_COLUMNS]))
                reference_kind = reference.loc[reference["kind"].eq(kind)]
                reference_kind_ids = set(pd.MultiIndex.from_frame(reference_kind[ID_COLUMNS]))
                counts.append(
                    {
                        "view": view,
                        "t": year,
                        "kind": kind,
                        "cutoff": cutoff,
                        "n_candidates": len(subset),
                        "n_positives": int(subset["label"].sum()),
                        "n_entries_vs_full": len(subset_ids - reference_ids),
                        "n_exits_vs_full": len(reference_kind_ids - subset_ids),
                        "n_shared_with_full": len(subset_ids & reference_kind_ids),
                        "n_missing_outcomes": int(subset["label"].isna().sum()),
                        "outcome_window_includes_feature_selection_years": year + 1 <= 2016,
                    }
                )
    pd.DataFrame(counts).to_csv(output_dir / "cohort_counts.csv", index=False)
    pd.DataFrame(cutoffs).drop(columns="outcome_years").to_csv(output_dir / "cutoffs.csv", index=False)
    result = {
        "status": "complete",
        "reference": "full DF20 adjacent mixed-front cohort",
        "horizon": horizon,
        "channels": list(channels),
        "positive_quantile": quantile,
        "outcome_data_dir": str(data_dir),
        "outcome_terms_dir": str(full_terms_dir),
        "label_order": ID_COLUMNS,
        "usage_years": usage_years,
        "cutoffs": cutoffs,
        "counts": counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def common_membership(
    classified_views: Mapping[str, pd.DataFrame],
    cohort: str,
) -> pd.DataFrame:
    """Intersect both candidate eligibility and requested history membership across views.

    Raw-first masks are intersected after applying each view's own raw-history
    flag. Their shared labels must agree before any forecasting is attempted.
    """
    if cohort not in ("adjacent", "raw_first_observed") or not classified_views:
        raise ValueError("Supply classified views and an adjacent or raw-first cohort.")
    selected: dict[str, pd.DataFrame] = {}
    common: set[tuple[Any, ...]] | None = None
    for view, frame in classified_views.items():
        frame = canonical_front_ids(frame, view)
        if "label" not in frame or frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{view} lacks fixed binary labels.")
        if cohort == "raw_first_observed":
            if "raw_first_observed" not in frame or frame["raw_first_observed"].isna().any():
                raise ValueError(f"{view} lacks raw-history membership.")
            if not frame["raw_first_observed"].isin([True, False]).all():
                raise ValueError(f"{view} has invalid raw-history membership.")
            frame = frame[frame["raw_first_observed"]]
        frame = frame.set_index(ID_COLUMNS)
        selected[view] = frame
        identifiers = set(frame.index)
        common = identifiers if common is None else common & identifiers
    shared_identifiers = sorted(common or set())
    result = pd.DataFrame(shared_identifiers, columns=ID_COLUMNS)
    if result.empty:
        return result.assign(label=pd.Series(dtype="int64"))
    index = pd.MultiIndex.from_tuples(shared_identifiers, names=ID_COLUMNS)
    label_arrays = [frame.loc[index, "label"].astype("int64").tolist() for frame in selected.values()]
    if any(labels != label_arrays[0] for labels in label_arrays[1:]):
        raise ValueError("Common IDs have conflicting fixed-reference outcome labels.")
    result["label"] = label_arrays[0]
    return result
