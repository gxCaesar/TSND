"""Explicit RC-block comparisons using the unchanged calibrated forecast recipe."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tsnd.journal.uncertainty import (
    CELL_COLUMNS,
    ID_COLUMNS,
    _comparisons,
    canonical_front_ids,
    node_multiplier_sensitivity,
    validate_predictions,
)
from tsnd.models.baseline_registry import forecast_feature_groups
from tsnd.models.decomposition import weighted_degree
from tsnd.models.ours import load_graph
from tsnd.training.forecaster import ForecasterSettings, _fit_eval_lightgbm

LOGGER = logging.getLogger(__name__)


def _union(groups: Mapping[str, Sequence[str]], members: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(column for member in members for column in groups[member]))


def journal_feature_groups() -> dict[str, tuple[str, ...]]:
    """Return exact B2+B7 structure plus explicit proximity/degree/RC additions.

    ``S`` keeps the original B2-then-B7 registry order. No native ``deg`` column
    is added to these new comparisons, since ``degree_min`` already expresses it.
    Equal-valued columns already present in the historical registry are retained.

    Returns:
        Ordered column groups for the seven pair-level comparisons.
    """
    structure = _union(forecast_feature_groups(["b2", "b7"]), ["b2", "b7"])
    return {
        "S": structure,
        "S+prox": (*structure, "prox"),
        "S+z": (*structure, "z"),
        "S+rc": (*structure, "rc"),
        "S+prox+z": (*structure, "prox", "z"),
        "S+prox+rc": (*structure, "prox", "rc"),
        "S+prox+z+rc": (*structure, "prox", "z", "rc"),
    }


def legacy_feature_groups() -> dict[str, tuple[str, ...]]:
    """Return the original mixed-front groups, preserving registry union order.

    Returns:
        Historical structural baseline and full TSND-then-B2-then-B7 union.
    """
    groups = forecast_feature_groups(["b2", "b7"])
    return {
        "legacy_S": _union(groups, ["b2", "b7"]),
        "legacy_mixed": _union(groups, ["tsnd", "b2", "b7"]),
    }


def validate_labeled_fronts(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate frozen labels without dropping missing or malformed identifiers.

    Args:
        frame: Front table carrying IDs and frozen binary labels.
        source: Source name for errors.

    Returns:
        Canonical table with the original labels and row order.

    Raises:
        ValueError: If ID or label fields are invalid.
    """
    out = canonical_front_ids(frame, source)
    if "label" not in out or out["label"].isna().any() or not out["label"].isin([0, 1]).all():
        raise ValueError(f"{source}: missing or nonbinary frozen labels")
    out["label"] = out["label"].astype(np.int64)
    return out


def attach_front_metadata(wide: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach IDs to provider rows only after checking its exact frozen-label order.

    Args:
        wide: Table assembled by the original CLI provider.
        labels: Concatenated label metadata in the provider's year/row order.

    Returns:
        Feature table with ``t`` and ``kind`` attached and canonical IDs.

    Raises:
        ValueError: If row counts, keys, types, or labels differ.
    """
    if len(wide) != len(labels):
        raise ValueError("provider rows differ from the frozen label row count")
    required = ["key", "kind_pair", "label", "prox", "rc"]
    if any(column not in wide for column in required):
        raise ValueError(f"provider table requires columns {required}")
    out = wide.reset_index(drop=True).copy()
    labels = validate_labeled_fronts(labels, "frozen labels").reset_index(drop=True)
    out["t"] = labels["t"]
    out["kind"] = np.where(out["kind_pair"] == 1, "pair", "term")
    if not out["kind_pair"].isin([0, 1]).all():
        raise ValueError("provider kind_pair must be binary")
    out = validate_labeled_fronts(out, "provider")
    if not out[[*ID_COLUMNS, "label"]].equals(labels[[*ID_COLUMNS, "label"]]):
        raise ValueError("provider order, IDs, types, or frozen labels do not match input label files")
    for column in ["prox", "rc"]:
        numeric = pd.to_numeric(out[column], errors="raise").to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(f"provider {column} contains infinite values")
    out["prox_defined"] = out["prox"].notna()
    return out


def add_degree_product(frame: pd.DataFrame, terms_dir: Path) -> pd.DataFrame:
    """Add the exact native log distinct-partner degree product, leaving terms NaN.

    Args:
        frame: Labeled native feature table.
        terms_dir: Frozen year-graph directory.

    Returns:
        Copy with ``z = log(deg_t(u) * deg_t(v) + 1)`` for pair rows.

    Raises:
        ValueError: If a scored pair endpoint is absent from its focal graph.
    """
    out = frame.copy()
    out["z"] = np.nan
    for year, block in out.groupby("t", sort=False):
        graph = load_graph(terms_dir, int(year))
        degree = weighted_degree(graph["edges"], graph["vocab"])
        pair_keys = block.loc[block["kind"] == "pair", "key"]
        values: list[float] = []
        for key in pair_keys:
            unit_u, unit_v = key.split("|")
            if unit_u not in degree or unit_v not in degree:
                raise ValueError(f"pair {key!r} lacks a focal-year graph endpoint in {year}")
            values.append(float(np.log(degree[unit_u] * degree[unit_v] + 1.0)))
        out.loc[pair_keys.index, "z"] = values
    return out


def select_fronts(
    frame: pd.DataFrame,
    cohort: str,
    classified: pd.DataFrame | None = None,
    common_ids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply ID masks and history classifications without recomputing outcomes.

    Args:
        frame: Adjacent-year frozen-label feature table.
        cohort: Adjacent, graph-first-observed, or raw-first-observed choice.
        classified: Full classified adjacent-year table with boolean cohort fields.
        common_ids: Optional exact ID mask for comparison across DF conditions.

    Returns:
        Selected feature rows, with their original labels and ordering.

    Raises:
        ValueError: If masks lack IDs, contain unknown IDs, or disagree with labels.
    """
    out = validate_labeled_fronts(frame, "forecast table")
    ids = pd.MultiIndex.from_frame(out[ID_COLUMNS])
    if common_ids is not None:
        common = canonical_front_ids(common_ids, "common-ID mask")
        requested = pd.MultiIndex.from_frame(common[ID_COLUMNS])
        if not requested.isin(ids).all():
            raise ValueError("common-ID mask contains IDs absent from frozen adjacent-year labels")
        out = out.loc[ids.isin(requested)].reset_index(drop=True)
        ids = pd.MultiIndex.from_frame(out[ID_COLUMNS])
    keep = np.ones(len(out), dtype=bool)
    if cohort != "adjacent":
        if cohort not in {"graph_first_observed", "raw_first_observed"}:
            raise ValueError(f"unknown cohort {cohort!r}")
        if classified is None:
            raise ValueError(f"{cohort} requires a classification table")
        classified = canonical_front_ids(classified, "classification")
        if cohort not in classified or classified[cohort].isna().any():
            raise ValueError(f"classification requires nonmissing {cohort} flags")
        if not classified[cohort].isin([True, False]).all():
            raise ValueError(f"classification {cohort} must be boolean")
        indexed = classified.set_index(ID_COLUMNS)
        if not ids.isin(indexed.index).all():
            raise ValueError("classification is missing frozen adjacent-year IDs")
        matched = indexed.loc[ids]
        if "label" in matched:
            if matched["label"].isna().any():
                raise ValueError("classification is missing frozen labels for matched IDs")
            if not np.array_equal(matched["label"].to_numpy(dtype=int), out["label"].to_numpy(dtype=int)):
                raise ValueError("classification disagrees with frozen outcome labels")
        keep &= matched[cohort].to_numpy(dtype=bool)
    return out.loc[keep].reset_index(drop=True)


def fit_prediction_groups(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    seed: int,
    settings: ForecasterSettings,
    condition: str,
    cohort: str,
    analysis: str,
) -> pd.DataFrame:
    """Fit every feature group on identical row IDs using the original core helper.

    Args:
        train: Selected training rows; native NaN features are preserved.
        test: Selected test rows, common to all groups.
        groups: Exact ordered model input columns.
        seed: Original model/embedding seed.
        settings: Unchanged model and random calibration recipe.
        condition: Input/DF condition name.
        cohort: History definition.
        analysis: ``all_rows``, ``complete_proximity``, or ``legacy_mixed``.

    Returns:
        Long predictions retaining every selected test row for every group.

    Raises:
        ValueError: If IDs, labels, or columns are invalid.
        RuntimeError: If the original helper cannot fit a selected group.
    """
    train = validate_labeled_fronts(train, "train")
    test = validate_labeled_fronts(test, "test")
    if set(train["t"]) & set(test["t"]) or train["t"].max() >= test["t"].min():
        raise ValueError("training years must precede the test year")
    results: list[pd.DataFrame] = []
    for group, columns in groups.items():
        missing = [column for column in columns if column not in train or column not in test]
        if missing:
            raise ValueError(f"{group}: missing feature columns {missing}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{group}: duplicate feature column names")
        fit = _fit_eval_lightgbm(train, test, columns, seed, settings)
        if fit.status != "ok":
            raise RuntimeError(f"{analysis}/{group}/seed {seed}: {fit.reason}")
        if not np.array_equal(fit.y_true, test[settings.label_column].to_numpy(dtype=int)):
            raise RuntimeError("core forecaster returned reordered or changed test labels")
        if len(fit.y_score) != len(test):
            raise RuntimeError("core forecaster omitted test predictions")
        prediction = test[[*ID_COLUMNS, "label", "prox_defined"]].copy()
        prediction["seed"] = seed
        prediction["group"] = group
        prediction["score"] = fit.y_score
        prediction["condition"] = condition
        prediction["cohort"] = cohort
        prediction["analysis"] = analysis
        results.append(prediction)
    return validate_predictions(pd.concat(results, ignore_index=True))


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Summarize AP per seed and paired deltas, keeping seed variability separate.

    Args:
        predictions: Exact per-front group predictions.

    Returns:
        Per-seed AP, mean paired deltas, counts, and proximity coverage.
    """
    predictions = validate_predictions(predictions)
    summaries: list[dict[str, Any]] = []
    for identity, cell in predictions.groupby(CELL_COLUMNS, sort=True):
        metrics: dict[str, dict[int, float]] = {}
        for (seed, group), block in cell.groupby(["seed", "group"], sort=True):
            metrics.setdefault(str(group), {})[int(seed)] = float(
                average_precision_score(block["label"], block["score"])
            )
        paired: dict[str, Any] = {}
        for baseline, group in _comparisons(set(metrics), None):
            by_seed = metrics[group]
            deltas = [value - metrics[baseline][seed] for seed, value in sorted(by_seed.items())]
            paired[f"{baseline} -> {group}"] = {
                "baseline": baseline, "augmented": group, "seeds": sorted(by_seed),
                "per_seed_delta_ap": deltas,
                "mean_delta_ap": float(np.mean(deltas)),
                "seed_delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            }
        first = next(iter(cell.groupby(["seed", "group"], sort=True)))[1]
        summaries.append({
            **dict(zip(CELL_COLUMNS, identity, strict=True)),
            "n_test": len(first), "n_positive": int(first["label"].sum()),
            "n_prox_defined": int(first["prox_defined"].sum()),
            "prox_defined_fraction": float(first["prox_defined"].mean()),
            "per_seed_ap": metrics, "paired_ap": paired,
        })
    return {"effect": "mean of per-seed paired AP increments", "seed_delta_sd_ddof": 1, "analyses": summaries}


def _integer_list(value: str) -> tuple[int, ...]:
    try:
        if ".." in value:
            start, stop = (int(part) for part in value.split(".."))
            result = tuple(range(start, stop + 1))
        else:
            result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use an inclusive range 2011..2015 or comma-separated integers"
        ) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer lists must be nonempty and contain no duplicates")
    return result


def _read_year(path: Path, year: int, labeled: bool = False) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "t" not in frame:
        frame["t"] = year
    if not frame["t"].eq(year).all():
        raise ValueError(f"{path}: contains rows outside year {year}")
    return validate_labeled_fronts(frame, str(path)) if labeled else canonical_front_ids(frame, str(path))


def _input_view(output_dir: Path, labels_dir: Path, terms_dir: Path) -> Path:
    view = output_dir / "provider_inputs"
    view.mkdir(exist_ok=True)
    for name, target in [("labels", labels_dir), ("terms", terms_dir)]:
        link = view / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"existing provider link points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"provider input path must be a new symlink: {link}")
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)
    return view


def _classifications(args: argparse.Namespace, year: int, config: Mapping[str, Any]) -> pd.DataFrame | None:
    if args.cohort == "adjacent":
        return None
    if args.cohort_dir is not None:
        path = args.cohort_dir / f"classified_{args.cohort_condition}_{year}.parquet"
        return _read_year(path, year)
    if args.cohort == "raw_first_observed":
        raise ValueError("raw_first_observed requires --cohort-dir with raw-MeSH classifications")
    from tsnd.cli import _fronts_born

    labels = _read_year(args.labels_dir / f"labels_{year}.parquet", year, labeled=True)
    first = _fronts_born(args.terms_dir, year, int(config["data"]["year_start"]))
    if first.empty:
        first = pd.DataFrame(columns=ID_COLUMNS)
    first = canonical_front_ids(first, "cumulative graph history")
    ids = pd.MultiIndex.from_frame(first[ID_COLUMNS])
    labels["graph_first_observed"] = pd.MultiIndex.from_frame(labels[ID_COLUMNS]).isin(ids)
    return labels


def run_forecast(args: argparse.Namespace) -> dict[str, Any]:
    """Run the explicit comparisons, exporting predictions without changing inputs.

    Args:
        args: Parsed CLI inputs with explicit artifact paths and frozen years.

    Returns:
        Summary written to the fresh output directory.

    Raises:
        ValueError: If paths, years, masks, or prediction alignment are invalid.
        FileExistsError: If prediction outputs already exist.
    """
    from tsnd.cli import _load_yaml, _provider

    if tuple(sorted(args.train_years)) != args.train_years or max(args.train_years) >= args.test_year:
        raise ValueError("train years must be strictly increasing and precede test year")
    if min(args.seeds) < 0:
        raise ValueError("seeds must be nonnegative")
    config = _load_yaml(str(args.config))
    config["seed_list"] = list(args.seeds)
    config["forecaster"]["seeds"] = list(args.seeds)
    config["forecaster"]["feature_groups"] = journal_feature_groups()
    settings = ForecasterSettings.from_config(config)
    if settings.label_column != "label":
        raise ValueError("journal runner expects the frozen binary label column named label")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("predictions_s*.parquet")) or (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to replace forecast outputs in {args.output_dir}")
    view = _input_view(args.output_dir, args.labels_dir, args.terms_dir)
    provider = _provider(
        config, args.terms_dir, args.embed_dir, args.decomp_dir, args.labels_dir, view, args.data_dir,
        ["b2", "b7"], args.train_years, args.test_year,
    )
    years = [*args.train_years, args.test_year]
    labels = {year: _read_year(args.labels_dir / f"labels_{year}.parquet", year, True) for year in years}
    classified = {year: _classifications(args, year, config) for year in years}
    common = {
        year: _read_year(args.common_ids_dir / f"year_{year}.parquet", year)
        if args.common_ids_dir is not None else None
        for year in years
    }
    condition = args.condition or args.cohort_condition
    legacy_only = bool(getattr(args, "legacy_only", False))
    legacy_enabled = bool(args.legacy_mixed or legacy_only)
    analyses = ["legacy_mixed"] if legacy_only else [
        "all_rows", "complete_proximity", *(["legacy_mixed"] if legacy_enabled else []),
    ]
    all_predictions: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    fit_cohorts: list[dict[str, Any]] = []
    signatures: dict[tuple[str, str], pd.DataFrame] = {}
    for seed in args.seeds:
        LOGGER.info("building original provider tables for seed %s", seed)
        raw_train, raw_test = provider(seed)
        selected: list[pd.DataFrame] = []
        for split, wide, metadata in [
            ("train", raw_train, pd.concat([labels[year] for year in args.train_years], ignore_index=True)),
            ("test", raw_test, labels[args.test_year]),
        ]:
            wide = attach_front_metadata(wide, metadata)
            parts = [
                select_fronts(block, args.cohort, classified[int(year)], common[int(year)])
                for year, block in wide.groupby("t", sort=False)
            ]
            chosen = add_degree_product(pd.concat(parts, ignore_index=True), args.terms_dir)
            coverage.append({
                "seed": seed, "split": split, "n_adjacent_labeled": len(wide),
                "n_after_history_and_common_ids": len(chosen),
                "n_pairs": int(chosen["kind"].eq("pair").sum()),
                "n_positive": int(chosen["label"].sum()),
                "n_prox_defined": int(chosen["prox_defined"].sum()),
            })
            selected.append(chosen)
        seed_predictions: list[pd.DataFrame] = []
        for analysis in analyses:
            tables: list[pd.DataFrame] = []
            for split, chosen in zip(["train", "test"], selected, strict=True):
                table = chosen.copy()
                if analysis != "legacy_mixed" and args.pair_only:
                    table = table.loc[table["kind"] == "pair"]
                if analysis == "complete_proximity":
                    table = table.loc[table["prox_defined"]]
                table = table.reset_index(drop=True)
                signature = table[[*ID_COLUMNS, "label", "prox_defined"]]
                previous = signatures.setdefault((analysis, split), signature)
                if not previous.equals(signature):
                    raise ValueError(f"{analysis}: IDs or proximity availability differ across seeds")
                fit_cohorts.append({
                    "seed": seed, "analysis": analysis, "split": split, "n_fronts": len(table),
                    "n_positive": int(table["label"].sum()),
                    "n_prox_defined": int(table["prox_defined"].sum()),
                })
                tables.append(table)
            groups = legacy_feature_groups() if analysis == "legacy_mixed" else journal_feature_groups()
            LOGGER.info("fitting seed=%s analysis=%s train=%s test=%s", seed, analysis, len(tables[0]),
                        len(tables[1]))
            seed_predictions.append(fit_prediction_groups(
                tables[0], tables[1], groups, seed, settings, condition, args.cohort, analysis,
            ))
        predictions = pd.concat(seed_predictions, ignore_index=True)
        predictions.to_parquet(args.output_dir / f"predictions_s{seed}.parquet", index=False)
        all_predictions.append(predictions)
    summary = summarize_predictions(pd.concat(all_predictions, ignore_index=True))
    summary.update({
        "status": "complete", "condition": condition, "cohort": args.cohort,
        "cohort_condition": args.cohort_condition,
        "train_years": list(args.train_years), "test_year": args.test_year, "seeds": list(args.seeds),
        "pair_only": args.pair_only, "legacy_mixed_control": legacy_enabled,
        "analysis_mode": "legacy_only" if legacy_only else "recombination",
        "groups": legacy_feature_groups() if legacy_only else journal_feature_groups(),
        "legacy_groups": legacy_feature_groups() if legacy_enabled else {},
        "lightgbm_params": settings.lightgbm_params, "calibration_fraction": settings.calibration_fraction,
        "min_calibration_samples": settings.min_calibration_samples, "coverage": coverage,
        "fit_cohorts": fit_cohorts,
        "inputs": {name: str(getattr(args, name)) for name in (
            "config", "data_dir", "terms_dir", "embed_dir", "decomp_dir", "labels_dir", "cohort_dir",
            "common_ids_dir",
        )},
        "uncertainty": "run joint --prediction-files mode across all conditions/cohorts once fits finish",
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Fit journal comparisons or jointly summarize saved prediction dependence.

    Args:
        argv: Optional explicit command-line arguments.

    Returns:
        Zero on successful completion.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "data-dir", "terms-dir", "embed-dir", "decomp-dir", "labels-dir"]:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=["adjacent", "graph_first_observed", "raw_first_observed"],
                        default="adjacent")
    parser.add_argument("--cohort-dir", type=Path)
    parser.add_argument("--cohort-condition", default="df20")
    parser.add_argument("--condition")
    parser.add_argument("--common-ids-dir", type=Path)
    parser.add_argument("--test-year", type=int, default=2016)
    parser.add_argument("--train-years", type=_integer_list, default=_integer_list("2011..2015"))
    parser.add_argument("--seeds", type=_integer_list, default=_integer_list("0..9"))
    parser.add_argument("--pair-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legacy-mixed", action="store_true")
    parser.add_argument("--legacy-only", action="store_true",
                        help="Fit only the original mixed-front comparison for cohort/corpus sensitivity.")
    parser.add_argument("--prediction-files", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--uncertainty-seed", type=int, default=20261017)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prediction_files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "dependence_sensitivity.json"
        if destination.exists():
            raise FileExistsError(f"refusing to replace {destination}")
        predictions = pd.concat([pd.read_parquet(path) for path in args.prediction_files], ignore_index=True)
        summary = node_multiplier_sensitivity(predictions, args.resamples, args.uncertainty_seed)
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        required = ["config", "data_dir", "terms_dir", "embed_dir", "decomp_dir", "labels_dir"]
        if any(getattr(args, name) is None for name in required):
            parser.error("fit mode requires explicit --config/--data-dir/--terms-dir/--embed-dir/"
                         "--decomp-dir/--labels-dir")
        run_forecast(args)
    return 0
