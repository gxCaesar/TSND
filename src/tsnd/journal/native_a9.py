"""Native raw-to-profile future-input audit in fresh, isolated artifact trees."""
from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from tsnd.data.preprocessing import build_year, build_years, configured_years
from tsnd.data.preprocessing import default_config as preprocessing_defaults
from tsnd.models.alignment import build_boundary, save_boundary
from tsnd.models.decomposition import RecombinationResult
from tsnd.models.embeddings import YearlySgnsEmbedder
from tsnd.models.ours import compute_decomposition, load_embeddings, load_graph

SENTINELS = ("__a9_future_sentinel_alpha__", "__a9_future_sentinel_beta__")
ATOL = 1e-9
RTOL = 1e-7


def compare_graphs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare production graph content directly, including order and edge counts."""
    fields = ("vocab", "uidx", "unit_df", "paper_units", "edges", "n_papers")
    equal = {name: bool(left[name] == right[name]) for name in fields}
    return {"passed": all(equal.values()), "field_equal": equal,
            "left_counts": {"vocab": len(left["vocab"]), "edges": len(left["edges"]),
                            "papers": int(left["n_papers"])},
            "right_counts": {"vocab": len(right["vocab"]), "edges": len(right["edges"]),
                             "papers": int(right["n_papers"])}}


def compare_numeric(left: Any, right: Any, *, atol: float = ATOL, rtol: float = RTOL) -> dict[str, Any]:
    """Report numeric agreement and maximum absolute error by direct numerical comparison."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    same_shape = a.shape == b.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    maximum = float(np.max(np.abs(a - b), initial=0.0)) if same_shape and finite else None
    return {"passed": bool(same_shape and finite and np.allclose(a, b, atol=atol, rtol=rtol)),
            "max_abs_error": maximum, "same_shape": same_shape, "finite": finite,
            "left_shape": list(a.shape), "right_shape": list(b.shape)}


def _compare_mapping(left: Mapping[Any, Any], right: Mapping[Any, Any]) -> dict[str, Any]:
    same_keys = left.keys() == right.keys()
    shared = sorted(left.keys() & right.keys(), key=str)
    result = compare_numeric([left[key] for key in shared], [right[key] for key in shared])
    result.update({"same_keys": same_keys, "left_count": len(left), "right_count": len(right)})
    result["passed"] = bool(result["passed"] and same_keys)
    return result


def compare_profiles(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare all native component mappings, residual fits, and bookkeeping."""
    checks = {name: _compare_mapping(left[name], right[name]) for name in ("LE", "NS", "deg")}
    rc_a = cast(RecombinationResult, left["rc"])
    rc_b = cast(RecombinationResult, right["rc"])
    checks.update({"RC": _compare_mapping(rc_a.rc, rc_b.rc),
                   "prox": _compare_mapping(rc_a.prox, rc_b.prox),
                   "coef": compare_numeric(rc_a.coef, rc_b.coef),
                   "bias": compare_numeric(rc_a.bias, rc_b.bias)})
    exact = {name: left[name] == right[name] for name in ("n_new_terms", "n_new_pairs")}
    exact["n_defined"] = rc_a.n_defined == rc_b.n_defined
    return {"passed": all(c["passed"] for c in checks.values()) and all(exact.values()),
            "numeric": checks, "exact_counts_equal": exact}


def _link_input(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def _make_views(
    config: Mapping[str, Any], data_dir: Path, output_dir: Path, year: int,
) -> tuple[list[int], list[int], pd.DataFrame]:
    bounded_config = {**config, "data": {**config["data"], "year_end": year}}
    allowed = configured_years(bounded_config)
    future = list(range(year + 1, year + 4))
    for observed_year in [*allowed, *future]:
        source = data_dir / f"works_{observed_year}.parquet"
        dates = pd.read_parquet(source, columns=["year"])["year"]
        if not dates.eq(observed_year).all():
            raise ValueError(f"Raw publication years disagree with filename: {source}")
        _link_input(source, output_dir / "with_future" / "raw" / source.name)
        if observed_year <= year:
            _link_input(source, output_dir / "prior_only" / "raw" / source.name)
    n_records = max(20, int(config["terms"]["min_df_per_year"]))
    sentinel_year = future[-1] + 1
    planted = pd.DataFrame({
        "id": [f"__a9_checker_future_{i}" for i in range(n_records)],
        "year": [sentinel_year] * n_records,
        "abstract": ["Synthetic future input used only to test the audit checker."] * n_records,
        "mesh": [list(SENTINELS) for _ in range(n_records)],
    })
    planted.to_parquet(output_dir / "with_future" / "raw" / f"works_{sentinel_year}.parquet", index=False)
    return allowed, [*future, sentinel_year], planted


def _run_view(
    config: Mapping[str, Any], root: Path, allowed: Sequence[int], year: int, seeds: Sequence[int],
) -> None:
    terms = root / "terms"
    preprocess_config = {**config, "terms": {**preprocessing_defaults()["terms"], **config["terms"]}}
    written = build_years(preprocess_config, allowed, root / "raw", terms)
    if len(written) != len(allowed):
        raise RuntimeError("Preprocessing skipped a required historical input year.")
    embedder = YearlySgnsEmbedder(config, terms, root / "embed", workers=1)
    align = config["align"]
    sweep = [float(tau) for tau in align["tau_sweep"]]
    tau = float(align.get("tau_main", sorted(sweep)[len(sweep) // 2]))
    (root / "decompose").mkdir()
    for seed in seeds:
        for target_year in (year - 1, year):
            units, matrix, _ = embedder.train_year(target_year, seed)
            embedder.save_year(target_year, seed, units, matrix)
        boundary = build_boundary(
            root / "embed", terms, year, seed, k=int(align["k_neighbors"]),
            tau_main=tau, tau_sweep=sweep, min_anchors=int(align["min_anchors"]),
            method="procrustes", anchor_selection=str(align.get("anchor_selection", "graph_stability")),
            freq_top_k=int(align.get("frequency_anchor_top_k", 5000)),
        )
        if boundary is None:
            raise RuntimeError(f"Native A9 cannot estimate alignment for year {year}, seed {seed}.")
        save_boundary(root / "align", boundary)
        profile = compute_decomposition(year, seed, root / "embed", terms, root / "align")
        with (root / "decompose" / f"decomp_{year}_s{seed}.pkl").open("wb") as handle:
            pickle.dump(profile, handle, protocol=4)


def _compare_views(
    output_dir: Path, allowed: Sequence[int], year: int, seeds: Sequence[int],
) -> dict[str, Any]:
    left, right = output_dir / "prior_only", output_dir / "with_future"
    graphs = {str(y): compare_graphs(load_graph(left / "terms", y), load_graph(right / "terms", y))
              for y in allowed}
    per_seed = []
    for seed in seeds:
        embeddings = {}
        for target in (year - 1, year):
            units_a, matrix_a, _ = load_embeddings(left / "embed", target, seed)
            units_b, matrix_b, _ = load_embeddings(right / "embed", target, seed)
            check = compare_numeric(matrix_a, matrix_b)
            check["units_equal"] = units_a == units_b
            check["passed"] = bool(check["passed"] and check["units_equal"])
            embeddings[str(target)] = check
        profiles = []
        rotations = []
        for root in (left, right):
            with (root / "decompose" / f"decomp_{year}_s{seed}.pkl").open("rb") as handle:
                profiles.append(pickle.load(handle))  # noqa: S301 - newly produced local artifact
            with (root / "align" / f"bnd_{year}_s{seed}.pkl").open("rb") as handle:
                rotations.append(pickle.load(handle)["R"])  # noqa: S301 - newly produced local artifact
        profile_check = compare_profiles(profiles[0], profiles[1])
        rotation_check = compare_numeric(rotations[0], rotations[1])
        per_seed.append({"seed": seed, "embeddings": embeddings, "profile": profile_check,
                         "rotation": rotation_check,
                         "passed": (all(c["passed"] for c in embeddings.values())
                                    and profile_check["passed"] and rotation_check["passed"])})
    return {"passed": all(c["passed"] for c in graphs.values()) and all(c["passed"] for c in per_seed),
            "graphs": graphs, "per_seed": per_seed}


def run_native_a9(
    config: Mapping[str, Any], data_dir: Path, output_dir: Path, *, year: int = 2016,
    seeds: Sequence[int] = tuple(range(10)),
) -> dict[str, Any]:
    """Rebuild bounded native profiles despite extra future raw files, and test a defect.

    The audit concerns dated records in this fixed snapshot. It does not recover
    historical metadata versions or establish event-time historical observability.
    SGNS workers=1 is an explicit deterministic audit setting, not a new method.
    """
    launch_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if not launch_hash_seed.isdigit():
        raise ValueError("Launch Python with a fixed numeric PYTHONHASHSEED, for example PYTHONHASHSEED=0.")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be a nonempty sequence of distinct nonnegative integers.")
    if year <= int(config["data"]["year_start"]):
        raise ValueError("The focal year needs at least one preceding year.")
    if config["terms"].get("unit_source") != "mesh":
        raise ValueError("This native raw audit requires the configured MeSH unit path.")
    if str(config["align"].get("method", "procrustes")).lower() != "procrustes":
        raise ValueError("This audit preserves the original Procrustes alignment method.")
    output_dir.mkdir(parents=True, exist_ok=False)
    allowed, exposed_future, planted = _make_views(config, data_dir, output_dir, year)
    for view in ("prior_only", "with_future"):
        _run_view(config, output_dir / view, allowed, year, seeds)
    invariant = _compare_views(output_dir, allowed, year, seeds)
    baseline = load_graph(output_dir / "prior_only" / "terms", year)
    focal_raw = pd.read_parquet(data_dir / f"works_{year}.parquet")
    pooled = pd.concat([focal_raw, planted], ignore_index=True)
    defective = build_year(pooled, config)  # Deliberate date-boundary defect, solely a checker fixture.
    control_dir = output_dir / "positive_control"
    control_dir.mkdir()
    with (control_dir / f"unbounded_graph_{year}.pkl").open("wb") as handle:
        pickle.dump(defective, handle, protocol=4)
    graph_check = compare_graphs(baseline, defective)
    control = {"purpose": "known-defect checker fixture, not a scientific ablation",
               "defect": "future raw records pooled into the focal-year graph without a date boundary",
               "graph_comparison": graph_check, "n_planted_distinct_records": len(planted),
               "n_added_papers": defective["n_papers"] - baseline["n_papers"],
               "sentinels_absent_from_bounded_graph": all(
                   unit not in baseline["vocab"] for unit in SENTINELS
               ),
               "sentinels_present_in_defective_graph": all(unit in defective["vocab"] for unit in SENTINELS)}
    control["detected"] = (not graph_check["passed"] and control["sentinels_absent_from_bounded_graph"]
                           and control["sentinels_present_in_defective_graph"]
                           and control["n_added_papers"] == len(planted))
    result = {"passed": bool(invariant["passed"] and control["detected"]), "year": year,
              "seeds": list(seeds), "workers": 1, "launch_pythonhashseed": launch_hash_seed,
              "atol": ATOL, "rtol": RTOL, "processed_raw_years": allowed,
              "future_years_exposed_only_to_second_view": exposed_future,
              "scope": "native raw-to-profile future-file invariance within this fixed snapshot",
              "historical_observability_established": False,
              "invariance": invariant, "positive_control": control}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "scientific_parameters.json").write_text(json.dumps(
        {key: config[key] for key in ("data", "terms", "embed", "align")}, indent=2
    ) + "\n")
    return result
