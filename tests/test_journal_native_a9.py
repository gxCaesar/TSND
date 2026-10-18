from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tsnd.data.preprocessing import build_year
from tsnd.journal.native_a9 import compare_graphs, compare_numeric, compare_profiles, run_native_a9
from tsnd.models.decomposition import RecombinationResult


def tiny_config():
    return {"data": {"year_start": 2010, "year_end": 2019},
            "terms": {"unit_source": "mesh", "min_df_per_year": 1, "top_terms": 1000,
                      "max_units_per_paper": 40, "unknown_df_rank": 1_000_000_000},
            "embed": {"sgns_dim": 4, "window": 2, "min_count": 1, "epochs": 2,
                      "train_window_years": 3},
            "align": {"k_neighbors": 3, "tau_sweep": [0.0], "min_anchors": 2,
                      "anchor_selection": "graph_stability"}}


def test_graph_comparison_detects_a_future_pooling_defect():
    config = tiny_config()
    bounded = build_year(pd.DataFrame({"mesh": [["a", "b"]] * 3}), config)
    defective = build_year(pd.DataFrame({"mesh": [["a", "b"]] * 3 + [["future_a", "future_b"]] * 20}), config)
    assert compare_graphs(bounded, bounded)["passed"]
    check = compare_graphs(bounded, defective)
    assert not check["passed"]
    assert not check["field_equal"]["vocab"]
    assert check["right_counts"]["papers"] - check["left_counts"]["papers"] == 20


def test_numeric_comparator_reports_error_and_rejects_nonfinite_or_shape_changes():
    assert compare_numeric([1.0], [1.0 + 5e-8])["passed"]
    failed = compare_numeric([1.0], [1.001])
    assert not failed["passed"]
    assert failed["max_abs_error"] == pytest.approx(0.001)
    assert not compare_numeric([1.0], [[1.0]])["passed"]
    assert not compare_numeric([np.nan], [np.nan])["passed"]
    assert compare_numeric([], [])["max_abs_error"] == 0.0


def test_profile_comparison_includes_proximity_keys_coefficients_and_counts():
    left = {"LE": {"new": 0.3}, "NS": {"a": 0.1}, "deg": {"a": 2},
            "rc": RecombinationResult({("a", "b"): 0.2}, {("a", "b"): 0.6}, (0.1, -1.0), 1.0, 1),
            "n_new_terms": 1, "n_new_pairs": 1}
    assert compare_profiles(left, left)["passed"]
    right = {**left, "rc": RecombinationResult({("a", "c"): 0.2}, {("a", "b"): 0.7},
                                               (0.2, -1.0), 1.0, 2)}
    check = compare_profiles(left, right)
    assert not check["passed"]
    assert not check["numeric"]["RC"]["same_keys"]
    assert not check["numeric"]["prox"]["passed"]
    assert not check["numeric"]["coef"]["passed"]
    assert not check["exact_counts_equal"]["n_defined"]


def test_audit_requires_fixed_launch_hashseed_and_fresh_output(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        run_native_a9(tiny_config(), tmp_path / "data", tmp_path / "out", seeds=[0])
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run_native_a9(tiny_config(), tmp_path / "data", tmp_path / "out", seeds=[0])


def test_native_raw_to_embedding_audit_and_known_defect_in_a_fixed_hash_subprocess(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "raw"
    data.mkdir()
    for year in range(2010, 2020):
        units = [["a", "b"]] * 20 + [["b", "c"]] * 20
        if year >= 2016:
            units += [["a", "c"]] * 20 + [["a", "new"]] * 20
        frame = pd.DataFrame({"id": [f"W{year}_{i}" for i in range(len(units))],
                              "year": [year] * len(units), "mesh": units})
        frame.to_parquet(data / f"works_{year}.parquet", index=False)
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(tiny_config()))
    out = tmp_path / "audit"
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(root / "src"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, str(root / "scripts/journal_native_a9.py"),
               "--config", str(config), "--data-dir", str(data), "--output-dir", str(out),
               "--year", "2016", "--seeds", "0", "1"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert (out / "summary.json").is_file() and not (out / "summary.json").is_symlink()
    assert summary["passed"] and summary["invariance"]["passed"]
    assert summary["processed_raw_years"] == list(range(2010, 2017))
    assert summary["future_years_exposed_only_to_second_view"] == [2017, 2018, 2019, 2020]
    assert summary["launch_pythonhashseed"] == "0" and summary["workers"] == 1
    assert not summary["historical_observability_established"]
    assert summary["positive_control"]["detected"]
    assert summary["positive_control"]["sentinels_present_in_defective_graph"]
    assert summary["positive_control"]["n_planted_distinct_records"] >= 20
    assert not summary["positive_control"]["graph_comparison"]["passed"]
    assert len(list((out / "prior_only" / "embed").glob("*.npz"))) == 4
    assert not list((out / "with_future" / "terms").glob("year_201[789].pkl"))
    for per_seed in summary["invariance"]["per_seed"]:
        assert per_seed["passed"]
        assert per_seed["embeddings"]["2016"]["max_abs_error"] == 0.0
    with (out / "prior_only" / "decompose" / "decomp_2016_s0.pkl").open("rb") as handle:
        profile = pickle.load(handle)
    assert profile["LE"] and profile["NS"] and profile["rc"].rc
    # No fixture row or additional work is ever appended to a canonical source parquet.
    assert len(pd.read_parquet(data / "works_2016.parquet")) == 80
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
