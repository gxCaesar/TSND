from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tsnd.data.preprocessing import build_year
from tsnd.journal.cohort import classify_cohort, raw_mesh_graph, summarize_cohort


def graph(rows, min_df=1, cap=40):
    return build_year(pd.DataFrame({"mesh": rows}), {"terms": {
        "unit_source": "mesh", "min_df_per_year": min_df, "top_terms": 50_000,
        "max_units_per_paper": cap, "unknown_df_rank": 1_000_000_000,
    }})


def test_raw_graph_keeps_rare_descriptors_and_more_than_40_units():
    units = [f"unit_{i:02}" for i in range(45)]
    raw = raw_mesh_graph(pd.DataFrame({"mesh": [units + ["humans", "male"]]}))
    assert len(raw["vocab"]) == 45
    assert len(raw["paper_units"][0]) == 45
    assert len(raw["edges"]) == 45 * 44 // 2
    assert "humans" not in raw["vocab"]


def test_first_observed_and_exact_reentry_gap():
    rows = {
        2010: [["long_a", "long_b"]],
        2011: [["gap_a", "gap_b"]],
        2012: [["old_a", "old_b"]],
        2013: [["long_a", "long_b"], ["gap_a", "gap_b"], ["new_a", "new_b"], ["old_a", "old_b"]],
    }
    graphs = {year: graph(units) for year, units in rows.items()}
    out = classify_cohort(graphs, graphs, 2013, start_year=2010, min_df=1).set_index("key")
    assert out.loc["new_a|new_b", "raw_first_observed"]
    assert out.loc["new_a|new_b", "graph_first_observed"]
    assert out.loc["gap_a|gap_b", "raw_class"] == "reentry_gap_1"
    assert out.loc["gap_a|gap_b", "raw_last_prior_year"] == 2011
    assert out.loc["gap_a|gap_b", "raw_gap_years"] == 1
    assert out.loc["long_a|long_b", "raw_class"] == "reentry_gap_ge2"
    assert out.loc["long_a|long_b", "raw_gap_years"] == 2
    assert "old_a|old_b" not in out.index
    assert not out.loc["gap_a|gap_b", "raw_first_observed"]


def test_threshold_induced_entry_does_not_become_raw_emergence():
    rows = {2010: [["a", "b"], ["a"]], 2011: [["a", "b"], ["a", "b"]]}
    raw = {y: raw_mesh_graph(pd.DataFrame({"mesh": units})) for y, units in rows.items()}
    filtered = {y: graph(units, min_df=2) for y, units in rows.items()}
    labels = pd.DataFrame({"kind": ["pair"], "key": ["b|a"], "t": [2011],
                           "label": [1], "breadth": [6.25], "thr": [4.5]})
    out = classify_cohort(raw, filtered, 2011, labels, min_df=2).set_index("key")
    pair = out.loc["a|b"]
    assert pair.raw_class == "present_previous"
    assert pair.graph_class == "first_observed"
    assert pair.graph_first_with_raw_history
    assert not pair.raw_first_observed
    assert pair.previous_graph_absence_cause == "df_threshold"
    assert pair.label == 1 and pair.breadth == 6.25 and pair.thr == 4.5
    assert not out.loc["b", "label_available"]
    assert pd.isna(out.loc["b", "label"])
    pd.testing.assert_frame_equal(labels, pd.DataFrame({
        "kind": ["pair"], "key": ["b|a"], "t": [2011],
        "label": [1], "breadth": [6.25], "thr": [4.5],
    }))


def test_per_paper_40_unit_cap_has_a_distinct_diagnostic():
    units = [f"unit_{i:02}" for i in range(41)]
    rows = {2010: [units, ["unit_40"]], 2011: [["unit_00", "unit_40"], []]}
    raw = {y: raw_mesh_graph(pd.DataFrame({"mesh": value})) for y, value in rows.items()}
    filtered = {y: graph(value) for y, value in rows.items()}
    out = classify_cohort(raw, filtered, 2011, min_df=1).set_index("key")
    pair = out.loc["unit_00|unit_40"]
    assert pair.raw_class == "present_previous"
    assert pair.graph_first_observed
    assert pair.previous_graph_absence_cause == "paper_unit_cap"
    assert pair.hidden_raw_history_causes == "paper_unit_cap"


def test_incompatible_raw_and_graph_inputs_fail():
    raw = {2010: graph([["a", "b"]]), 2011: graph([["a", "b"]])}
    mismatched = {2010: graph([["a", "b"]]), 2011: graph([["x", "y"]])}
    with pytest.raises(ValueError, match="absent from raw"):
        classify_cohort(raw, mismatched, 2011, min_df=1)


def test_empty_candidate_cohort_preserves_schema_and_summary():
    graphs = {2010: graph([["a", "b"]]), 2011: graph([["a", "b"]])}
    labels = pd.DataFrame({"kind": ["pair"], "key": ["a|b"], "label": [1]})
    empty = classify_cohort(graphs, graphs, 2011, labels, min_df=1)
    assert empty.empty
    assert {"kind", "key", "t", "raw_first_observed", "graph_first_observed", "label"} <= set(empty)
    assert all(row["n_candidates"] == 0 for row in summarize_cohort(empty, empty, "df20", 2011))


def test_summary_counts_exits_as_well_as_new_unlabelled_candidates():
    history = {2010: graph([[]]), 2011: graph([["a", "b"]])}
    baseline = classify_cohort(history, history, 2011, min_df=1)
    labels = pd.DataFrame({"kind": ["pair"], "key": ["a|b"], "label": [1]})
    other_history = {2010: graph([[]]), 2011: graph([["a", "c"]])}
    other = classify_cohort(other_history, other_history, 2011, labels, min_df=1)
    row = next(r for r in summarize_cohort(other, baseline, "df10", 2011) if r["kind"] == "pair")
    assert row["n_entries_vs_df20"] == row["n_exits_vs_df20"] == 1
    assert row["n_shared_with_df20"] == 0
    assert row["n_label_missing"] == 1


def test_cohort_cli_writes_all_candidate_labels_and_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    terms = tmp_path / "terms"
    labels = tmp_path / "labels"
    for path in (data, terms, labels):
        path.mkdir()
    for year, units in {2010: [["a"], []], 2011: [["a", "b"], ["a", "b"]]}.items():
        pd.DataFrame({"mesh": units}).to_parquet(data / f"works_{year}.parquet", index=False)
        with (terms / f"year_{year}.pkl").open("wb") as handle:
            pickle.dump(graph(units, min_df=2), handle)
    original = pd.DataFrame({"kind": ["pair"], "key": ["a|b"], "t": [2011],
                             "label": [1], "breadth": [8.0], "thr": [7.0]})
    original.to_parquet(labels / "labels_2011.parquet", index=False)
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    command = [sys.executable, str(root / "scripts/journal_cohort_sensitivity.py"),
               "--data-dir", str(data), "--terms-dir", str(terms), "--labels-dir", str(labels),
               "--output-dir", str(tmp_path / "output"), "--years", "2011", "--min-df", "df20=2"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    output = pd.read_parquet(tmp_path / "output/classified_df20_2011.parquet")
    assert len(output) == 3  # two terms plus their pair, including unlabelled terms
    assert output["label_available"].sum() == 1
    assert output.loc[output["kind"].eq("pair"), "label"].iloc[0] == 1
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert len(summary["rows"]) == 2
    pd.testing.assert_frame_equal(original, pd.read_parquet(labels / "labels_2011.parquet"))
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "already exist" in repeated.stderr
