from __future__ import annotations

import pickle

import pandas as pd
import pytest

from tsnd.data.preprocessing import build_year
from tsnd.evaluation.labels import YearUsage, build_inverted_index, label_cohort
from tsnd.journal.fixed_outcomes import (
    assign_fixed_outcomes,
    build_fixed_reference_outcomes,
    common_membership,
)


def usage():
    units = [["a", "b"]] * 4 + [["c", "d"]] + [["x", "y"]] * 3
    return YearUsage(
        build_inverted_index(units),
        [f"v{i}" for i in range(8)],
        [[f"s{i}"] for i in range(8)],
        [[] for _ in range(8)],
    )


def test_new_view_candidates_use_the_full_reference_cutoff_without_rethresholding():
    full = pd.DataFrame({"t": [2011, 2011], "kind": ["pair", "pair"], "key": ["a|b", "c|d"]})
    reference = label_cohort(full, [usage()], ["venues", "subfields"], 0.5)
    assert reference.thr.tolist() == [4.0, 4.0]
    candidates = pd.concat([full, pd.DataFrame({"t": [2011], "kind": ["pair"], "key": ["x|y"]})])
    labeled = assign_fixed_outcomes(candidates, [usage()], ["venues", "subfields"], 4.0).set_index("key")
    assert labeled.loc["x|y", "breadth"] == 3.0
    assert labeled.loc["x|y", "label"] == 0
    assert labeled.loc["a|b", "label"] == 1
    # Re-estimating on this three-row view would incorrectly make x|y positive.
    rethresholded = label_cohort(candidates, [usage()], ["venues", "subfields"], 0.5).set_index("key")
    assert rethresholded.loc["x|y", "label"] == 1


def test_raw_first_common_masks_intersect_history_flags_not_only_candidate_ids():
    full = pd.DataFrame(
        {
            "t": [2016, 2016],
            "kind": ["pair", "pair"],
            "key": ["a|b", "c|d"],
            "label": [1, 0],
            "raw_first_observed": [False, True],
        }
    )
    capped = full.assign(raw_first_observed=[True, True])
    adjacent = common_membership({"full": full, "capped": capped}, "adjacent")
    raw_first = common_membership({"full": full, "capped": capped}, "raw_first_observed")
    assert adjacent.key.tolist() == ["a|b", "c|d"]
    assert raw_first.key.tolist() == ["c|d"]
    with pytest.raises(ValueError, match="conflicting"):
        common_membership({"full": full, "capped": capped.assign(label=[0, 0])}, "adjacent")


def test_full_reference_outcomes_include_training_windows_before_2017(tmp_path, monkeypatch):
    config = {
        "data": {"year_start": 2010},
        "terms": {
            "unit_source": "mesh",
            "min_df_per_year": 20,
            "top_terms": 50_000,
            "max_units_per_paper": 40,
            "unknown_df_rank": 1_000_000_000,
        },
        "label": {
            "horizon": 3,
            "breadth_channels": ["venues", "institutions", "subfields"],
            "positive_quantile": 0.9,
        },
    }
    data, full, capped = (tmp_path / name for name in ["data", "full_terms", "capped_terms"])
    for path in (data, full, capped):
        path.mkdir()
    for year in range(2010, 2015):
        units = [["a", "b"]] * 20 if year >= 2011 else [["a"]] * 20
        rows = pd.DataFrame(
            {"mesh": units, "venue_id": [f"v{year}"] * 20, "subfield_ids": [[f"s{year}"]] * 20}
        )
        rows.to_parquet(data / f"works_{year}.parquet", index=False)
        graph = build_year(rows, config)
        with (full / f"year_{year}.pkl").open("wb") as handle:
            pickle.dump(graph, handle)
        # A sampled feature view can lack all future usage; labels must ignore it.
        sampled_graph = graph if year <= 2011 else build_year(rows.iloc[:0], config)
        with (capped / f"year_{year}.pkl").open("wb") as handle:
            pickle.dump(sampled_graph, handle)
    import tsnd.journal.fixed_outcomes as module

    calls = []
    original = module.label_cohort

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "label_cohort", counted)
    result = build_fixed_reference_outcomes(
        config, data, full, {"full": full, "capped": capped}, tmp_path / "outcomes", years=[2011]
    )
    assert calls == [1]
    assert result["usage_years"] == [2012, 2013, 2014]
    assert result["cutoffs"][0]["outcome_years"] == [2012, 2013, 2014]
    observed_full = pd.read_parquet(tmp_path / "outcomes/views/full/labels_2011.parquet")
    observed_capped = pd.read_parquet(tmp_path / "outcomes/views/capped/labels_2011.parquet")
    pd.testing.assert_frame_equal(observed_full, observed_capped)
    assert observed_capped.loc[observed_capped.key.eq("a|b"), "breadth"].iloc[0] == 3.0
    assert observed_capped["label"].eq(1).all()
    assert not observed_capped["label"].isna().any()
