from __future__ import annotations

import argparse
import itertools
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from tsnd.journal import forecast
from tsnd.models.decomposition import RecombinationResult, weighted_degree
from tsnd.training.forecaster import ForecasterSettings, GroupFitResult, _fit_eval_lightgbm


def settings_config():
    return {
        "seed_list": [0], "data": {"year_start": 2010},
        "forecaster": {
            "feature_groups": forecast.journal_feature_groups(), "label_column": "label",
            "calibration_fraction": .2, "min_calibration_samples": 2, "min_cohort_samples": 2,
            "bootstrap_samples": 5, "ci_alpha": .05, "ece_bins": 5, "precision_k": 2,
            "use_survival_model": False,
            "lightgbm_params": {
                "n_estimators": 5, "num_leaves": 3, "min_child_samples": 1, "verbosity": -1, "n_jobs": 1,
            },
        },
    }


def feature_fixture(year=2016):
    frame = pd.DataFrame({
        "t": year, "kind": "pair", "key": ["a|b", "a|c", "b|c", "c|d"],
        "label": [1, 0, 1, 0], "prox": [.3, np.nan, .8, .1], "rc": [.1, np.nan, -.2, .5],
        "z": [1., 2., 3., 1.], "prox_defined": [True, False, True, True],
    })
    for column in forecast.journal_feature_groups()["S"]:
        frame[column] = 1. if column == "kind_pair" else np.arange(4, dtype=float)
    return frame


def test_explicit_structure_and_historical_registry_order_are_preserved():
    groups = forecast.journal_feature_groups()
    assert list(groups) == ["S", "S+prox", "S+z", "S+rc", "S+prox+z", "S+prox+rc", "S+prox+z+rc"]
    assert groups["S"][:3] == ("kind_pair", "n_papers", "degree_min")
    assert not set(["deg", "prox", "rc", "le", "z"]) & set(groups["S"])
    assert groups["S+prox+z+rc"] == (*groups["S"], "prox", "z", "rc")
    legacy = forecast.legacy_feature_groups()
    assert legacy["legacy_S"] == groups["S"]
    assert legacy["legacy_mixed"][:6] == ("kind_pair", "le", "rc", "prox", "deg", "n_papers")
    assert "deg" in legacy["legacy_mixed"] and "degree_min" in legacy["legacy_mixed"]


def test_classification_and_common_ids_preserve_frozen_labels_and_nan_rows():
    frame = feature_fixture()
    classified = frame[["t", "kind", "key", "label"]].copy()
    classified["raw_first_observed"] = [False, True, True, False]
    common = frame.loc[[1, 2], ["t", "kind", "key"]]
    chosen = forecast.select_fronts(frame, "raw_first_observed", classified, common)
    assert chosen["key"].tolist() == ["a|c", "b|c"]
    assert chosen["label"].tolist() == [0, 1]
    assert chosen["prox"].isna().sum() == 1
    assert len(forecast.select_fronts(frame, "adjacent")) == 4


def test_df_common_mask_precedes_condition_specific_history_validation():
    frame = feature_fixture()
    common = frame.loc[[1, 2], ["t", "kind", "key"]]
    classified = frame.loc[[1, 2], ["t", "kind", "key", "label"]].copy()
    classified["raw_first_observed"] = [True, False]
    chosen = forecast.select_fronts(frame, "raw_first_observed", classified, common)
    assert chosen["key"].tolist() == ["a|c"]
    assert chosen["label"].tolist() == [0]
    assert chosen["prox"].isna().all()


@pytest.mark.parametrize("defect", ["missing_classification_id", "changed_label", "unknown_common_id"])
def test_mask_alignment_rejects_consequential_defects(defect):
    frame = feature_fixture()
    classified = frame[["t", "kind", "key", "label"]].copy()
    classified["raw_first_observed"] = True
    common = None
    if defect == "missing_classification_id":
        classified = classified.iloc[1:]
    elif defect == "changed_label":
        classified.loc[0, "label"] = 0
    else:
        common = pd.DataFrame({"t": [2016], "kind": ["pair"], "key": ["x|y"]})
    with pytest.raises(ValueError):
        forecast.select_fronts(frame, "raw_first_observed", classified, common)


def test_metadata_attachment_rejects_reordered_and_duplicate_ids():
    frame = feature_fixture()
    wide = frame.drop(columns=["t", "kind", "prox_defined"])
    labels = frame[["t", "kind", "key", "label"]]
    attached = forecast.attach_front_metadata(wide, labels)
    assert len(attached) == 4
    assert attached["prox_defined"].tolist() == [True, False, True, True]
    with pytest.raises(ValueError, match="do not match"):
        forecast.attach_front_metadata(wide.iloc[::-1], labels)
    with pytest.raises(ValueError, match="duplicate"):
        forecast.attach_front_metadata(wide, labels.assign(key=["a|b"] * 4))


def test_all_feature_groups_fit_identical_ids_and_keep_native_nan(monkeypatch):
    settings = ForecasterSettings.from_config(settings_config())
    train, test = feature_fixture(2015), feature_fixture(2016)
    calls = []

    def fitted(train_rows, test_rows, columns, seed, actual_settings):
        calls.append((train_rows["key"].tolist(), test_rows["key"].tolist(), columns))
        assert actual_settings is settings and seed == 0
        assert test_rows["prox"].isna().sum() == 1
        return GroupFitResult("ok", None, test_rows["label"].to_numpy(), np.array([.7, .2, .6, .1]), {})

    monkeypatch.setattr(forecast, "_fit_eval_lightgbm", fitted)
    predictions = forecast.fit_prediction_groups(
        train, test, forecast.journal_feature_groups(), 0, settings, "df20", "adjacent", "all_rows",
    )
    assert len(calls) == 7
    assert len(predictions) == 7 * len(test)
    assert all(train_ids == test_ids == test["key"].tolist() for train_ids, test_ids, _ in calls)
    assert predictions.groupby("group")["prox_defined"].sum().eq(3).all()


def test_real_lightgbm_predictions_match_core_recipe():
    settings = ForecasterSettings.from_config(settings_config())
    train, test = feature_fixture(2015), feature_fixture(2016)
    groups = {name: forecast.journal_feature_groups()[name] for name in ["S", "S+prox"]}
    predictions = forecast.fit_prediction_groups(
        train, test, groups, 0, settings, "df20", "adjacent", "all_rows",
    )
    for name, columns in groups.items():
        direct = _fit_eval_lightgbm(train, test, columns, 0, settings)
        np.testing.assert_array_equal(predictions.loc[predictions["group"] == name, "score"], direct.y_score)


@pytest.mark.parametrize("legacy_only", [False, True])
@pytest.mark.parametrize("masked_missing_endpoint", [False, True])
def test_actual_runner_exports_pair_and_legacy_controls(tmp_path, legacy_only, masked_missing_endpoint):
    directories = {name: tmp_path / name for name in ["terms", "labels", "decomp", "embed", "data"]}
    for directory in directories.values():
        directory.mkdir()
    units = [f"u{i}" for i in range(8)]
    pairs = list(itertools.combinations(units, 2))
    for year in [2010, 2011, 2012]:
        selected = [] if year == 2010 else pairs[(year - 2011) * 12:(year - 2010) * 12]
        vocab = [*units, *([f"term{year}"] if year > 2010 else [])]
        index = {unit: number for number, unit in enumerate(vocab)}
        graph = {
            "vocab": vocab, "uidx": index, "unit_df": {unit: 2 for unit in vocab},
            "edges": {(index[u], index[v]): 2 for u, v in selected},
            "paper_units": [list(pair) for pair in selected] + [[f"term{year}"]],
            "n_papers": len(selected) + 1,
        }
        with (directories["terms"] / f"year_{year}.pkl").open("wb") as handle:
            pickle.dump(graph, handle)
        if year == 2010:
            continue
        labels = pd.DataFrame({
            "kind": ["pair"] * 12 + ["term"],
            "key": ["|".join(pair) for pair in selected] + [f"term{year}"],
            "label": [i % 2 for i in range(13)],
        })
        if masked_missing_endpoint:
            mask_dir = tmp_path / "common"
            mask_dir.mkdir(exist_ok=True)
            labels.assign(t=year)[["t", "kind", "key"]].to_parquet(
                mask_dir / f"year_{year}.parquet", index=False,
            )
            labels = pd.concat([
                labels, pd.DataFrame({"kind": ["pair"], "key": ["missing|u0"], "label": [0]}),
            ], ignore_index=True)
        labels.to_parquet(directories["labels"] / f"labels_{year}.parquet", index=False)
        proximity = {pair: (float(i) / 12 if i > 1 else np.nan) for i, pair in enumerate(selected)}
        decomposition = {
            "LE": {f"term{year}": .5}, "NS": {}, "deg": weighted_degree(graph["edges"], vocab),
            "rc": RecombinationResult(
                {pair: 1 - value for pair, value in proximity.items()}, proximity, (0., -1.), 1., 10,
            ),
        }
        with (directories["decomp"] / f"decomp_{year}_s0.pkl").open("wb") as handle:
            pickle.dump(decomposition, handle)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings_config()), encoding="utf-8")
    output = tmp_path / "output"
    args = argparse.Namespace(
        config=config_path, data_dir=directories["data"], terms_dir=directories["terms"],
        embed_dir=directories["embed"], decomp_dir=directories["decomp"], labels_dir=directories["labels"],
        output_dir=output, cohort="adjacent", cohort_dir=None, cohort_condition="df20", condition="toy",
        common_ids_dir=tmp_path / "common" if masked_missing_endpoint else None,
        test_year=2012, train_years=(2011,), seeds=(0,),
        pair_only=True, legacy_mixed=True, legacy_only=legacy_only,
    )
    summary = forecast.run_forecast(args)
    assert summary["status"] == "complete"
    rows = pd.read_parquet(output / "predictions_s0.parquet")
    expected = {"legacy_mixed": 13 * 2}
    if not legacy_only:
        expected.update({"all_rows": 12 * 7, "complete_proximity": 10 * 7})
        assert rows.loc[rows["analysis"] == "all_rows", "prox_defined"].eq(False).sum() == 2 * 7
    assert rows.groupby("analysis").size().to_dict() == expected
    assert rows.loc[rows["analysis"] == "legacy_mixed", "kind"].eq("term").sum() == 2
    assert json.loads((output / "summary.json").read_text())["pair_only"] is True
    with pytest.raises(FileExistsError):
        forecast.run_forecast(args)
