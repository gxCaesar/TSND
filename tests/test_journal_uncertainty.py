from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from tsnd.journal.uncertainty import (
    FixedScoreAveragePrecision,
    canonical_front_ids,
    node_multiplier_sensitivity,
    node_product_weights,
    validate_predictions,
)


@pytest.mark.parametrize("score_case", ["untied", "tied", "constant"])
@pytest.mark.parametrize("weight_case", ["ones", "linear", "lognormal", "with_zeros"])
def test_fixed_score_weighted_ap_matches_independent_sklearn_oracle(score_case, weight_case):
    rng = np.random.default_rng(8172)
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1] * 8)
    scores = rng.uniform(size=len(labels))
    if score_case == "tied":
        scores = np.round(scores, 1)
    elif score_case == "constant":
        scores[:] = .4
    if weight_case == "ones":
        weights = np.ones(len(labels))
    elif weight_case == "linear":
        weights = np.linspace(.05, 3., len(labels))
    else:
        weights = rng.lognormal(mean=0., sigma=2., size=len(labels))
        if weight_case == "with_zeros":
            weights[::3] = 0.
    cached = FixedScoreAveragePrecision(labels, scores)
    assert cached.evaluate(weights) == pytest.approx(
        average_precision_score(labels, scores, sample_weight=weights), abs=1e-14, rel=1e-14,
    )


def test_fixed_score_ap_treats_a_constant_score_as_one_tie_group():
    labels = np.array([1, 1, 0, 0])
    weights = np.array([2., 3., 5., 7.])
    result = FixedScoreAveragePrecision(labels, [.5] * 4).evaluate(weights)
    assert result == pytest.approx(5. / 17.)


def test_fixed_score_ap_matches_sklearn_for_actual_exponential_endpoint_weights():
    fronts = pd.DataFrame({
        "t": 2016, "kind": "pair", "key": ["a|b", "a|c", "b|c", "b|d", "c|d", "d|e"],
    })
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([.8, .6, .6, .3, .3, .3])
    cached = FixedScoreAveragePrecision(labels, scores)
    rng = np.random.default_rng(42)
    for _ in range(13):
        nodes = dict(zip("abcde", rng.exponential(size=5), strict=True))
        weights = node_product_weights(fronts, nodes)
        assert cached.evaluate(weights) == pytest.approx(
            average_precision_score(labels, scores, sample_weight=weights), abs=1e-14, rel=1e-14,
        )


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([0, 1], [.5]), ([[0], [1]], [.2, .8]), ([0, 1], [[.2], [.8]]),
    ([0, 2], [.2, .8]), ([0, np.nan], [.2, .8]), ([0, 1], [.2, np.nan]), ([0, 1], [.2, np.inf]),
])
def test_fixed_score_ap_rejects_invalid_prediction_dimensions_or_values(labels, scores):
    with pytest.raises(ValueError):
        FixedScoreAveragePrecision(labels, scores)


@pytest.mark.parametrize("weights", [
    [], [1.], [[1.], [1.]], [1., -1.], [1., np.nan], [1., np.inf], [0., 0.],
])
def test_fixed_score_ap_rejects_invalid_weight_dimensions_or_values(weights):
    cached = FixedScoreAveragePrecision([0, 1], [.2, .8])
    with pytest.raises(ValueError):
        cached.evaluate(weights)


def prediction_fixture() -> pd.DataFrame:
    parts = []
    for seed in [0, 1]:
        for group, score in [("S", [.6, .5, .4, .3]), ("S+rc", [.9, .2, .8, .1])]:
            parts.append(pd.DataFrame({
                "t": [2016] * 4, "kind": ["pair"] * 4, "key": ["a|b", "a|c", "b|c", "c|d"],
                "label": [1, 0, 1, 0], "seed": seed, "group": group, "score": score,
                "condition": "df20", "cohort": "adjacent", "prox_defined": [True, False, True, True],
            }))
    return pd.concat(parts, ignore_index=True)


def test_shared_node_weights_and_endpoint_order_invariance():
    fronts = pd.DataFrame({"t": 2016, "kind": "pair", "key": ["a|b", "a|c", "b|c"]})
    values = {"a": 2., "b": 3., "c": 5.}
    np.testing.assert_array_equal(node_product_weights(fronts, values), [6., 10., 15.])
    reverse = fronts.assign(key=["b|a", "c|a", "c|b"])
    np.testing.assert_array_equal(node_product_weights(reverse, values), [6., 10., 15.])


def test_equal_predictors_have_zero_paired_effect_in_every_draw():
    predictions = prediction_fixture()
    predictions.loc[predictions["group"] == "S+rc", "score"] = [.6, .5, .4, .3] * 2
    summary = node_multiplier_sensitivity(predictions, seed=9)
    result = summary["results"][0]
    assert summary["n_resamples"] == 2000
    assert result["mean_delta_ap"] == 0
    assert result["dependence_sensitivity_interval"] == [0., 0.]
    assert result["draw_mean_paired_delta_ap"] == [0.] * 2000
    assert result["n_fronts"] == 4
    assert result["n_prox_defined"] == 3


def test_one_draw_uses_the_same_concept_weights_across_cohorts_and_seeds():
    predictions = prediction_fixture()
    subset = predictions.loc[predictions["key"].isin(["a|b", "a|c"])].copy()
    subset["cohort"] = "raw_first_observed"
    combined = pd.concat([predictions, subset], ignore_index=True)
    summary = node_multiplier_sensitivity(combined, n_resamples=3, seed=17)
    multipliers = dict(zip(["a", "b", "c", "d"], np.random.default_rng(17).exponential(size=4), strict=True))
    for result in summary["results"]:
        cell = combined.loc[(combined["cohort"] == result["cohort"]) & (combined["seed"] == 0)]
        baseline = cell.loc[cell["group"] == "S"].sort_values(["t", "kind", "key"])
        augmented = cell.loc[cell["group"] == "S+rc"].sort_values(["t", "kind", "key"])
        weights = node_product_weights(baseline, multipliers)
        expected = average_precision_score(baseline["label"], augmented["score"], sample_weight=weights)
        expected -= average_precision_score(baseline["label"], baseline["score"], sample_weight=weights)
        assert result["draw_mean_paired_delta_ap"][0] == pytest.approx(expected)
    assert summary == node_multiplier_sensitivity(combined, n_resamples=3, seed=17)


def test_primary_effect_averages_paired_seed_metrics_instead_of_probabilities():
    predictions = prediction_fixture()
    changed = (predictions["seed"] == 1) & (predictions["group"] == "S+rc")
    predictions.loc[changed, "score"] = [.1, .9, .2, .8]
    result = node_multiplier_sensitivity(predictions, n_resamples=3)["results"][0]
    deltas = [r["augmented_ap"] - r["baseline_ap"] for r in result["per_seed"]]
    assert result["mean_delta_ap"] == pytest.approx(np.mean(deltas))
    assert result["seed_delta_sd"] == pytest.approx(np.std(deltas, ddof=1))
    averaged = predictions.groupby(["group", "key"], sort=True)[["score", "label"]].mean()
    wrong = average_precision_score(averaged.loc["S+rc", "label"], averaged.loc["S+rc", "score"])
    wrong -= average_precision_score(averaged.loc["S", "label"], averaged.loc["S", "score"])
    assert result["mean_delta_ap"] != pytest.approx(wrong)


@pytest.mark.parametrize("defect", ["missing_column", "missing_id", "duplicate", "changed_label", "score"])
def test_prediction_schema_rejects_known_defects(defect):
    predictions = prediction_fixture()
    if defect == "missing_column":
        predictions = predictions.drop(columns="prox_defined")
    elif defect == "missing_id":
        predictions = predictions.drop(index=0)
    elif defect == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif defect == "changed_label":
        predictions.loc[0, "label"] = 0
    else:
        predictions.loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        validate_predictions(predictions)


def test_reversed_pair_duplicate_is_rejected():
    duplicate = pd.DataFrame({"t": 2016, "kind": "pair", "key": ["a|b", "b|a"]})
    with pytest.raises(ValueError, match="duplicate"):
        canonical_front_ids(duplicate)


def test_missing_front_id_and_missing_node_multiplier_are_rejected():
    fronts = pd.DataFrame({"t": 2016, "kind": "pair", "key": ["a|b"]})
    with pytest.raises(ValueError, match="missing front IDs"):
        canonical_front_ids(fronts.assign(key=None))
    with pytest.raises(ValueError, match="missing concept multiplier"):
        node_product_weights(fronts, {"a": 1.})


def test_missing_model_group_in_one_seed_is_rejected():
    predictions = prediction_fixture()
    predictions = predictions.loc[~((predictions["seed"] == 1) & (predictions["group"] == "S+rc"))]
    with pytest.raises(ValueError, match="missing model groups"):
        validate_predictions(predictions)


def test_residual_increment_is_also_compared_after_raw_proximity_and_degree():
    predictions = prediction_fixture()
    raw = predictions.loc[predictions["group"] == "S+rc"].copy()
    raw["group"] = "S+prox+z"
    full = raw.copy()
    full["group"] = "S+prox+z+rc"
    summary = node_multiplier_sensitivity(pd.concat([predictions, raw, full]), n_resamples=3)
    nested = [r for r in summary["results"] if r["baseline"] == "S+prox+z"]
    assert len(nested) == 1 and nested[0]["augmented"] == "S+prox+z+rc"
    assert nested[0]["mean_delta_ap"] == 0.
