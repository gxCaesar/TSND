"""Exercise typed profiles and observation histories with small synthetic inputs.

Run from the repository root with ``PYTHONPATH=src python examples/offline_profile.py``.
These hand-constructed arrays and records are not experimental measurements.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tsnd.data.preprocessing import build_year
from tsnd.journal.cohort import classify_cohort, raw_mesh_graph
from tsnd.models.decomposition import lexical_emergence, neighborhood_shift, recombination


def main() -> None:
    """Print finite components and expose an entry caused by DF filtering."""
    settings = {"terms": {"unit_source": "mesh", "min_df_per_year": 2,
                          "top_terms": 50_000, "max_units_per_paper": 40,
                          "unknown_df_rank": 1_000_000_000}}
    records = {2010: [["a", "b"], ["a"]], 2011: [["a", "b"], ["a", "b"]]}
    frames = {year: pd.DataFrame({"mesh": rows}) for year, rows in records.items()}
    raw = {year: raw_mesh_graph(frame) for year, frame in frames.items()}
    graph = {year: build_year(frame, settings) for year, frame in frames.items()}
    classified = classify_cohort(raw, graph, 2011, start_year=2010, min_df=2).set_index("key")
    pair = classified.loc["a|b"]
    assert pair.graph_first_observed and not pair.raw_first_observed
    assert pair.previous_graph_absence_cause == "df_threshold"

    prior = np.eye(2)
    le = lexical_emergence(np.array([[1., 1.]]), prior)
    ns = neighborhood_shift(np.array([[0., 1.]]), np.array([[1., 0.]]))
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    proximity = np.array([.8, .2, np.nan])
    log_degree = np.log1p([4., 6., 9.])
    rc = recombination(pairs, log_degree, proximity)
    assert rc.n_defined == 2 and pairs[2] not in rc.rc
    errors = [abs(value - (rc.coef[0] * log_degree[i] + rc.coef[1] * proximity[i] + rc.bias))
              for i, key in enumerate(pairs) if (value := rc.rc.get(key)) is not None]
    assert max(errors) < 1e-12
    print(json.dumps({
        "input_type": "synthetic_example", "lexical_emergence": le.tolist(),
        "neighborhood_shift": ns.tolist(),
        "recombination": {"|".join(key): value for key, value in rc.rc.items()},
        "pairs_with_prior_proximity": rc.n_defined,
        "pair_without_prior_proximity": "b|c",
        "raw_class": pair.raw_class, "graph_class": pair.graph_class,
        "graph_entry_cause": pair.previous_graph_absence_cause,
        "residual_identity_max_abs_error": max(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
