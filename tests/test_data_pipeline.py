"""Tests for TSND data artifacts built from toy paper corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd

from tsnd.data.ingest import reconstruct_abstract
from tsnd.data.preprocessing import build_year
from tsnd.data.splits import RollingSplit, validate_disjoint_ordered_splits

START_POSITION = 0
ABSTRACT_TEXT = "graph neural models forecast scientific novelty"
TOY_PAPER_UNITS = (
    ("graph_neural_networks", "scientific_novelty", "citation_prediction"),
    ("graph_neural_networks", "representation_learning", "citation_prediction"),
    ("temporal_alignment", "scientific_novelty", "representation_learning"),
    ("semantic_drift", "temporal_alignment", "citation_prediction"),
    ("semantic_drift", "scientific_novelty", "graph_neural_networks"),
)

AbstractIndex = dict[str, list[int]]


def _build_abstract_inverted_index(text: str) -> AbstractIndex:
    """Build an OpenAlex-shaped ``abstract_inverted_index``."""
    inverted_index: AbstractIndex = {}
    for position, token in enumerate(text.split(), start=START_POSITION):
        inverted_index.setdefault(token, []).append(position)
    return inverted_index


def test_abstract_inverted_index_does_reconstruct_original_text() -> None:
    """Assert production abstract inversion round-trips token positions."""
    inverted_index = _build_abstract_inverted_index(ABSTRACT_TEXT)

    reconstructed = reconstruct_abstract(inverted_index)

    assert reconstructed == ABSTRACT_TEXT


def test_vocab_and_cooccurrence_graph_does_link_toy_corpus() -> None:
    """Assert the production graph builder links a five-paper toy corpus."""
    frame = pd.DataFrame({"mesh": [list(row) for row in TOY_PAPER_UNITS]})
    config = {
        "terms": {
            "unit_source": "mesh",
            "min_df_per_year": 1,
            "top_terms": 10_000,
            "max_units_per_paper": 100,
            "unknown_df_rank": 1_000_000,
            "mesh_stoplist": [],
        }
    }

    graph = build_year(frame, config)

    vocab = cast(Sequence[str], graph["vocab"])
    uidx = cast(Mapping[str, int], graph["uidx"])
    edges = cast(Mapping[tuple[int, int], int], graph["edges"])
    assert len(uidx) == len(vocab)
    assert edges


def test_leakage_split_assertion_does_detect_overlapping_paper_ids() -> None:
    """Assert the production split gate rejects overlapping temporal cohorts."""
    split = RollingSplit(
        train_years=(2010, 2011),
        val_years=(2011, 2012),
        test_years=(2013,),
        horizons=(1,),
        data_year_end=2015,
    )

    try:
        validate_disjoint_ordered_splits(split)
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("overlapping train/validation years were accepted")
