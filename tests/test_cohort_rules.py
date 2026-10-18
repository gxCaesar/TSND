r"""The adjacent-year and cumulative first-ever candidate rules must differ, offline.

The paper's central negative result contrasts two candidate sets:

    adjacent_t   = E_t \\ E_{t-1}
    first_ever_t = E_t \\ (E_{start} u ... u E_{t-1})

and reports that the forecasting increment present on the first set vanishes on the second,
because the difference between them is exactly the pairs that lapsed and reappeared. These
tests build tiny year graphs on disk and check that both rules are implemented and that they
separate a lapsed-and-reappeared pair from a pair first observed since the corpus start. No data download, no
embedding, no model.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytest.importorskip("pandas")

from tsnd.cli import _fronts_born, _prior_state  # noqa: E402


def write_graph(terms_dir: Path, year: int, pairs: list[tuple[str, str]]) -> None:
    """Persist a year graph in the pickle layout the pipeline reads."""
    vocab = sorted({unit for pair in pairs for unit in pair})
    index = {unit: i for i, unit in enumerate(vocab)}
    edges = [(index[u], index[v]) for u, v in pairs]
    terms_dir.mkdir(parents=True, exist_ok=True)
    with open(terms_dir / f"year_{year}.pkl", "wb") as handle:
        pickle.dump({"vocab": vocab, "edges": edges, "paper_units": []}, handle, protocol=4)


@pytest.fixture()
def terms(tmp_path: Path) -> Path:
    """A four-year history with one lapsed pair and one genuine newcomer at 2013.

    2010  (a,b)            <- (a,b) is first seen here
    2011  (a,b), (c,d)
    2012  (c,d)            <- (a,b) lapses
    2013  (a,b), (c,d), (e,f)
             ^ re-emerges      ^ first observed since 2010
    """
    terms_dir = tmp_path / "terms"
    write_graph(terms_dir, 2010, [("a", "b")])
    write_graph(terms_dir, 2011, [("a", "b"), ("c", "d")])
    write_graph(terms_dir, 2012, [("c", "d")])
    write_graph(terms_dir, 2013, [("a", "b"), ("c", "d"), ("e", "f")])
    return terms_dir


def pairs_of(frame) -> set[str]:
    return set(frame[frame["kind"] == "pair"]["key"])


def test_adjacent_rule_admits_the_lapsed_pair(terms: Path) -> None:
    """(a,b) vanished in 2012, so the adjacent-year rule calls it new in 2013."""
    assert pairs_of(_fronts_born(terms, 2013)) == {"a|b", "e|f"}


def test_first_ever_rule_excludes_the_lapsed_pair(terms: Path) -> None:
    """Against cumulative history only (e,f) has never been seen before."""
    assert pairs_of(_fronts_born(terms, 2013, start_year=2010)) == {"e|f"}


def test_the_two_rules_differ_exactly_by_the_re_emerging_pair(terms: Path) -> None:
    """This difference is the conflation the paper measures at 59 percent on oncology."""
    adjacent = pairs_of(_fronts_born(terms, 2013))
    first_ever = pairs_of(_fronts_born(terms, 2013, start_year=2010))
    assert first_ever < adjacent
    assert adjacent - first_ever == {"a|b"}


def test_prior_state_scans_the_requested_window(terms: Path) -> None:
    """Adjacent looks at one year; cumulative accumulates every earlier year."""
    units_adj, pairs_adj = _prior_state(terms, 2013, None)
    units_cum, pairs_cum = _prior_state(terms, 2013, 2010)
    assert pairs_adj == {("c", "d")}
    assert pairs_cum == {("a", "b"), ("c", "d")}
    assert units_adj == {"c", "d"}
    assert units_cum == {"a", "b", "c", "d"}


def test_first_ever_is_left_censored_at_the_start_year(terms: Path) -> None:
    """Starting the history at 2012 hides the 2010-2011 usage of (a,b).

    This is the left-censoring the paper discloses: first-ever means first observed since the
    corpus start year, not first in the literature.
    """
    late = pairs_of(_fronts_born(terms, 2013, start_year=2012))
    assert "a|b" in late, "a pair used only before the start year looks first-ever"
