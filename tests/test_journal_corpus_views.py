from __future__ import annotations

import pandas as pd
import pytest

from tsnd.journal.corpus_views import materialize_corpus_views, select_corpus_view


def papers(year, n=12):
    return pd.DataFrame({"id": [f"W{year}_{i}" for i in range(n)], "year": [year] * n,
                         "acquisition_rank": list(range(n)), "mesh": [["a", "b"]] * n})


def test_api_order_is_capped_by_rank_not_file_row_order():
    frame = papers(2016).sample(frac=1, random_state=9)
    selected = select_corpus_view(frame, "api_order", cap=4)
    assert selected["acquisition_rank"].tolist() == [0, 1, 2, 3]
    assert len(select_corpus_view(frame, "full", cap=4)) == 12
    assert len(frame) == 12


def test_random_view_is_reproducible_and_keeps_future_outcome_corpus_full():
    frame = pd.concat([papers(2016, 50), papers(2017, 50), papers(2019, 50)], ignore_index=True)
    first = select_corpus_view(frame, "random", cap=10, seed=7)
    second = select_corpus_view(frame.sample(frac=1, random_state=3), "random", cap=10, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("year").size().to_dict() == {2016: 10, 2017: 50, 2019: 50}
    assert first[first.year.eq(2016)]["acquisition_rank"].tolist() != list(range(10))
    other = select_corpus_view(frame, "random", cap=10, seed=8)
    assert set(first[first.year.eq(2016)].id) != set(other[other.year.eq(2016)].id)
    pd.testing.assert_frame_equal(first[first.year.gt(2016)], other[other.year.gt(2016)])


def test_random_year_draw_does_not_depend_on_other_year_sizes():
    frame = pd.concat([papers(2015, 40), papers(2016, 40)], ignore_index=True)
    both = select_corpus_view(frame, "random", cap=6, seed=2)
    alone = select_corpus_view(papers(2016, 40), "random", cap=6, seed=2)
    assert both[both.year.eq(2016)].id.tolist() == alone.id.tolist()


@pytest.mark.parametrize("problem", ["missing_rank", "duplicate_rank", "duplicate_id", "string_rank"])
def test_invalid_snapshot_identity_or_rank_fails(problem):
    frame = papers(2016)
    if problem == "missing_rank":
        frame = frame.drop(columns="acquisition_rank")
    elif problem == "duplicate_rank":
        frame.loc[1, "acquisition_rank"] = 0
    elif problem == "string_rank":
        frame["acquisition_rank"] = frame["acquisition_rank"].astype(str)
    else:
        frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError):
        select_corpus_view(frame, "random", cap=4)


def test_materialized_views_share_complete_future_memberships(tmp_path):
    source = tmp_path / "full_snapshot"
    source.mkdir()
    for year in (2016, 2017):
        papers(year, 20).to_parquet(source / f"works_{year}.parquet", index=False)
    summary = materialize_corpus_views(source, tmp_path / "views", years=[2016, 2017],
                                       cap=5, sampling_seeds=[0, 1])
    assert len(summary) == 8
    assert summary[summary.year.eq(2017)].n_retained.eq(20).all()
    original = pd.read_parquet(source / "works_2017.parquet")
    for name in ("full", "api_order", "random_s0", "random_s1"):
        actual = pd.read_parquet(tmp_path / "views" / name / "works_2017.parquet")
        pd.testing.assert_frame_equal(actual, original)
    with pytest.raises(FileExistsError):
        materialize_corpus_views(source, tmp_path / "views", years=[2016, 2017],
                                 cap=5, sampling_seeds=[0, 1])
