"""The reference collector must distinguish complete coverage from unusable pages."""

from __future__ import annotations

import urllib.error

import pandas as pd
import pytest

from tsnd.journal.reference_corpus import ReferenceAccessError, request_page, validate_rows, works_url


def reference_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["W1", "W2"], "year": [2016, 2016], "mesh": [["a"], ["b"]],
        "acquisition_rank": [0, 1],
    })


def test_coverage_accepts_a_complete_year() -> None:
    result = validate_rows(reference_rows(), 2016, 2)
    assert result["complete"] is True
    assert result["unique_ids"] == 2


@pytest.mark.parametrize("defect", ["missing", "duplicate", "future", "rank"])
def test_coverage_rejects_specific_acquisition_defects(defect: str) -> None:
    rows = reference_rows()
    if defect == "missing":
        rows = rows.iloc[:1]
    elif defect == "duplicate":
        rows.loc[1, "id"] = "W1"
    elif defect == "future":
        rows.loc[1, "year"] = 2017
    else:
        rows.loc[1, "acquisition_rank"] = 0
    assert validate_rows(rows, 2016, 2)["complete"] is False


def test_queries_contain_no_credential_parameter() -> None:
    url = works_url(2016, "a cursor")
    assert "api_key" not in url
    assert "has_abstract" in url
    assert "abstract_inverted_index" not in url


def test_access_error_does_not_expose_key(monkeypatch) -> None:
    secret = "test-secret-that-must-not-appear"

    def deny(request, timeout):
        assert request.headers["Authorization"] == f"Bearer {secret}"
        raise urllib.error.HTTPError(request.full_url, 401, "denied", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", deny)
    with pytest.raises(ReferenceAccessError) as error:
        request_page(works_url(2016, "*"), secret)
    assert secret not in str(error.value)
