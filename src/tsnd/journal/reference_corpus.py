"""Collect a complete, dated OpenAlex reference without exposing API credentials."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tsnd.data.ingest import work_to_row

API_ROOT = "https://api.openalex.org"
SELECT_FIELDS = "id,publication_year,primary_topic,topics,mesh,primary_location,cited_by_count"


class ReferenceAccessError(RuntimeError):
    """The caller must supply valid access before collection can continue."""


def utc_now() -> str:
    """Return the acquisition time in UTC."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def works_url(year: int, cursor: str, per_page: int = 100) -> str:
    """Build the public query; authentication is never included in its URL."""
    params = {
        "filter": (
            f"publication_year:{year},has_abstract:true,has_pmid:true,"
            "primary_topic.subfield.id:subfields/2730"
        ),
        "select": SELECT_FIELDS,
        "per_page": str(per_page),
        "cursor": cursor,
    }
    return f"{API_ROOT}/works?{urllib.parse.urlencode(params)}"


def request_page(url: str, api_key: str) -> tuple[dict[str, Any], int]:
    """Fetch a page using a bearer token and sanitized failure messages."""
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "tsnd-reference-corpus/1"}
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            page = json.loads(payload)
            if not isinstance(page, dict) or not isinstance(page.get("results"), list):
                raise ValueError("OpenAlex response lacks a results list")
            return page, len(payload)
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (401, 403):
                raise ReferenceAccessError(
                    f"OpenAlex returned HTTP {status}; supply an activated, authorized API key."
                ) from None
            if status not in (429, 500, 502, 503, 504) or attempt == 4:
                message = f"OpenAlex request failed with HTTP {status}; no key was logged."
                raise RuntimeError(message) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                message = "OpenAlex connection failed after five attempts; partial pages are retained."
                raise RuntimeError(message) from None
        time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable retry state")


def validate_rows(rows: pd.DataFrame, year: int, expected: int) -> dict[str, int | bool]:
    """Check coverage, identity and dating before calling an acquisition complete."""
    required = {"id", "year", "mesh", "acquisition_rank"}
    if expected <= 0:
        raise ValueError("the oncology reference must have a positive query count in every year")
    if not required.issubset(rows):
        raise ValueError(f"reference rows lack required columns: {sorted(required - set(rows))}")
    duplicates = int(rows["id"].duplicated().sum())
    wrong_year = int((rows["year"] != year).sum())
    duplicate_ranks = int(rows["acquisition_rank"].duplicated().sum())
    return {
        "query_count": expected,
        "received_rows": len(rows),
        "unique_ids": int(rows["id"].nunique()),
        "duplicate_ids": duplicates,
        "wrong_year_rows": wrong_year,
        "duplicate_ranks": duplicate_ranks,
        "complete": bool(len(rows) == expected and not duplicates and not wrong_year and not duplicate_ranks),
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_year(output_dir: Path, year: int, api_key: str) -> dict[str, Any]:
    """Collect or resume one year, retaining regular page files until validated."""
    pages_dir = output_dir / "pages" / str(year)
    pages_dir.mkdir(parents=True, exist_ok=True)
    state_path = pages_dir / "state.json"
    target = output_dir / f"works_{year}.parquet"
    state: dict[str, Any]
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            frame = pd.read_parquet(target)
            coverage = validate_rows(frame, year, int(state["query_count"]))
            if not coverage["complete"]:
                raise ValueError(f"previously completed reference year {year} failed its coverage check")
            return state
    else:
        state = {
            "year": year, "started_utc": utc_now(), "next_cursor": "*", "pages": 0,
            "received_rows": 0, "traffic_bytes": 0, "complete": False,
            "public_query": works_url(year, "*"),
        }
        _write_json(state_path, state)
    while state["next_cursor"]:
        page_number = int(state["pages"])
        page_path = pages_dir / f"page_{page_number:06d}.parquet"
        metadata_path = page_path.with_suffix(".json")
        if page_path.exists() != metadata_path.exists():
            message = f"partial page transaction for year {year}, page {page_number}; inspect before resume"
            raise ValueError(message)
        if page_path.exists():
            page_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if page_metadata["requested_cursor"] != state["next_cursor"]:
                raise ValueError("saved page cursor disagrees with the resume state")
            frame = pd.read_parquet(page_path)
        else:
            page, transferred = request_page(works_url(year, str(state["next_cursor"])), api_key)
            records = page["results"]
            frame = pd.DataFrame([work_to_row(work, year, "oncology") for work in records])
            if frame.empty:
                frame = pd.DataFrame(columns=["id", "year", "mesh"])
            start_rank = int(state["received_rows"])
            frame["acquisition_rank"] = range(start_rank, start_rank + len(frame))
            meta = page.get("meta", {})
            page_metadata = {
                "requested_cursor": state["next_cursor"], "next_cursor": meta.get("next_cursor"),
                "query_count": int(meta["count"]), "traffic_bytes": transferred, "received_utc": utc_now(),
            }
            temporary = page_path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(page_path)
            _write_json(metadata_path, page_metadata)
        if "query_count" not in state:
            state["query_count"] = page_metadata["query_count"]
        state["last_query_count"] = page_metadata["query_count"]
        next_cursor = page_metadata["next_cursor"]
        if next_cursor == state["next_cursor"]:
            raise ValueError("OpenAlex returned a non-advancing cursor")
        state["next_cursor"] = next_cursor
        state["pages"] = page_number + 1
        state["received_rows"] += len(frame)
        state["traffic_bytes"] += int(page_metadata["traffic_bytes"])
        _write_json(state_path, state)
        if int(state["pages"]) % 25 == 0:
            print(f"year={year} rows={state['received_rows']} query_count={state['query_count']}", flush=True)
        time.sleep(0.1)
    frames = [
        pd.read_parquet(pages_dir / f"page_{number:06d}.parquet")
        for number in range(int(state["pages"]))
    ]
    combined = pd.concat(frames, ignore_index=True)
    coverage = validate_rows(combined, year, int(state["query_count"]))
    state.update(coverage)
    state["finished_utc"] = utc_now()
    state["query_count_changed"] = state["query_count"] != state["last_query_count"]
    if state["query_count_changed"]:
        state["complete"] = False
    temporary = target.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(target)
    _write_json(state_path, state)
    if not state["complete"]:
        raise ValueError(f"year {year} acquisition coverage is inconsistent; preserved for inspection")
    return state


def collect_reference(
    output_dir: Path, years: list[int], api_key_env: str = "OPENALEX_API_KEY",
) -> dict[str, Any]:
    """Collect the current reference using only the caller's injected environment key."""
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ReferenceAccessError(f"Inject an activated OpenAlex key as environment variable {api_key_env}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "domain": "oncology", "subfield_id": "2730", "acquisition_type": "current_api_reference",
        "historical_snapshot_extension": False, "abstract_filter": True, "require_pmid": True,
        "downloaded_abstract_text": False, "select_fields": SELECT_FIELDS.split(","),
        "started_utc": utc_now(), "years": [],
    }
    for year in years:
        row = collect_year(output_dir, year, api_key)
        result["years"].append(row)
        _write_json(output_dir / "acquisition.json", result)
    result["complete"] = all(row["complete"] for row in result["years"])
    result["finished_utc"] = utc_now()
    _write_json(output_dir / "acquisition.json", result)
    return result
