"""OpenAlex data ingest for TSND.

The default path uses the OpenAlex API exactly as the validated pilot did:
request only downstream fields with ``select=``, reconstruct abstracts from
``abstract_inverted_index``, paginate with cursors, and retry transient request
failures with bounded exponential backoff.

For full-scale runs, ``config["data"]["source"] == "openalex_snapshot"`` or an
explicit ``snapshot_path`` enables a streaming OpenAlex works snapshot reader.
It scans compressed JSON-lines files, filters records by year and configured
topic/subfield domains, and writes only the filtered rows to parquet in chunks.
This avoids storing the full raw works dump locally. The API path remains the
default because it is simpler and adequate at pilot scale.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

WORK_COLUMNS = [
    "id",
    "year",
    "abstract",
    "concepts",
    "subfield_ids",
    "venue_id",
    "cited_by_count",
    "domain",
    "mesh",
]


@dataclass(frozen=True)
class DomainSpec:
    """A configured OpenAlex topic/subfield domain."""

    name: str
    subfield_search: str | None = None
    subfield_id: str | None = None
    topic_search: str | None = None


def default_config() -> dict[str, Any]:
    """Return a runnable default config for local API-scale ingest."""
    return {
        "data": {
            "source": "openalex_api",
            "api_base": "https://api.openalex.org",
            "api_key_env": "OPENALEX_API_KEY",
            "user_agent": "tsnd-data",
            "select_fields": [
                "id",
                "publication_year",
                "abstract_inverted_index",
                "primary_topic",
                "topics",
                "mesh",
                "primary_location",
                "cited_by_count",
            ],
            "domains": [
                {"name": "ai", "subfield_search": "artificial intelligence"},
                {"name": "biomedical-ai", "subfield_search": "biomedical artificial intelligence"},
                {"name": "quantum-materials", "subfield_search": "quantum materials"},
            ],
            "year_start": 2010,
            "year_end": 2024,
            "has_abstract": True,
            "require_pmid": False,
            "per_page": 200,
            "max_papers": 1_000_000,
            "out_dir": "data",
            "mailto": None,
            "api_retries": 5,
            "api_timeout_s": 60,
            "api_backoff_initial_s": 1.0,
            "api_backoff_max_s": 30.0,
            "polite_sleep_s": 0.1,
            "progress_every": 20_000,
            "snapshot_path": None,
            "snapshot_glob": "**/*.gz",
            "snapshot_chunk_rows": 50_000,
        }
    }


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load`` and default fallbacks."""
    cfg = default_config()
    if path is None:
        return cfg
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config must be a mapping: {config_path}")
    merged = _deep_merge(cfg, loaded)
    raw_data = loaded.get("data", {})
    if isinstance(raw_data, dict) and "domains" not in raw_data and "subfield_search" in raw_data:
        search = str(raw_data["subfield_search"])
        merged["data"]["domains"] = [{"name": _slug(search), "subfield_search": search}]
    return merged


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mapping objects."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _slug(value: str) -> str:
    """Normalize a display string into a stable domain label."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)


def domain_specs(config: Mapping[str, Any]) -> list[DomainSpec]:
    """Parse ``config.data.domains`` into typed domain specs."""
    data_cfg = _data_config(config)
    raw_domains = data_cfg.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise ValueError("config.data.domains must be a non-empty list")
    specs: list[DomainSpec] = []
    for item in raw_domains:
        if isinstance(item, str):
            specs.append(DomainSpec(name=_slug(item), subfield_search=item))
            continue
        if not isinstance(item, dict):
            raise TypeError("each domain must be a string or mapping")
        name = str(item.get("name") or item.get("domain") or item.get("subfield_search") or "")
        if not name:
            raise ValueError("domain mapping requires name or subfield_search")
        spec = DomainSpec(
            name=name,
            subfield_search=_optional_str(item.get("subfield_search") or item.get("search")),
            subfield_id=_strip_openalex_id(_optional_str(item.get("subfield_id"))),
            topic_search=_optional_str(item.get("topic_search")),
        )
        if spec.subfield_search is None and spec.subfield_id is None and spec.topic_search is None:
            raise ValueError(f"domain {name!r} needs subfield_search, subfield_id, or topic_search")
        specs.append(spec)
    return specs


def _data_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the data section from a config mapping."""
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("config.data must be a mapping")
    return data_cfg


def _optional_str(value: object) -> str | None:
    """Return a stripped string or ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_openalex_id(value: str | None) -> str | None:
    """Return the terminal OpenAlex id component."""
    if value is None:
        return None
    return value.rstrip("/").split("/")[-1]


def _request_json(url: str, data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """GET a JSON URL with retry and bounded exponential backoff."""
    retries = _positive_int(data_cfg, "api_retries")
    timeout_s = _positive_int(data_cfg, "api_timeout_s")
    backoff_initial = _positive_float(data_cfg, "api_backoff_initial_s")
    backoff_max = _positive_float(data_cfg, "api_backoff_max_s")
    user_agent = str(data_cfg["user_agent"])
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                loaded = json.load(response)
            if not isinstance(loaded, dict):
                raise TypeError(f"OpenAlex response is not a JSON object: {url}")
            return loaded
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            wait_s = min(backoff_initial * (2**attempt), backoff_max)
            LOGGER.warning(
                "GET failed (%d/%d), retry in %.1fs: %s",
                attempt + 1,
                retries,
                wait_s,
                str(exc)[: int(data_cfg.get("log_error_chars", 160))],
            )
            time.sleep(wait_s)
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last_error


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    """Read and validate a positive integer config value."""
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _positive_float(config: Mapping[str, Any], key: str) -> float:
    """Read and validate a positive float config value."""
    value = float(config[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0")
    return value


def resolve_subfield(search: str, config: Mapping[str, Any], api_key: str | None, mailto: str | None) -> str:
    """Resolve an OpenAlex subfield search string to a terminal subfield id."""
    data_cfg = _data_config(config)
    api_base = str(data_cfg["api_base"]).rstrip("/")
    params: dict[str, str] = {"search": search}
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    url = f"{api_base}/subfields?{urllib.parse.urlencode(params)}"
    response = _request_json(url, data_cfg)
    results = response.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"no OpenAlex subfield found for search={search!r}")
    first = results[0]
    if not isinstance(first, dict):
        raise TypeError(f"unexpected OpenAlex subfield result for search={search!r}")
    resolved = _strip_openalex_id(_optional_str(first.get("id")))
    if resolved is None:
        raise RuntimeError(f"OpenAlex subfield result has no id for search={search!r}")
    LOGGER.info("resolved subfield %r -> %s (%s)", search, resolved, first.get("display_name"))
    return resolved


def resolve_domains(config: Mapping[str, Any], api_key: str | None, mailto: str | None) -> list[DomainSpec]:
    """Resolve configured domain searches into subfield ids where needed."""
    specs = domain_specs(config)
    resolved: list[DomainSpec] = []
    for spec in specs:
        subfield_id = spec.subfield_id
        if subfield_id is None and spec.subfield_search is not None:
            subfield_id = resolve_subfield(spec.subfield_search, config, api_key, mailto)
        resolved.append(
            DomainSpec(
                name=spec.name,
                subfield_search=spec.subfield_search,
                subfield_id=subfield_id,
                topic_search=spec.topic_search,
            )
        )
    return resolved


def reconstruct_abstract(inverted_index: Mapping[str, Any] | None) -> str:
    """Reconstruct an OpenAlex abstract from ``{token: [positions]}``."""
    if not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for token, raw_positions in inverted_index.items():
        if not isinstance(raw_positions, Iterable) or isinstance(raw_positions, (str, bytes)):
            continue
        for raw_position in raw_positions:
            try:
                pairs.append((int(raw_position), str(token)))
            except (TypeError, ValueError):
                LOGGER.debug("skip invalid abstract position token=%r position=%r", token, raw_position)
    pairs.sort(key=lambda item: item[0])
    return " ".join(token for _, token in pairs)


def topic_names_and_subfields(work: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Extract lowercase topic names and terminal subfield ids from a work."""
    topic_names: list[str] = []
    subfield_ids: list[str] = []
    raw_topics = work.get("topics") or []
    if not isinstance(raw_topics, list):
        return topic_names, subfield_ids
    for topic in raw_topics:
        if not isinstance(topic, dict):
            continue
        display_name = topic.get("display_name")
        if display_name:
            topic_names.append(str(display_name).lower())
        subfield = topic.get("subfield") or {}
        if isinstance(subfield, dict):
            subfield_id = _strip_openalex_id(_optional_str(subfield.get("id")))
            if subfield_id:
                subfield_ids.append(subfield_id)
    return topic_names, sorted(set(subfield_ids))


def mesh_descriptors(work: Mapping[str, Any]) -> list[str]:
    """Extract normalized MeSH descriptor names from an OpenAlex work."""
    descriptors: list[str] = []
    seen: set[str] = set()
    raw_mesh = work.get("mesh") or []
    if not isinstance(raw_mesh, list):
        return descriptors
    for item in raw_mesh:
        if not isinstance(item, dict):
            continue
        descriptor_name = item.get("descriptor_name")
        if descriptor_name is None:
            continue
        name = str(descriptor_name).strip().lower()
        if name and name not in seen:
            descriptors.append(name)
            seen.add(name)
    return descriptors


def venue_id(work: Mapping[str, Any]) -> str | None:
    """Return the terminal OpenAlex source id for a work's primary venue."""
    primary_location = work.get("primary_location") or {}
    if not isinstance(primary_location, dict):
        return None
    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        return None
    return _strip_openalex_id(_optional_str(source.get("id")))


def work_to_row(work: Mapping[str, Any], year: int, domain: str) -> dict[str, Any]:
    """Convert an OpenAlex work object into the preserved parquet row schema."""
    topic_names, subfield_ids = topic_names_and_subfields(work)
    work_id = _strip_openalex_id(_optional_str(work.get("id")))
    if work_id is None:
        raise ValueError("OpenAlex work has no id")
    return {
        "id": work_id,
        "year": int(work.get("publication_year") or year),
        "abstract": reconstruct_abstract(_mapping_or_none(work.get("abstract_inverted_index"))),
        "concepts": topic_names,
        "subfield_ids": subfield_ids,
        "venue_id": venue_id(work),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "domain": domain,
        "mesh": mesh_descriptors(work),
    }


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    """Return ``value`` if it is a mapping, otherwise ``None``."""
    return value if isinstance(value, Mapping) else None


def pull_domain_year(
    domain: DomainSpec,
    year: int,
    config: Mapping[str, Any],
    api_key: str | None,
    mailto: str | None,
    max_papers: int | None = None,
) -> pd.DataFrame:
    """Pull one domain-year from the OpenAlex API with cursor pagination."""
    data_cfg = _data_config(config)
    if domain.subfield_id is None and domain.topic_search is None:
        raise ValueError(f"domain {domain.name!r} must be resolved before API pull")
    per_page = _positive_int(data_cfg, "per_page")
    row_cap = max_papers if max_papers is not None else _positive_int(data_cfg, "max_papers")
    if row_cap <= 0:
        raise ValueError("max_papers must be > 0")
    rows: list[dict[str, Any]] = []
    total: int | None = None
    cursor: str | None = "*"
    while cursor and len(rows) < row_cap:
        url = _works_url(domain, year, cursor, config, api_key, mailto)
        page = _request_json(url, data_cfg)
        meta = page.get("meta") or {}
        if total is None and isinstance(meta, dict):
            total = int(meta.get("count") or 0)
            LOGGER.info("domain=%s year=%d: %d works to pull", domain.name, year, total)
        results = page.get("results")
        if not isinstance(results, list):
            raise TypeError("OpenAlex works response missing list results")
        for work in results:
            if not isinstance(work, dict):
                continue
            rows.append(work_to_row(work, year, domain.name))
            if len(rows) >= row_cap:
                break
        cursor = str(meta.get("next_cursor")) if isinstance(meta, dict) and meta.get("next_cursor") else None
        progress_every = _positive_int(data_cfg, "progress_every")
        if len(rows) % progress_every < per_page:
            LOGGER.info("domain=%s year=%d: %d / %s pulled", domain.name, year, len(rows), total)
        sleep_s = float(data_cfg["polite_sleep_s"])
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return _rows_to_frame(rows)


def _works_url(
    domain: DomainSpec,
    year: int,
    cursor: str,
    config: Mapping[str, Any],
    api_key: str | None,
    mailto: str | None,
) -> str:
    """Build an OpenAlex works API URL for one cursor page."""
    data_cfg = _data_config(config)
    api_base = str(data_cfg["api_base"]).rstrip("/")
    filters = _filters_for_domain_year(domain, year, data_cfg)
    select_fields = data_cfg.get("select_fields")
    if not isinstance(select_fields, list) or not select_fields:
        raise ValueError("config.data.select_fields must be a non-empty list")
    select_field_names = [str(field) for field in select_fields]
    if "mesh" not in select_field_names:
        select_field_names.append("mesh")
    params: dict[str, str | int] = {
        "filter": ",".join(filters),
        "select": ",".join(select_field_names),
        "per-page": _positive_int(data_cfg, "per_page"),
        "cursor": cursor,
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    return f"{api_base}/works?{urllib.parse.urlencode(params)}"


def _filters_for_domain_year(domain: DomainSpec, year: int, data_cfg: Mapping[str, Any]) -> list[str]:
    """Build OpenAlex filter clauses for a configured domain-year."""
    filters = [f"publication_year:{year}"]
    if bool(data_cfg.get("has_abstract", True)):
        filters.append("has_abstract:true")
    if bool(data_cfg.get("require_pmid", False)):
        filters.append("has_pmid:true")
    if domain.subfield_id is not None:
        filters.append(f"primary_topic.subfield.id:subfields/{domain.subfield_id}")
    elif domain.topic_search is not None:
        filters.append(f"default.search:{domain.topic_search}")
    else:
        raise ValueError(f"domain {domain.name!r} has no API filter")
    return filters


def pull_year(
    domains: Sequence[DomainSpec],
    year: int,
    config: Mapping[str, Any],
    api_key: str | None,
    mailto: str | None,
    max_papers: int | None = None,
) -> pd.DataFrame:
    """Pull all configured domains for one publication year."""
    frames = [
        pull_domain_year(domain, year, config, api_key, mailto, max_papers=max_papers)
        for domain in domains
    ]
    if not frames:
        return pd.DataFrame(columns=WORK_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    return _postprocess_frame(frame)


def _rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a parquet-ready frame with the preserved column order."""
    frame = pd.DataFrame(rows, columns=WORK_COLUMNS)
    return _postprocess_frame(frame)


def _postprocess_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop empty abstracts and enforce the public work parquet schema."""
    if frame.empty:
        return pd.DataFrame(columns=WORK_COLUMNS)
    missing = [column for column in WORK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"work frame missing columns: {missing}")
    out = frame.loc[:, WORK_COLUMNS].copy()
    out = out[out["abstract"].astype(str).str.len() > 0].reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    out["cited_by_count"] = out["cited_by_count"].fillna(0).astype(int)
    return out


def write_api_years(
    config: Mapping[str, Any],
    years: Sequence[int],
    out_dir: Path,
    max_papers: int | None = None,
) -> pd.DataFrame:
    """Pull API data and write ``works_{year}.parquet`` files."""
    data_cfg = _data_config(config)
    api_key_env = str(data_cfg["api_key_env"])
    api_key = os.environ.get(api_key_env)
    mailto = _optional_str(data_cfg.get("mailto"))
    domains = resolve_domains(config, api_key, mailto)
    manifest: list[dict[str, int]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        out_path = out_dir / f"works_{year}.parquet"
        if out_path.exists():
            n_rows = len(pd.read_parquet(out_path, columns=["id"]))
            LOGGER.info("year %d already exists (%s), skip", year, out_path)
        else:
            frame = pull_year(domains, year, config, api_key, mailto, max_papers=max_papers)
            frame.to_parquet(out_path, index=False)
            n_rows = len(frame)
            LOGGER.info(
                "year %d: wrote %d rows -> %s (%.1f MB)",
                year,
                n_rows,
                out_path,
                out_path.stat().st_size / float(data_cfg.get("bytes_per_mb", 1_000_000)),
            )
        manifest.append({"year": int(year), "n": int(n_rows)})
    manifest_frame = pd.DataFrame(manifest, columns=["year", "n"])
    manifest_frame.to_csv(out_dir / "manifest.csv", index=False)
    LOGGER.info("DONE. total rows: %d", int(manifest_frame["n"].sum()) if not manifest_frame.empty else 0)
    return manifest_frame


def stream_snapshot_year_to_parquet(
    snapshot_path: Path,
    domains: Sequence[DomainSpec],
    year: int,
    out_path: Path,
    config: Mapping[str, Any],
    max_papers: int | None = None,
) -> int:
    """Stream-filter an OpenAlex works snapshot year into one parquet artifact.

    The output preserves the work parquet schema, with shape ``(n_papers, 9)``.
    Rows are buffered only up to ``config.data.snapshot_chunk_rows`` before being
    written, so the full raw snapshot is never materialized in memory.
    """
    data_cfg = _data_config(config)
    chunk_rows = _positive_int(data_cfg, "snapshot_chunk_rows")
    row_cap = max_papers if max_papers is not None else _positive_int(data_cfg, "max_papers")
    if row_cap <= 0:
        raise ValueError("max_papers must be > 0")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("streaming snapshot ingest requires pyarrow") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    writer: Any | None = None
    buffer: list[dict[str, Any]] = []
    n_rows = 0
    try:
        for work in iter_snapshot_works(snapshot_path, str(data_cfg["snapshot_glob"])):
            publication_year = work.get("publication_year")
            if int(publication_year or -1) != year:
                continue
            if bool(data_cfg.get("has_abstract", True)) and not work.get("abstract_inverted_index"):
                continue
            for domain_name in matching_domains(work, domains):
                buffer.append(work_to_row(work, year, domain_name))
                if len(buffer) >= chunk_rows:
                    writer = _write_snapshot_chunk(buffer, tmp_path, writer, pa, pq)
                    n_rows += len(buffer)
                    buffer.clear()
                if n_rows + len(buffer) >= row_cap:
                    break
            if n_rows + len(buffer) >= row_cap:
                break
        if buffer:
            writer = _write_snapshot_chunk(buffer, tmp_path, writer, pa, pq)
            n_rows += len(buffer)
            buffer.clear()
        if writer is None:
            pd.DataFrame(columns=WORK_COLUMNS).to_parquet(tmp_path, index=False)
        else:
            writer.close()
            writer = None
        tmp_path.replace(out_path)
        LOGGER.info("snapshot year %d: wrote %d rows -> %s", year, n_rows, out_path)
        return n_rows
    finally:
        if writer is not None:
            writer.close()


def _write_snapshot_chunk(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    writer: Any | None,
    pa_module: Any,
    pq_module: Any,
) -> Any:
    """Write one filtered snapshot chunk and return the active parquet writer."""
    frame = _rows_to_frame(rows)
    table = pa_module.Table.from_pandas(frame, preserve_index=False)
    active_writer = writer
    if active_writer is None:
        active_writer = pq_module.ParquetWriter(path, table.schema)
    active_writer.write_table(table)
    return active_writer


def iter_snapshot_works(snapshot_path: Path, glob_pattern: str) -> Iterator[dict[str, Any]]:
    """Yield work objects from a local OpenAlex snapshot path."""
    for path in _snapshot_files(snapshot_path, glob_pattern):
        LOGGER.info("scan snapshot file %s", path)
        with _open_text(path) as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    loaded = json.loads(text)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("skip invalid JSON line in %s: %s", path, exc)
                    continue
                if isinstance(loaded, dict):
                    yield loaded
                elif isinstance(loaded, list):
                    for item in loaded:
                        if isinstance(item, dict):
                            yield item


def _snapshot_files(snapshot_path: Path, glob_pattern: str) -> list[Path]:
    """Return snapshot files from a file or directory path."""
    if snapshot_path.is_file():
        return [snapshot_path]
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"snapshot_path does not exist: {snapshot_path}")
    return sorted(path for path in snapshot_path.glob(glob_pattern) if path.is_file())


def _open_text(path: Path) -> TextIO:
    """Open plain-text or gzip-compressed snapshot files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def matching_domains(work: Mapping[str, Any], domains: Sequence[DomainSpec]) -> list[str]:
    """Return all configured domain names matched by one work."""
    topic_names, subfield_ids = topic_names_and_subfields(work)
    subfield_set = set(subfield_ids)
    matches: list[str] = []
    for domain in domains:
        matched = False
        if domain.subfield_id is not None and domain.subfield_id in subfield_set:
            matched = True
        if domain.topic_search is not None:
            query = domain.topic_search.lower()
            matched = matched or any(query in topic for topic in topic_names)
        if matched:
            matches.append(domain.name)
    return matches


def write_snapshot_years(
    config: Mapping[str, Any],
    years: Sequence[int],
    snapshot_path: Path,
    out_dir: Path,
    max_papers: int | None = None,
) -> pd.DataFrame:
    """Stream-filter snapshot works and write per-year parquet artifacts."""
    data_cfg = _data_config(config)
    api_key_env = str(data_cfg["api_key_env"])
    api_key = os.environ.get(api_key_env)
    mailto = _optional_str(data_cfg.get("mailto"))
    domains = resolve_domains(config, api_key, mailto)
    manifest: list[dict[str, int]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        out_path = out_dir / f"works_{year}.parquet"
        if out_path.exists():
            n_rows = len(pd.read_parquet(out_path, columns=["id"]))
            LOGGER.info("year %d already exists (%s), skip", year, out_path)
        else:
            n_rows = stream_snapshot_year_to_parquet(
                snapshot_path=snapshot_path,
                domains=domains,
                year=year,
                out_path=out_path,
                config=config,
                max_papers=max_papers,
            )
        manifest.append({"year": int(year), "n": int(n_rows)})
    manifest_frame = pd.DataFrame(manifest, columns=["year", "n"])
    manifest_frame.to_csv(out_dir / "manifest.csv", index=False)
    return manifest_frame


def configured_years(config: Mapping[str, Any], single_year: int | None = None) -> list[int]:
    """Return the ingest year list from config or a CLI override."""
    if single_year is not None:
        return [int(single_year)]
    data_cfg = _data_config(config)
    start = int(data_cfg["year_start"])
    end = int(data_cfg["year_end"])
    if end < start:
        raise ValueError("config.data.year_end must be >= year_start")
    return list(range(start, end + 1))


def output_dir(config: Mapping[str, Any], override: str | Path | None = None) -> Path:
    """Return the parquet output directory from config or a CLI override."""
    if override is not None:
        return Path(override)
    internal = config.get("_data_dir")
    if internal is not None:
        return Path(str(internal))
    data_cfg = _data_config(config)
    configured = Path(str(data_cfg["out_dir"]))
    root = config.get("_root")
    return Path(str(root)) / configured if root is not None else configured


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for OpenAlex ingest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--year", type=int, default=None, help="pull a single year")
    parser.add_argument("--max-n", type=int, default=None, help="override max papers per domain-year")
    parser.add_argument("--source", choices=["openalex_api", "openalex_snapshot"], default=None)
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_yaml_config(args.config)
    data_cfg = dict(_data_config(config))
    if args.source is not None:
        data_cfg["source"] = args.source
    if args.snapshot_path is not None:
        data_cfg["snapshot_path"] = args.snapshot_path
    config["data"] = data_cfg

    years = configured_years(config, args.year)
    out_dir = output_dir(config, args.out_dir)
    source = str(data_cfg["source"])
    snapshot = _optional_str(data_cfg.get("snapshot_path"))
    if source == "openalex_snapshot" or snapshot is not None:
        if snapshot is None:
            raise ValueError("snapshot ingest requires config.data.snapshot_path or --snapshot-path")
        write_snapshot_years(config, years, Path(snapshot), out_dir, max_papers=args.max_n)
    elif source == "openalex_api":
        write_api_years(config, years, out_dir, max_papers=args.max_n)
    else:
        raise ValueError(f"unsupported data source: {source}")


if __name__ == "__main__":
    main()
