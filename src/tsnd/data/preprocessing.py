"""Vocabulary and yearly co-occurrence graph preprocessing for TSND.

This module preserves the pilot ``year_{y}.pkl`` artifact schema:
``vocab``, ``uidx``, ``unit_df``, ``paper_units``, ``edges``, and
``n_papers``. The graph builder is year-local: no terms or edges from future
years are read when constructing ``G_t``.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z][a-z\-]{1,}")
DEFAULT_STOPWORDS = frozenset(
    """a an the this that these those of in on at to for from by with without within into
over under again further then once here there all any both each few more most other some such no
nor not only own same so than too very can will just should now we our us you your they their it
its he she his her i me my as is are was were be been being have has had do does did doing would
could may might must shall and or but if because while which who whom whose what when where why how
between among per via using used use based able about above after before below during through up
down out off across also however therefore thus hence whereas moreover furthermore approach method
methods results result paper study propose proposed novel new model models data show shows showed
present presents using used given provide provides one two three high low large small different
various several many much first second third etc al fig figure table section""".split()
)

YEAR_GRAPH_KEYS = {"vocab", "uidx", "unit_df", "paper_units", "edges", "n_papers"}

# Standard NLM MeSH check tags: administrative demographic descriptors attached to
# almost every indexed article (Humans appears in ~90% of records). They are not
# topical and would form spurious high-degree hubs in the co-occurrence graph, so
# the mesh unit path drops them by default. Override with terms.mesh_stoplist.
DEFAULT_MESH_CHECK_TAGS = frozenset({
    "humans", "animals", "male", "female", "pregnancy",
    "infant, newborn", "infant", "child, preschool", "child", "adolescent",
    "adult", "young adult", "middle aged", "aged", "aged, 80 and over",
})


def default_config() -> dict[str, Any]:
    """Return local defaults compatible with the pilot artifact interface."""
    return {
        "data": {
            "year_start": 2010,
            "year_end": 2024,
            "out_dir": "data",
        },
        "artifacts": {
            "out_dir": "artifacts",
        },
        "terms": {
            "min_df_per_year": 20,
            "top_terms": 8000,
            "max_units_per_paper": 40,
            "use_spacy_nounchunks": False,
            "ngram_min": 2,
            "ngram_max": 3,
            "min_token_len": 3,
            "unknown_df_rank": 1_000_000_000,
            "spacy_model": "en_core_web_sm",
            "spacy_n_process": 1,
            "spacy_batch_size": 1000,
            "spacy_disable": ["ner"],
            "spacy_max_units_per_paper": 40,
            "pickle_protocol": 4,
        },
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
    return _deep_merge(cfg, loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mapping objects."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def as_list(value: object) -> list[Any]:
    """Normalize parquet list-like columns into plain Python lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return []


def normalize_unit(value: str) -> str:
    """Normalize a vocabulary unit string to the pilot underscore format."""
    return re.sub(r"\s+", "_", value.strip().lower())


def extract_terms_fast(abstracts: Sequence[str], terms_config: Mapping[str, Any]) -> list[set[str]]:
    """Extract adjacent content-word n-grams from abstracts.

    Args:
        abstracts: Paper abstracts with shape ``(n_papers,)``.
        terms_config: ``config["terms"]`` values controlling token length and
            n-gram range.

    Returns:
        A list of per-paper term sets with length ``n_papers``.
    """
    ngram_min = int(terms_config["ngram_min"])
    ngram_max = int(terms_config["ngram_max"])
    min_token_len = int(terms_config["min_token_len"])
    if ngram_min <= 0 or ngram_max < ngram_min:
        raise ValueError("ngram range must satisfy 0 < ngram_min <= ngram_max")
    if min_token_len <= 0:
        raise ValueError("min_token_len must be > 0")
    out: list[set[str]] = []
    for abstract in abstracts:
        tokens = [
            token
            for token in TOKEN_RE.findall(str(abstract).lower())
            if len(token) >= min_token_len and token not in DEFAULT_STOPWORDS
        ]
        terms: set[str] = set()
        for width in range(ngram_min, ngram_max + 1):
            if len(tokens) < width:
                continue
            for start in range(len(tokens) - width + 1):
                terms.add("_".join(tokens[start : start + width]))
        out.append(terms)
    return out


def extract_terms_spacy(abstracts: Sequence[str], terms_config: Mapping[str, Any]) -> list[set[str]]:
    """Extract noun chunks with spaCy ``nlp.pipe``.

    Args:
        abstracts: Paper abstracts with shape ``(n_papers,)``.
        terms_config: ``config["terms"]`` including ``spacy_model``,
            ``spacy_n_process``, ``spacy_batch_size``, and
            ``spacy_max_units_per_paper``.

    Returns:
        A list of per-paper noun-chunk term sets with length ``n_papers``.
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy noun-chunk extraction requires the spacy package") from exc

    model_name = str(terms_config["spacy_model"])
    disable = [str(item) for item in as_list(terms_config.get("spacy_disable"))]
    n_process = int(terms_config["spacy_n_process"])
    batch_size = int(terms_config["spacy_batch_size"])
    per_paper_cap = int(terms_config["spacy_max_units_per_paper"])
    if n_process <= 0 or batch_size <= 0 or per_paper_cap <= 0:
        raise ValueError("spaCy n_process, batch_size, and per-paper cap must be > 0")

    nlp = spacy.load(model_name, disable=disable)
    out: list[set[str]] = []
    texts = (str(abstract or "") for abstract in abstracts)
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
        terms: set[str] = set()
        for chunk in doc.noun_chunks:
            term = _normalize_phrase(chunk.text)
            if term:
                terms.add(term)
            if len(terms) >= per_paper_cap:
                break
        out.append(terms)
    return out


def _normalize_phrase(text: str) -> str | None:
    """Normalize and validate a noun phrase."""
    tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in DEFAULT_STOPWORDS
    ]
    if not tokens:
        return None
    return "_".join(tokens)


def extract_terms(abstracts: Sequence[str], config: Mapping[str, Any]) -> list[set[str]]:
    """Select the configured term extractor."""
    terms_config = _terms_config(config)
    if bool(terms_config.get("use_spacy_nounchunks", False)):
        return extract_terms_spacy(abstracts, terms_config)
    return extract_terms_fast(abstracts, terms_config)


def build_year(df: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Build one yearly vocabulary and co-occurrence graph.

    Args:
        df: Work rows with columns ``abstract`` and ``concepts``; shape
            ``(n_papers, n_columns)``.
        config: TSND configuration mapping.

    Returns:
        A dictionary preserving the pilot ``year_{y}.pkl`` graph schema.
    """
    raw_terms_config = config.get("terms", {})
    if isinstance(raw_terms_config, dict) and raw_terms_config.get("unit_source") == "mesh":
        return _build_year_from_mesh(df, raw_terms_config)

    _require_columns(df, ["abstract", "concepts"])
    terms_config = _terms_config(config)
    unit_source = str(terms_config.get("unit_source", "text"))
    if unit_source != "text":
        raise ValueError("terms.unit_source must be 'text' or 'mesh'")
    max_units_per_paper = int(terms_config["max_units_per_paper"])
    if max_units_per_paper <= 0:
        raise ValueError("terms.max_units_per_paper must be > 0")

    abstracts = [str(value or "") for value in df["abstract"].tolist()]
    per_paper_terms = extract_terms(abstracts, config)
    if len(per_paper_terms) != len(df):
        raise RuntimeError("term extractor returned a different number of papers")

    term_df: Counter[str] = Counter()
    for paper_terms in per_paper_terms:
        for unit in paper_terms:
            term_df[unit] += 1

    min_df = int(terms_config["min_df_per_year"])
    top_terms = int(terms_config["top_terms"])
    if min_df <= 0 or top_terms <= 0:
        raise ValueError("terms.min_df_per_year and terms.top_terms must be > 0")
    kept_terms = {unit for unit, count in term_df.items() if count >= min_df}
    if len(kept_terms) > top_terms:
        kept_terms = {unit for unit, _ in term_df.most_common(top_terms)}

    paper_units: list[list[str]] = []
    unit_df: Counter[str] = Counter()
    unknown_df_rank = int(terms_config["unknown_df_rank"])
    for paper_idx, row in enumerate(df.itertuples(index=False)):
        concepts = {
            normalize_unit(str(concept))
            for concept in as_list(getattr(row, "concepts"))
            if str(concept).strip()
        }
        units = concepts | {unit for unit in per_paper_terms[paper_idx] if unit in kept_terms}
        if len(units) > max_units_per_paper:
            # Preserve pilot behavior: keep units with the lowest within-year
            # extracted-term document frequency to bound graph density.
            units = set(
                sorted(units, key=lambda unit: (term_df.get(unit, unknown_df_rank), unit))[
                    :max_units_per_paper
                ]
            )
        ordered = sorted(units)
        paper_units.append(ordered)
        for unit in ordered:
            unit_df[unit] += 1

    vocab = sorted(unit_df)
    uidx = {unit: idx for idx, unit in enumerate(vocab)}
    edges: Counter[tuple[int, int]] = Counter()
    for paper_row_units in paper_units:
        indices = [uidx[unit] for unit in paper_row_units]
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                edge = (left, right) if left < right else (right, left)
                edges[edge] += 1

    graph = {
        "vocab": vocab,
        "uidx": uidx,
        "unit_df": dict(unit_df),
        "paper_units": paper_units,
        "edges": dict(edges),
        "n_papers": len(df),
    }
    validate_year_graph_schema(graph)
    return graph


def _build_year_from_mesh(df: pd.DataFrame, terms_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build one yearly graph directly from per-paper MeSH descriptors."""
    if "mesh" not in df.columns:
        raise ValueError(
            "terms.unit_source='mesh' requires a 'mesh' column in works parquet; "
            "re-ingest with the mesh field"
        )

    max_units_per_paper = int(terms_config["max_units_per_paper"])
    if max_units_per_paper <= 0:
        raise ValueError("terms.max_units_per_paper must be > 0")

    stoplist = {
        str(name).strip().lower()
        for name in terms_config.get("mesh_stoplist", DEFAULT_MESH_CHECK_TAGS)
    }
    per_paper_units = [
        {
            str(unit)
            for unit in as_list(value)
            if str(unit).strip() and str(unit).strip().lower() not in stoplist
        }
        for value in df["mesh"].tolist()
    ]
    unit_df_all: Counter[str] = Counter()
    for units in per_paper_units:
        for unit in units:
            unit_df_all[unit] += 1

    min_df = int(terms_config["min_df_per_year"])
    top_terms = int(terms_config["top_terms"])
    if min_df <= 0 or top_terms <= 0:
        raise ValueError("terms.min_df_per_year and terms.top_terms must be > 0")
    kept_units = {unit for unit, count in unit_df_all.items() if count >= min_df}
    if len(kept_units) > top_terms:
        kept_units = {unit for unit, _ in unit_df_all.most_common(top_terms)}

    paper_units: list[list[str]] = []
    unit_df: Counter[str] = Counter()
    unknown_df_rank = int(terms_config["unknown_df_rank"])
    for units in per_paper_units:
        units = {unit for unit in units if unit in kept_units}
        if len(units) > max_units_per_paper:
            units = set(
                sorted(units, key=lambda unit: (unit_df_all.get(unit, unknown_df_rank), unit))[
                    :max_units_per_paper
                ]
            )
        ordered = sorted(units)
        paper_units.append(ordered)
        for unit in ordered:
            unit_df[unit] += 1

    vocab = sorted(unit_df)
    uidx = {unit: idx for idx, unit in enumerate(vocab)}
    edges: Counter[tuple[int, int]] = Counter()
    for paper_row_units in paper_units:
        indices = [uidx[unit] for unit in paper_row_units]
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                edge = (left, right) if left < right else (right, left)
                edges[edge] += 1

    graph = {
        "vocab": vocab,
        "uidx": uidx,
        "unit_df": dict(unit_df),
        "paper_units": paper_units,
        "edges": dict(edges),
        "n_papers": len(df),
    }
    validate_year_graph_schema(graph)
    return graph


def validate_year_graph_schema(graph: Mapping[str, Any]) -> None:
    """Validate the preserved yearly graph artifact schema."""
    missing = YEAR_GRAPH_KEYS.difference(graph)
    if missing:
        raise ValueError(f"year graph missing keys: {sorted(missing)}")
    if not isinstance(graph["vocab"], list) or not isinstance(graph["uidx"], dict):
        raise TypeError("year graph vocab and uidx have invalid types")
    if not isinstance(graph["paper_units"], list) or not isinstance(graph["edges"], dict):
        raise TypeError("year graph paper_units and edges have invalid types")
    if int(graph["n_papers"]) != len(graph["paper_units"]):
        raise ValueError("year graph n_papers does not match paper_units length")


def write_year_graph(graph: Mapping[str, Any], path: Path, pickle_protocol: int) -> None:
    """Write one trusted local graph artifact with the pilot pkl schema."""
    validate_year_graph_schema(graph)
    if pickle_protocol <= 0:
        raise ValueError("pickle_protocol must be > 0")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(graph), handle, protocol=pickle_protocol)


def build_years(config: Mapping[str, Any], years: Sequence[int], data_dir: Path, out_dir: Path) -> list[Path]:
    """Build and write ``year_{y}.pkl`` graph artifacts for configured years."""
    terms_config = _terms_config(config)
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        src = data_dir / f"works_{year}.parquet"
        if not src.exists():
            LOGGER.warning("missing %s, skip", src)
            continue
        df = pd.read_parquet(src)
        graph = build_year(df, config)
        out_path = out_dir / f"year_{year}.pkl"
        write_year_graph(graph, out_path, int(terms_config["pickle_protocol"]))
        written.append(out_path)
        LOGGER.info(
            "year %d: |V|=%d, |E|=%d, papers=%d",
            year,
            len(graph["vocab"]),
            len(graph["edges"]),
            graph["n_papers"],
        )
    return written


def configured_years(config: Mapping[str, Any], single_year: int | None = None) -> list[int]:
    """Return preprocessing years from config or a CLI override."""
    if single_year is not None:
        return [int(single_year)]
    data_cfg = _data_config(config)
    start = int(data_cfg["year_start"])
    end = int(data_cfg["year_end"])
    if end < start:
        raise ValueError("data.year_end must be >= data.year_start")
    return list(range(start, end + 1))


def data_dir(config: Mapping[str, Any]) -> Path:
    """Return the input parquet directory from config."""
    internal = config.get("_data_dir")
    if internal is not None:
        return Path(str(internal))
    configured = Path(str(_data_config(config)["out_dir"]))
    root = config.get("_root")
    return Path(str(root)) / configured if root is not None else configured


def terms_out_dir(config: Mapping[str, Any]) -> Path:
    """Return the yearly graph output directory from config."""
    if config.get("_art_dir"):
        return Path(str(config["_art_dir"])) / "terms"
    artifacts_cfg = config.get("artifacts", {})
    if isinstance(artifacts_cfg, dict) and artifacts_cfg.get("out_dir"):
        configured = Path(str(artifacts_cfg["out_dir"]))
        root = config.get("_root")
        artifact_root = Path(str(root)) / configured if root is not None else configured
        return artifact_root / "terms"
    return Path("artifacts") / "terms"


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate required dataframe columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing required columns: {missing}")


def _data_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the data section from a config mapping."""
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError("config.data must be a mapping")
    return data_cfg


def _terms_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the terms section from a config mapping."""
    terms_config = config.get("terms", {})
    if not isinstance(terms_config, dict):
        raise TypeError("config.terms must be a mapping")
    return terms_config


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for yearly vocabulary preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--n-process", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_yaml_config(args.config)
    if args.n_process is not None:
        terms_config = dict(_terms_config(config))
        terms_config["spacy_n_process"] = args.n_process
        config["terms"] = terms_config
    years = configured_years(config, args.year)
    build_years(config, years, data_dir(config), terms_out_dir(config))


if __name__ == "__main__":
    main()
