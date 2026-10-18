"""Staged pipeline entry point: ``python -m tsnd.cli <stage> --config <cfg>``.

Stages, in order:
    data       OpenAlex ingest                  -> tsnd.data.ingest.main
    preprocess vocabulary + co-occurrence graph  -> tsnd.data.preprocessing.main
    embed      yearly SGNS embeddings            -> tsnd.models.embeddings.main
    align      stable-anchor Procrustes          -> tsnd.models.alignment.main
    decompose  LE / NS / RC per cohort and seed  -> this module + tsnd.models.ours
    label      leakage-safe uptake breadth       -> this module + tsnd.evaluation.labels
    forecast   calibrated LightGBM per group     -> this module + tsnd.training.forecaster
    axioms     axiom-coverage table              -> this module + tsnd.evaluation
    all        every stage above in order

The first four stages reuse each module's own argv main. The later stages are wired
here against the validated decomposition, labelling and forecasting functions.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from tsnd.data.ingest import main as ingest_main
from tsnd.data.preprocessing import main as preprocessing_main
from tsnd.evaluation.a10_harness import run_identifiability
from tsnd.evaluation.axioms import NoveltyMetric, run_axiom_suite
from tsnd.evaluation.focal_sampler import sample_focal_papers
from tsnd.evaluation.labels import YearUsage, build_inverted_index, label_cohort
from tsnd.models.alignment import main as alignment_main
from tsnd.models.baseline_registry import build_baseline_features, forecast_feature_groups
from tsnd.models.embeddings import main as embeddings_main
from tsnd.models.ours import assemble_features, compute_decomposition
from tsnd.models.specter_embed import main as specter_main
from tsnd.training.forecaster import run_from_provider
from tsnd.utils.logging_config import configure_logging


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings merge key by key; every other value, including lists, is replaced wholesale so
    that a child config can shorten a list rather than only extend it.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load the orchestration config as a plain dict, resolving ``inherits:``.

    A config may declare ``inherits: <file>``, resolved relative to its own directory. The
    parent is loaded first and the child is deep-merged on top, so a child overrides only the
    keys it names. Chains are followed and cycles raise rather than loop.

    cli only needs to read config values to dispatch stages; each stage module validates
    its own slice of the config, so cli does not impose the strict pydantic model here.
    """
    resolved = os.path.abspath(path)
    if resolved in _seen:
        chain = " -> ".join(os.path.basename(p) for p in (*_seen, resolved))
        raise ValueError(f"circular config inheritance: {chain}")
    with open(resolved) as handle:
        cfg = yaml.safe_load(handle) or {}
    parent_name = cfg.pop("inherits", None)
    if parent_name is None:
        return cfg
    parent_path = os.path.join(os.path.dirname(resolved), str(parent_name))
    parent = _load_yaml(parent_path, (*_seen, resolved))
    return _deep_merge(parent, cfg)

logger = logging.getLogger(__name__)

STAGES = ("data", "preprocess", "embed", "align", "decompose", "label", "specter", "forecast", "axioms")


def _cfg_get(cfg: object, *path: str, default: Any = None) -> Any:
    """Read a nested key from a mapping- or object-style config, with a default."""
    cur: Any = cfg
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def _dirs(cfg: object, config_path: str) -> tuple[Path, Path]:
    """Resolve the configured data and artifact roots."""
    root = Path(config_path).resolve().parent.parent
    tag = _cfg_get(cfg, "run_tag", default="main")
    data_dir = root / str(_cfg_get(cfg, "data", "out_dir", default="data"))
    artifact_name = _cfg_get(cfg, "artifacts", "out_dir", default=f"artifacts_{tag}")
    art_dir = root / str(artifact_name)
    art_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, art_dir


def _years(cfg: object) -> tuple[int, int]:
    return int(_cfg_get(cfg, "data", "year_start", default=2015)), int(
        _cfg_get(cfg, "data", "year_end", default=2022))


def _seeds(cfg: object) -> list[int]:
    return list(_cfg_get(cfg, "seed_list", default=[0, 1]))


def _load_graph(terms_dir: Path, year: int) -> dict[str, Any]:
    with open(terms_dir / f"year_{year}.pkl", "rb") as handle:
        return cast(dict[str, Any], pickle.load(handle))  # noqa: S301  trusted local artifact


def _prior_state(terms_dir: Path, t: int, start_year: int | None) -> tuple[set[str], set[tuple[str, str]]]:
    """Units and pairs seen before year ``t``.

    With ``start_year=None`` this is the ADJACENT-YEAR rule: only year ``t-1`` counts, so a pair
    that lapsed for a year and reappeared is treated as new. With a ``start_year`` it is the
    CUMULATIVE-HISTORY rule: every graph from ``start_year`` to ``t-1`` counts. Cumulative history
    is bounded below by the corpus start, so "first-ever" here means first observed since
    ``start_year``. The reported history-filtered forecast instead preserves labels assigned to
    the adjacent-year cohort; see ``scripts/e1_firstever.py``.
    """
    years = [t - 1] if start_year is None else list(range(start_year, t))
    units: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for year in years:
        graph = _load_graph(terms_dir, year)
        vocab = graph["vocab"]
        units.update(vocab)
        for (i, j) in graph["edges"]:
            u, v = vocab[i], vocab[j]
            pairs.add((u, v) if u < v else (v, u))
    return units, pairs


def _fronts_born(terms_dir: Path, t: int, start_year: int | None = None) -> pd.DataFrame:
    """Terms and pairs entering at year ``t``, against adjacent-year or cumulative history.

    ``start_year=None`` reproduces the adjacent-year candidate set. Passing the corpus start year
    yields a cumulative-history set and then labels that set separately. The reported
    history-filtered forecast uses the same cumulative-history membership test as a post-filter
    on adjacent-year labels. The two candidate sets differ exactly by pairs that lapsed and
    reappeared.
    """
    g_t = _load_graph(terms_dir, t)
    prior_units, prior_pairs = _prior_state(terms_dir, t, start_year)
    rows: list[dict[str, Any]] = []
    for unit in g_t["vocab"]:
        if unit not in prior_units:
            rows.append({"kind": "term", "key": unit, "t": t})
    for (i, j) in g_t["edges"]:
        u, v = g_t["vocab"][i], g_t["vocab"][j]
        key = (u, v) if u < v else (v, u)
        if key not in prior_pairs:
            rows.append({"kind": "pair", "key": f"{key[0]}|{key[1]}", "t": t})
    return pd.DataFrame(rows)


def _to_str_list(value: Any) -> list[str]:
    """Coerce a parquet list-cell (numpy array, list, or None) to a flat list of strings."""
    if value is None:
        return []
    return [str(item) for item in value]


def _build_usage(terms_dir: Path, data_dir: Path, year: int) -> YearUsage:
    graph = _load_graph(terms_dir, year)
    works = pd.read_parquet(data_dir / f"works_{year}.parquet", columns=["venue_id", "subfield_ids"])
    return YearUsage(
        inverted=build_inverted_index(graph["paper_units"]),
        venue_of=works["venue_id"].tolist(),
        subfields_of=[_to_str_list(x) for x in works["subfield_ids"].tolist()],
        institutions_of=[[] for _ in range(len(works))],
    )


def stage_decompose(cfg: object, terms_dir: Path, emb_dir: Path, align_dir: Path, out_dir: Path) -> None:
    """Write LE/NS/RC decomposition artifacts for every configured year and seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = _years(cfg)
    for t in range(y0 + 1, y1 + 1):
        for seed in _seeds(cfg):
            try:
                decomposition = compute_decomposition(t, seed, emb_dir, terms_dir, align_dir)
            except FileNotFoundError:
                logger.warning("decompose: missing alignment for t=%d seed=%d, skipping", t, seed)
                continue
            with open(out_dir / f"decomp_{t}_s{seed}.pkl", "wb") as handle:
                pickle.dump(decomposition, handle, protocol=4)
            logger.info("decompose t=%d seed=%d: %d new terms, %d new pairs",
                        t, seed, decomposition["n_new_terms"], decomposition["n_new_pairs"])


def stage_label(cfg: object, terms_dir: Path, data_dir: Path, out_dir: Path) -> None:
    """Construct and write forward-uptake labels for configured cohort years."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = _years(cfg)
    horizon = int(_cfg_get(cfg, "label", "horizon", default=2))
    channels = tuple(_cfg_get(cfg, "label", "breadth_channels", default=["venues", "subfields"]))
    channels = tuple(c for c in channels if c != "institutions")  # institutions deferred (pilot)
    quantile = float(_cfg_get(cfg, "label", "positive_quantile", default=0.90))
    # candidate_rule: "adjacent" (default, absent from G_{t-1}) or "first_ever" (absent from
    # every graph from the corpus start year onward). The released runs used "adjacent" and
    # then filtered the labelled tables with scripts/e1_firstever.py. Direct "first_ever"
    # constructs the cumulative-history cohort before thresholding, so it is a distinct label
    # policy and is not the reported history-filtered result.
    rule = str(_cfg_get(cfg, "label", "candidate_rule", default="adjacent"))
    if rule not in ("adjacent", "first_ever"):
        raise ValueError(f"label.candidate_rule must be 'adjacent' or 'first_ever', got {rule!r}")
    start_year = y0 if rule == "first_ever" else None
    for t in range(y0 + 1, y1 - horizon + 1):
        usages = [_build_usage(terms_dir, data_dir, y) for y in range(t + 1, t + horizon + 1)
                  if (terms_dir / f"year_{y}.pkl").exists()]
        labelled = label_cohort(_fronts_born(terms_dir, t, start_year), usages, channels, quantile)
        labelled.to_parquet(out_dir / f"labels_{t}.parquet", index=False)


# Only B6 (Shibayama-Yin) reads the per-seed SGNS embeddings; B2/B3/B4/B5/B7/B8 are
# seed-independent (graph, works text, or SPECTER paper vectors), so their feature blocks
# are built once per cohort year and reused across seeds instead of rebuilt 3x.
_SEED_DEPENDENT_BASELINES = frozenset({"b6"})


def _merge_block(wide: pd.DataFrame, block: pd.DataFrame, label: str) -> pd.DataFrame:
    """Left-merge a baseline block on ``key``, dropping columns already present."""
    new_columns = [c for c in block.columns if c != "key" and c not in wide.columns]
    if not new_columns:
        return wide
    merged = wide.merge(block.loc[:, ["key", *new_columns]], on="key", how="left", validate="m:1")
    if len(merged) != len(wide):
        raise AssertionError(f"baseline {label} merge changed row count: {len(wide)} -> {len(merged)}")
    return merged


def _provider(cfg: Mapping[str, Any], terms_dir: Path, emb_dir: Path, decomp_dir: Path, label_dir: Path,
              art_dir: Path, data_dir: Path, baselines: Sequence[str],
              train_years: Sequence[int], test_year: int) -> Callable[
                  [int], tuple[pd.DataFrame, pd.DataFrame]
              ]:
    """Build a seed-specific (train, test) table provider.

    Each table is the per-seed TSND feature table (``assemble_features``) left-merged with
    the requested baseline feature blocks by ``key``. The seed-independent baseline block is
    built once per cohort year and cached, so a 3-seed run does not rebuild B2/B3/B4/B7/B8
    three times. Only the per-seed B6 block is rebuilt per seed.
    """
    indep_names = [b for b in baselines if b not in _SEED_DEPENDENT_BASELINES]
    dep_names = [b for b in baselines if b in _SEED_DEPENDENT_BASELINES]
    indep_cache: dict[int, pd.DataFrame] = {}
    group_columns = {c for cols in forecast_feature_groups(baselines).values() for c in cols}

    def indep_block(t: int, keys: list[str]) -> pd.DataFrame:
        if t not in indep_cache:
            block = pd.DataFrame({"key": keys})
            for name in indep_names:
                feat = build_baseline_features(name, t, cfg, art_dir, data_dir, terms_dir, emb_dir,
                                               list(train_years), seed=0)
                block = _merge_block(block, feat, name)
            indep_cache[t] = block
        return indep_cache[t]

    def provide(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        def table(t: int) -> pd.DataFrame:
            labels = pd.read_parquet(label_dir / f"labels_{t}.parquet")
            with open(decomp_dir / f"decomp_{t}_s{seed}.pkl", "rb") as handle:
                decomposition = pickle.load(handle)  # noqa: S301  trusted local artifact
            wide = assemble_features(labels, decomposition, terms_dir, t)
            if not baselines:
                return wide
            wide = _merge_block(wide, indep_block(t, wide["key"].tolist()), "seed_independent")
            for name in dep_names:
                feat = build_baseline_features(name, t, cfg, art_dir, data_dir, terms_dir, emb_dir,
                                               list(train_years), seed)
                wide = _merge_block(wide, feat, name)
            missing = sorted(c for c in group_columns if c not in wide.columns)
            if missing:
                raise AssertionError(f"forecast feature columns missing from wide table: {missing}")
            return wide
        train = pd.concat([table(t) for t in train_years], ignore_index=True)
        return train, table(test_year)
    return provide


def stage_forecast(cfg: Mapping[str, Any], terms_dir: Path, emb_dir: Path, decomp_dir: Path,
                   label_dir: Path, art_dir: Path, data_dir: Path) -> None:
    """Fit configured forecast groups and write their evaluation record."""
    import json

    y0, y1 = _years(cfg)
    horizon = int(_cfg_get(cfg, "label", "horizon", default=2))
    test_year = y1 - horizon
    train_years = list(range(y0 + 1, test_year))
    baselines = [str(b) for b in (_cfg_get(cfg, "forecaster", "baselines", default=[]) or [])]
    if baselines:
        groups = forecast_feature_groups(baselines)
        # Component ablations (ABL-1 drop-LE, ABL-2 drop-RC): gate the TSND feature group
        # on features.include_le / features.include_rc so the forecaster isolates each
        # component's contribution. Both default True, so full runs are unchanged. Applied
        # before the combined groups below, so a tsnd+structure group inherits the reduced
        # TSND columns. Dropping RC removes its raw prior-proximity input too.
        if "tsnd" in groups:
            if not bool(_cfg_get(cfg, "features", "include_le", default=True)):
                groups["tsnd"] = [c for c in groups["tsnd"] if c != "le"]
            if not bool(_cfg_get(cfg, "features", "include_rc", default=True)):
                groups["tsnd"] = [c for c in groups["tsnd"] if c not in ("rc", "prox")]
            logger.info("forecast: tsnd group columns: %s", groups["tsnd"])
        # Optional incremental-value ablation: union member groups' columns (deduped,
        # order-preserving) into a combined group so we can test e.g. tsnd+structure
        # vs structure-alone without changing any base group's definition.
        for spec in (_cfg_get(cfg, "forecaster", "combined_groups", default=[]) or []):
            name = str(spec["name"])
            cols: list[str] = []
            for member in [str(m) for m in spec["members"]]:
                for col in groups.get(member, []):
                    if col not in cols:
                        cols.append(col)
            if cols:
                groups[name] = cols
        if isinstance(cfg, dict):
            cfg.setdefault("forecaster", {})["feature_groups"] = groups
        logger.info("forecast: %d feature groups: %s", len(groups), sorted(groups))
    provider = _provider(cfg, terms_dir, emb_dir, decomp_dir, label_dir, art_dir, data_dir,
                         baselines, train_years, test_year)
    results = run_from_provider(provider, cfg)
    with open(art_dir / "forecast.json", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("forecast: wrote %s", art_dir / "forecast.json")


def stage_specter(cfg: object, config_path: str, art_dir: Path) -> None:
    """Precompute SPECTER2 paper embeddings, only when B5 is a requested baseline."""
    baselines = [str(b) for b in (_cfg_get(cfg, "forecaster", "baselines", default=[]) or [])]
    if "b5" not in baselines:
        logger.info("specter: b5 not in forecaster.baselines, skipping SPECTER2 precompute")
        return
    logger.info("specter: precomputing SPECTER2 embeddings for B5 -> %s", art_dir / "specter")
    specter_main(["--config", config_path])


def _first_domain(cfg: object) -> str:
    domains = _cfg_get(cfg, "data", "domains", default=[])
    if isinstance(domains, list) and domains:
        first = domains[0]
        if isinstance(first, dict):
            return str(first.get("name") or first.get("id") or "ai")
        return str(first)
    return "ai"


def _axiom_embedder(cfg: Mapping[str, Any]) -> Any:
    """Build the text embedder for the B9 density axiom metrics (BGE-M3 or a CPU smoke)."""
    kind = str(_cfg_get(cfg, "axioms", "embedder", default="bge"))
    if kind == "hashing":
        from tsnd.models.bge_embedder import HashingEmbedder
        return HashingEmbedder(int(_cfg_get(cfg, "axioms", "hashing_dim", default=256)))
    from tsnd.models.bge_embedder import BgeEmbedder
    return BgeEmbedder.from_config(cfg)


def _pool_axiom_table(
    cfg: Mapping[str, Any],
    art_dir: Path,
    data_dir: Path,
    focal_year: int,
    seed: int,
) -> dict[str, Any]:
    """Run A1-A9 over the B9 density metrics on one focal sample. A5/A6 stay gated off."""
    from tsnd.models.baselines.density_metrics import (
        FastTextLofMetric,
        RndMetric,
        SemNovelMetric,
        YinMetric,
    )

    y0, _ = _years(cfg)
    focal_domain = str(_cfg_get(cfg, "axioms", "focal_domain", default=_first_domain(cfg)))
    focal_papers, future_docs = sample_focal_papers(
        data_dir,
        focal_domain=focal_domain,
        focal_year=int(focal_year),
        n_focal=int(_cfg_get(cfg, "axioms", "n_focal", default=100)),
        base_pool_size=int(_cfg_get(cfg, "axioms", "base_pool_size", default=1500)),
        distant_pool_size=int(_cfg_get(cfg, "axioms", "distant_pool_size", default=1000)),
        future_pool_size=int(_cfg_get(cfg, "axioms", "future_pool_size", default=500)),
        year_start=int(y0),
        seed=int(seed),
    )
    if not focal_papers:
        logger.warning("axioms: no focal papers sampled for domain=%s year=%d; skipping A1-A9",
                       focal_domain, focal_year)
        return {}
    embedder = _axiom_embedder(cfg)
    from tsnd.evaluation.tsnd_score import (
        TemporalPoolFilterMetric,
        TemporalRecombinationMetric,
        build_year_map,
    )
    future_for_map = future_docs[0] if future_docs else ()
    year_map = build_year_map(
        [fp.text for fp in focal_papers], int(focal_year),
        focal_papers[0].base_pool, focal_papers[0].pool_years, future_for_map,
    )
    metrics: dict[str, NoveltyMetric] = {
        "TSND_RC": TemporalRecombinationMetric(embedder, year_map),
        "RND": RndMetric(embedder),
        "SemNovel": SemNovelMetric(embedder),
        "Yin": YinMetric(embedder),
        "FastTextLOF": FastTextLofMetric(embedder),
    }
    if bool(_cfg_get(cfg, "axioms", "include_temporalized_baselines", default=False)):
        # Temporalized density baselines (reviewer M1): the same D<=t cutoff TSND uses,
        # applied to each static density metric, so they satisfy A9 by construction while
        # still failing the other axioms and offering no decomposition. Default off so the
        # canonical axiom table is unchanged.
        metrics.update({
            "SemNovel-Dt": TemporalPoolFilterMetric(SemNovelMetric(embedder), year_map),
            "Yin-Dt": TemporalPoolFilterMetric(YinMetric(embedder), year_map),
            "FastTextLOF-Dt": TemporalPoolFilterMetric(FastTextLofMetric(embedder), year_map),
        })
    table: dict[str, Any] = {}
    for name, metric in metrics.items():
        report = run_axiom_suite(metric, focal_papers, name, future_docs_by_focal=future_docs, seed=int(seed))
        table[name] = {"pass_rate": report.pass_rate, "n_eligible": report.n_eligible,
                       "average_pass_rate": report.average_pass_rate}
        logger.info("axioms A1-A9 %s: avg pass %.3f", name, report.average_pass_rate)
    if bool(_cfg_get(cfg, "axioms", "run_llm_judge", default=False)):
        logger.warning("axioms: B10 LLM judge requested but is gated off (needs an API backend; "
                       "see docs/AXIOMS.md). Skipping.")
    return {"focal_domain": focal_domain, "focal_year": int(focal_year), "metrics": table}


def stage_axioms(cfg: Mapping[str, Any], art_dir: Path, data_dir: Path) -> None:
    """Write the axiom-coverage table: A10 identifiability (ours) + A1-A9 (B9 density metrics)."""
    import json

    seeds = _seeds(cfg)
    y0, y1 = _years(cfg)
    horizon = int(_cfg_get(cfg, "label", "horizon", default=2))
    focal_year = int(_cfg_get(cfg, "axioms", "focal_year", default=y1 - horizon))

    a10_records: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            a10_records.append(run_identifiability(focal_year, seed, art_dir, cfg))
        except FileNotFoundError as exc:
            logger.warning("axioms A10: missing artifact for t=%d seed=%d: %s", focal_year, seed, exc)
    a10_scores = [r["identifiability"] for r in a10_records
                  if not r.get("skipped") and np.isfinite(r.get("identifiability", float("nan")))]
    a10_mean = float(np.mean(a10_scores)) if a10_scores else float("nan")
    a10_std = float(np.std(a10_scores)) if a10_scores else float("nan")

    pool_table: dict[str, Any] = {}
    if bool(_cfg_get(cfg, "axioms", "run_pool_axioms", default=True)):
        pool_table = _pool_axiom_table(cfg, art_dir, data_dir, focal_year, seeds[0])

    out = {
        "a10_decomposition_identifiability": {
            "mean": a10_mean, "std": a10_std, "n_seeds": len(a10_scores), "per_seed": a10_records,
        },
        "a1_a9_pool_axioms": pool_table,
    }
    with open(art_dir / "axioms.json", "w") as handle:
        json.dump(out, handle, indent=2, default=str)
    logger.info("axioms: A10 identifiability mean=%.3f std=%.3f over %d seeds; wrote %s",
                a10_mean, a10_std, len(a10_scores), art_dir / "axioms.json")


def _run_module_stage(stage: str, config_path: str) -> None:
    cfg = _load_yaml(config_path)
    with open(config_path) as handle:
        raw = yaml.safe_load(handle) or {}
    # These four stages reopen the config file themselves, so a child declaring ``inherits:``
    # would reach them unmerged and fail or silently run on defaults. Hand them a materialized
    # copy of the merged mapping instead, written beside the original so that every
    # directory-relative path in the config still resolves the same way. A self-contained
    # config compares equal here and is passed through untouched.
    root = Path(config_path).resolve().parent.parent
    data_dir, art_dir = _dirs(cfg, config_path)
    cfg["_root"] = str(root)
    cfg["_data_dir"] = str(data_dir)
    cfg["_art_dir"] = str(art_dir)
    merged_path = None
    if raw != cfg:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".merged.yaml", dir=os.path.dirname(os.path.abspath(config_path)),
            delete=False,
        ) as merged:
            yaml.safe_dump(cfg, merged, sort_keys=False)
            merged_path = merged.name
    argv = ["--config", merged_path or config_path]
    if stage == "embed":
        argv += ["--workers", str(_cfg_get(cfg, "compute", "workers", default=8))]
    try:
        {"data": ingest_main, "preprocess": preprocessing_main,
         "embed": embeddings_main, "align": alignment_main}[stage](argv)
    finally:
        if merged_path is not None:
            os.unlink(merged_path)


def run_stage(stage: str, config_path: str) -> None:
    """Run one named pipeline stage from a YAML configuration."""
    cfg = _load_yaml(config_path)
    if stage in ("data", "preprocess", "embed", "align"):
        _run_module_stage(stage, config_path)
        return
    data_dir, art_dir = _dirs(cfg, config_path)
    terms_dir, emb_dir, align_dir = art_dir / "terms", art_dir / "embed", art_dir / "align"
    decomp_dir, label_dir = art_dir / "decompose", art_dir / "labels"
    if stage == "decompose":
        stage_decompose(cfg, terms_dir, emb_dir, align_dir, decomp_dir)
    elif stage == "label":
        stage_label(cfg, terms_dir, data_dir, label_dir)
    elif stage == "specter":
        stage_specter(cfg, config_path, art_dir)
    elif stage == "forecast":
        label_override = _cfg_get(cfg, "forecaster", "label_dir", default=None)
        if label_override is not None:
            root = Path(config_path).resolve().parent.parent
            label_dir = root / str(label_override)
        stage_forecast(cfg, terms_dir, emb_dir, decomp_dir, label_dir, art_dir, data_dir)
    elif stage == "axioms":
        stage_axioms(cfg, art_dir, data_dir)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured pipeline stage, or all stages in dependency order."""
    parser = argparse.ArgumentParser(prog="tsnd.cli", description="TSND staged pipeline")
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    configure_logging("INFO")
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        logger.info("=== stage: %s ===", stage)
        run_stage(stage, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
