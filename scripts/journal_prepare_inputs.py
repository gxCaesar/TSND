"""Rebuild one prespecified observation-corpus condition without changing outcomes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from tsnd.cli import _deep_merge, _load_yaml, stage_decompose
from tsnd.data.preprocessing import build_years, default_config
from tsnd.models.alignment import run_alignment
from tsnd.models.embeddings import YearlySgnsEmbedder


def main() -> int:
    """Build the complete historical input chain for a fresh condition directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--year-end", type=int, default=2016)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _deep_merge(default_config(), _load_yaml(str(args.config)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg["data"]["year_end"] = args.year_end
    cfg["data"]["out_dir"] = str(args.data_dir.resolve())
    cfg["terms"]["min_df_per_year"] = args.min_df
    cfg["seed_list"] = args.seeds
    cfg["_art_dir"] = str(args.output_dir.resolve())
    cfg["_data_dir"] = str(args.data_dir.resolve())
    cfg["artifacts"]["out_dir"] = str(args.output_dir.resolve())
    cfg["run_tag"] = args.condition
    workers = int(cfg.get("compute", {}).get("workers", 8))
    years = list(range(int(cfg["data"]["year_start"]), args.year_end + 1))
    (args.output_dir / "condition.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    terms, embed = args.output_dir / "terms", args.output_dir / "embed"
    paths = build_years(cfg, years, args.data_dir, terms)
    if len(paths) != len(years):
        raise RuntimeError("a required historical graph was skipped")
    YearlySgnsEmbedder(cfg, terms, embed, workers=workers).run()
    run_alignment(cfg, args.config)
    stage_decompose(cfg, terms, embed, args.output_dir / "align", args.output_dir / "decompose")
    expected = [
        args.output_dir / "decompose" / f"decomp_{year}_s{seed}.pkl"
        for year in years[1:] for seed in args.seeds
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"condition has missing native decompositions: {missing}")
    summary = {
        "condition": args.condition, "min_df": args.min_df, "years": years, "seeds": args.seeds,
        "sgns_workers": workers, "expected_decompositions": len(expected), "missing_decompositions": 0,
        "outcomes_modified": False, "status": "complete",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
