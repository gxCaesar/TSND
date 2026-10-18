"""Classify retained raw and DF-specific graph histories with frozen labels."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tsnd.journal.cohort import (
    classify_cohort,
    front_counts,
    load_graph_history,
    raw_mesh_graph,
    summarize_cohort,
)


def main() -> int:
    """Write per-condition membership tables and complete candidate-count summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--terms-dir", required=True, type=Path, help="Frozen DF20 graph directory.")
    parser.add_argument("--condition", action="append", default=[], metavar="NAME=TERMS_DIR")
    parser.add_argument("--min-df", action="append", default=[], metavar="NAME=VALUE",
                        help="Required for condition names that do not contain df10, df20, or df30.")
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2011, 2017)))
    parser.add_argument("--start-year", type=int, default=2010)
    args = parser.parse_args()
    conditions = {"df20": args.terms_dir}
    for spec in args.condition:
        name, path = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in conditions:
            parser.error(f"Invalid or duplicate condition name: {name}")
        conditions[name] = Path(path)
    minimums = {name: int(value) for name, value in (spec.split("=", 1) for spec in args.min_df)}
    for name in conditions:
        match = re.search(r"df(10|20|30)(?:\D|$)", name, flags=re.IGNORECASE)
        if name not in minimums and match:
            minimums[name] = int(match.group(1))
        if name not in minimums:
            parser.error(f"Provide --min-df {name}=VALUE.")
    years = sorted(set(args.years))
    if not years or years[0] <= args.start_year:
        parser.error("Every cohort year must be after --start-year.")
    targets = [args.output_dir / f"classified_{name}_{year}.parquet" for name in conditions for year in years]
    targets += [args.output_dir / "summary.json", args.output_dir / "summary.csv"]
    if any(path.exists() for path in targets):
        parser.error("Output files already exist; select a fresh --output-dir.")
    raw_graphs = {
        year: raw_mesh_graph(pd.read_parquet(args.data_dir / f"works_{year}.parquet", columns=["mesh"]))
        for year in range(args.start_year, years[-1] + 1)
    }
    graphs = {name: load_graph_history(path, args.start_year, years[-1]) for name, path in conditions.items()}
    raw_counts = {year: front_counts(graph) for year, graph in raw_graphs.items()}
    graph_counts = {name: {year: front_counts(graph) for year, graph in history.items()}
                    for name, history in graphs.items()}
    summaries = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        labels = pd.read_parquet(args.labels_dir / f"labels_{year}.parquet")
        classified = {
            name: classify_cohort(raw_graphs, history, year, labels,
                                  start_year=args.start_year, min_df=minimums[name],
                                  raw_counts=raw_counts, graph_counts=graph_counts[name])
            for name, history in graphs.items()
        }
        baseline_ids = set(zip(classified["df20"]["kind"], classified["df20"]["key"]))
        for name, frame in classified.items():
            frame["in_df20_candidates"] = [
                (kind, key) in baseline_ids for kind, key in zip(frame.kind, frame.key)
            ]
            frame.to_parquet(args.output_dir / f"classified_{name}_{year}.parquet", index=False)
            summaries.extend(summarize_cohort(frame, classified["df20"], name, year))
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    payload = {"start_year": args.start_year, "years": years, "outcome_labels": "original_frozen",
               "raw_history": "retained papers before annual DF and per-paper unit filtering",
               "rows": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary_rows": len(summaries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
