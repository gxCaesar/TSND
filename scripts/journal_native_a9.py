"""Audit native graph/SGNS/profile invariance to additional future raw files."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tsnd.cli import _load_yaml
from tsnd.journal.native_a9 import run_native_a9


def main() -> int:
    """Run the isolated raw-input audit and return failure if any comparison fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_native_a9(_load_yaml(str(args.config)), args.data_dir, args.output_dir,
                           year=args.year, seeds=args.seeds)
    print(json.dumps({"passed": result["passed"], "summary": str(args.output_dir / "summary.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
