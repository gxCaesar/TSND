"""Acquire the current complete oncology reference with an injected OpenAlex key."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from tsnd.journal.reference_corpus import ReferenceAccessError, collect_reference


def main() -> int:
    """Run one non-overlapping, resumable reference acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2010, 2020)))
    parser.add_argument("--api-key-env", default="OPENALEX_API_KEY")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another collector is already using this output directory")
        try:
            result = collect_reference(args.output_dir, args.years, args.api_key_env)
        except ReferenceAccessError as exc:
            print(str(exc))
            return 3
    print(json.dumps({"complete": result["complete"], "years": len(result["years"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
