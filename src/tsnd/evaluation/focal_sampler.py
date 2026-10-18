"""Sample OpenAlex focal papers for the pool-relative axiom suite."""
from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsnd.evaluation.axioms import FocalPaper

logger = logging.getLogger(__name__)

__all__ = ["load_domain_year_abstracts", "sample_focal_papers"]

_WORKS_FILE_RE = re.compile(r"^works_(\d+)\.parquet$")
_PARQUET_COLUMNS = ("abstract", "year", "domain")
_A7_A8_MIN_BASE_POOL = 600


@dataclass(frozen=True)
class _PaperRecord:
    abstract: str
    year: int


def load_domain_year_abstracts(data_dir: Path, year: int, domain: str) -> list[str]:
    """Load cleaned abstracts for one OpenAlex domain-year slice.

    Args:
        data_dir: Directory containing ``works_{year}.parquet`` files.
        year: Publication year to select.
        domain: Domain label to select from the parquet ``domain`` column.

    Returns:
        Nonempty, stripped abstracts with shape ``(n_docs,)``. Missing year files
        return an empty list and emit a warning.
    """
    path = Path(data_dir) / f"works_{year}.parquet"
    if not path.exists():
        logger.warning("OpenAlex parquet file not found: %s", path)
        return []

    records = _read_records(
        [path],
        domain=domain,
        min_year=year,
        max_year_exclusive=year + 1,
    )
    return [record.abstract for record in records]


def sample_focal_papers(
    data_dir: Path,
    *,
    focal_domain: str,
    focal_year: int,
    n_focal: int = 100,
    base_pool_size: int = 1500,
    distant_pool_size: int = 1000,
    future_pool_size: int = 500,
    year_start: int | None = None,
    seed: int = 0,
) -> tuple[list[FocalPaper], list[tuple[str, ...]]]:
    """Sample focal papers and shared reference pools for A1-A9 evaluation.

    The sampled base pool, distant pool, and future pool are shared by all focal
    papers. This is cheaper than per-focal resampling and keeps pool-relative
    metric variation attributable to the focal text rather than to pool noise.
    ``future_docs_by_focal`` repeats the same immutable tuple once per focal paper
    so it remains index-aligned with ``focal_papers`` for
    ``tsnd.evaluation.axioms.run_axiom_suite``.

    Citations are intentionally left empty for every focal paper because the
    OpenAlex referenced-works pull is not part of this sampler yet. A5 and A6
    therefore auto-skip in the axiom suite by design.

    Args:
        data_dir: Directory containing per-year ``works_{year}.parquet`` files.
        focal_domain: Domain label of the focal papers and same-domain pools.
        focal_year: Publication year of the focal papers.
        n_focal: Maximum number of focal abstracts to sample.
        base_pool_size: Maximum number of same-domain, pre-focal-year abstracts
            to include in ``FocalPaper.base_pool`` with shape ``(n_base,)``.
        distant_pool_size: Maximum number of other-domain abstracts to include in
            ``FocalPaper.distant_pool`` with shape ``(n_distant,)``.
        future_pool_size: Maximum number of same-domain, post-focal-year abstracts
            to include in each A9 future pool with shape ``(n_future,)``.
        year_start: Optional inclusive lower bound for base-pool publication years.
        seed: Seed for the single ``np.random.default_rng`` used by all sampling.

    Returns:
        A tuple ``(focal_papers, future_docs_by_focal)``. ``focal_papers`` has
        shape ``(n_sampled_focal,)``. ``future_docs_by_focal`` has the same length,
        and each entry is the shared future-doc tuple used for A9.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist or contains no work files.
        ValueError: If a requested sample size is negative, or ``year_start`` is
            later than ``focal_year``.
    """
    _validate_inputs(
        n_focal=n_focal,
        base_pool_size=base_pool_size,
        distant_pool_size=distant_pool_size,
        future_pool_size=future_pool_size,
        year_start=year_start,
        focal_year=focal_year,
    )

    rng = np.random.default_rng(seed)
    data_path = Path(data_dir)
    year_files = _available_work_files(data_path)

    focal_records = _read_records(
        _paths_for_range(year_files, min_year=focal_year, max_year_exclusive=focal_year + 1),
        domain=focal_domain,
        min_year=focal_year,
        max_year_exclusive=focal_year + 1,
    )
    base_records = _read_records(
        _paths_for_range(year_files, min_year=year_start, max_year_exclusive=focal_year),
        domain=focal_domain,
        min_year=year_start,
        max_year_exclusive=focal_year,
    )
    distant_records = _read_records(
        _paths_for_range(year_files, max_year_exclusive=focal_year + 1),
        excluded_domain=focal_domain,
        max_year_exclusive=focal_year + 1,
    )
    future_records = _read_records(
        _paths_for_range(year_files, min_year=focal_year + 1),
        domain=focal_domain,
        min_year=focal_year + 1,
    )

    sampled_focal = _sample_records(focal_records, n_focal, rng)
    sampled_base = _sample_records(base_records, base_pool_size, rng)
    sampled_distant = _sample_records(distant_records, distant_pool_size, rng)
    sampled_future = _sample_records(future_records, future_pool_size, rng)

    base_pool = tuple(record.abstract for record in sampled_base)
    pool_years = tuple(record.year for record in sampled_base)
    distant_pool = tuple(record.abstract for record in sampled_distant)
    future_docs = tuple(record.abstract for record in sampled_future)

    if len(base_pool) < _A7_A8_MIN_BASE_POOL:
        logger.warning(
            "sampled base_pool has %d papers; A7/A8 require at least %d and will be ineligible",
            len(base_pool),
            _A7_A8_MIN_BASE_POOL,
        )
    if not distant_pool:
        logger.warning("no other-domain rows found for focal_domain=%s; A4 will be ineligible", focal_domain)
    if not future_docs:
        logger.warning("no future same-domain rows found after year %d; A9 will be ineligible", focal_year)

    focal_papers = [
        FocalPaper(
            text=record.abstract,
            base_pool=base_pool,
            cited=(),
            distant_pool=distant_pool,
            year=focal_year,
            pool_years=pool_years,
        )
        for record in sampled_focal
    ]
    future_docs_by_focal = [future_docs for _ in focal_papers]

    logger.info(
        "sampled %d focal papers for domain=%s year=%d; base=%d distant=%d future=%d; cited left empty",
        len(focal_papers),
        focal_domain,
        focal_year,
        len(base_pool),
        len(distant_pool),
        len(future_docs),
    )
    return focal_papers, future_docs_by_focal


def _validate_inputs(
    *,
    n_focal: int,
    base_pool_size: int,
    distant_pool_size: int,
    future_pool_size: int,
    year_start: int | None,
    focal_year: int,
) -> None:
    for name, value in (
        ("n_focal", n_focal),
        ("base_pool_size", base_pool_size),
        ("distant_pool_size", distant_pool_size),
        ("future_pool_size", future_pool_size),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative; got {value}.")
    if year_start is not None and year_start > focal_year:
        raise ValueError(f"year_start must be <= focal_year; got {year_start} > {focal_year}.")


def _available_work_files(data_dir: Path) -> list[tuple[int, Path]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    year_files = []
    for path in sorted(data_dir.glob("works_*.parquet")):
        match = _WORKS_FILE_RE.fullmatch(path.name)
        if match is not None:
            year_files.append((int(match.group(1)), path))

    if not year_files:
        raise FileNotFoundError(f"no works_*.parquet files found in {data_dir}")
    return sorted(year_files, key=lambda item: (item[0], item[1].name))


def _paths_for_range(
    year_files: Sequence[tuple[int, Path]],
    *,
    min_year: int | None = None,
    max_year_exclusive: int | None = None,
) -> list[Path]:
    return [
        path
        for year, path in year_files
        if (min_year is None or year >= min_year)
        and (max_year_exclusive is None or year < max_year_exclusive)
    ]


def _read_records(
    paths: Sequence[Path],
    *,
    domain: str | None = None,
    excluded_domain: str | None = None,
    min_year: int | None = None,
    max_year_exclusive: int | None = None,
) -> list[_PaperRecord]:
    import pandas as pd

    records: list[_PaperRecord] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=list(_PARQUET_COLUMNS))
        abstracts = frame["abstract"].astype("string").str.strip()
        domains = frame["domain"].astype("string").str.strip()
        years = pd.to_numeric(frame["year"], errors="coerce")

        mask = abstracts.notna() & domains.notna() & years.notna() & (abstracts != "")
        if domain is not None:
            mask &= domains == domain
        if excluded_domain is not None:
            mask &= domains != excluded_domain
        if min_year is not None:
            mask &= years >= min_year
        if max_year_exclusive is not None:
            mask &= years < max_year_exclusive

        selected = pd.DataFrame({"abstract": abstracts[mask], "year": years[mask]})
        records.extend(
            _PaperRecord(abstract=str(row.abstract), year=int(row.year))
            for row in selected.itertuples(index=False)
        )
    return records


def _sample_records(
    records: Sequence[_PaperRecord],
    sample_size: int,
    rng: np.random.Generator,
) -> list[_PaperRecord]:
    if sample_size == 0 or not records:
        return []
    count = min(sample_size, len(records))
    indices = rng.choice(len(records), size=count, replace=False)
    return [records[int(index)] for index in indices]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test OpenAlex focal-paper sampling for TSND axioms.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing works_{year}.parquet files.",
    )
    parser.add_argument("--domain", required=True, help="Focal domain label.")
    parser.add_argument("--year", type=int, required=True, help="Focal publication year.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser


def _main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    focal_papers, future_docs_by_focal = sample_focal_papers(
        args.data_dir,
        focal_domain=args.domain,
        focal_year=args.year,
        seed=args.seed,
    )
    base_size = len(focal_papers[0].base_pool) if focal_papers else 0
    distant_size = len(focal_papers[0].distant_pool) if focal_papers else 0
    future_size = len(future_docs_by_focal[0]) if future_docs_by_focal else 0
    print(
        "built "
        f"focal_papers={len(focal_papers)} "
        f"base_pool={base_size} "
        f"distant_pool={distant_size} "
        f"future_pool={future_size}"
    )


if __name__ == "__main__":
    _main()
