"""Audit LLM novelty judges with the axiom suite.

This runs the eight Liu-Zhai axioms plus A9 (no-future-leakage) on one or more LLM
novelty judges, producing the first axiom-coverage table that places LLM judges next to
the classical density metrics and our own decomposition components. The expectation,
from the motivation of this work, is that LLM judges (like every prior metric) fail the
temporal axioms, which is the gap the temporal axioms A9 and A10 formalize.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from tsnd.evaluation.axioms import (
    FocalPaper,
    NoveltyMetric,
    Pool,
    SuiteReport,
    run_axiom_suite,
)

logger = logging.getLogger(__name__)


def audit_metrics(
    metrics: dict[str, NoveltyMetric],
    focal_papers: Sequence[FocalPaper],
    future_docs_by_focal: Sequence[Pool] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the axiom suite on each named metric and tabulate per-axiom pass rates.

    Args:
        metrics: mapping from display name to a NoveltyMetric (density metrics, LLM
            judges, or our pool-relative components).
        focal_papers: the sampled focal papers with pools and metadata.
        future_docs_by_focal: post-focal-year documents per focal paper, for A9.
        seed: RNG seed passed to the suite.

    Returns:
        A DataFrame with one row per metric and one column per axiom (pass rate), plus an
        ``average`` column. Rows are sorted by average pass rate descending.
    """
    reports: list[SuiteReport] = []
    for name, metric in metrics.items():
        report = run_axiom_suite(
            metric, focal_papers, name,
            future_docs_by_focal=future_docs_by_focal, seed=seed,
        )
        reports.append(report)
    axioms = sorted({a for r in reports for a in r.pass_rate})
    table = []
    for report in reports:
        row: dict[str, object] = {"metric": report.metric_name}
        for axiom in axioms:
            row[axiom] = report.pass_rate.get(axiom, float("nan"))
        row["average"] = report.average_pass_rate
        table.append(row)
    df = pd.DataFrame(table).sort_values("average", ascending=False).reset_index(drop=True)
    logger.info("axiom-coverage table built for %d metrics over %d axioms",
                len(reports), len(axioms))
    return df
