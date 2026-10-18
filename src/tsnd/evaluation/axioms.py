"""Axiomatic evaluation of scientific-novelty metrics (the theory contribution).

This module implements the eight axioms of the Axiomatic Benchmark (Liu and Zhai,
arXiv:2604.15145) plus two temporal axioms introduced by this work:

  A1 Self-recognition       adding an exact copy of P to the pool lowers Score(P)
  A2 Paraphrase invariance  adding a meaning-preserving rephrase of P lowers Score(P)
  A3 Distributed coverage   spreading fragments of P across neighbours lowers Score(P),
                            monotonically in fragment size (the Ax3grad gradient check)
  A4 Unrelatedness          a distant-field pool raises Score(P)
  A5 Citation relevance     removing P's cited papers from the pool raises Score(P)
  A6 Citation primacy       a references-only pool lowers Score(P) vs pool minus references
  A7 Temporal accumulation  an older reference slice raises Score(P)
  A8 Temporal accumulation  a newer reference slice lowers Score(P)
  A9 No-future-leakage      Score_t(P) is invariant to any change in documents dated > t
  A10 Decomposition         corpus interventions move each novelty component independently
      identifiability        and the estimator attributes each to the right component

Each axiom is a binary pass/fail per focal paper. A metric is summarised by its pass
rate over a sample of focal papers, following the benchmark protocol. The suite takes
any novelty metric expressed through the ``NoveltyMetric`` protocol, so it audits our
own decomposition components, the density-based baselines (RND, SemNovel, Yin), and
LLM novelty judges on the same footing.

Reference protocol numbers (Liu and Zhai, arXiv:2604.15145): the best single metric
(RND) averages 71.5 percent pass; no metric passes A7 and A8 together (Yin 94/11,
SemNovel 0/100); the Ax3 gradient is hardest for all (6 to 31 percent). A9 and A10
target exactly the temporal dimension on which every prior metric fails.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

Pool = Sequence[str]


class NoveltyMetric(Protocol):
    """A novelty metric scores a focal text against a reference pool.

    Higher score means more novel. Implementations include the density baselines
    (RND, SemNovel, Yin), our pool-relative novelty components, and prompted LLM
    judges. ``score`` must be deterministic given its inputs.
    """

    def score(self, focal: str, pool: Pool) -> float:
        """Return the novelty of ``focal`` relative to ``pool`` (higher = more novel)."""
        ...


@dataclass(frozen=True)
class FocalPaper:
    """One focal paper and the metadata the axioms need.

    Attributes:
        text: the focal abstract or idea text.
        base_pool: same-task papers published strictly before the focal year.
        cited: the subset of ``base_pool`` that the focal paper cites.
        distant_pool: papers from an unrelated field (for A4).
        year: focal publication year (for the temporal axioms).
        pool_years: publication year of each paper in ``base_pool`` (same order).
    """

    text: str
    base_pool: tuple[str, ...]
    cited: tuple[str, ...] = ()
    distant_pool: tuple[str, ...] = ()
    year: int | None = None
    pool_years: tuple[int, ...] = ()


@dataclass
class AxiomResult:
    """Pass/fail outcome of one axiom on one focal paper."""

    axiom: str
    passed: bool
    detail: str = ""


@dataclass
class SuiteReport:
    """Per-axiom pass rates for one metric over a sample of focal papers."""

    metric_name: str
    n_focal: int
    pass_rate: dict[str, float] = field(default_factory=dict)
    n_eligible: dict[str, int] = field(default_factory=dict)

    @property
    def average_pass_rate(self) -> float:
        """Return the mean pass rate over eligible axioms."""
        rates = [r for r in self.pass_rate.values() if not np.isnan(r)]
        return float(np.mean(rates)) if rates else float("nan")


# Reference paraphraser / fragmenter hooks. Defaults are deterministic and offline so
# the suite runs without an LLM; the full run can pass an LLM-backed paraphraser.
def _identity_paraphrase(text: str) -> str:
    """Trivial meaning-preserving rewrite: reorder sentences. Deterministic."""
    parts = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(reversed(parts)) + "." if parts else text


def _sentence_fragments(text: str, k: int) -> list[str]:
    """Split text into k roughly equal fragments (for the distributed-coverage axiom)."""
    sents = [s.strip() for s in text.split(".") if s.strip()]
    if not sents:
        return [text]
    k = max(1, min(k, len(sents)))
    sizes = np.array_split(np.arange(len(sents)), k)
    return [". ".join(sents[idx[0] : idx[-1] + 1]) + "." for idx in sizes if len(idx)]


# ---------------------------------------------------------------------------
# The eight Liu-Zhai axioms. Each returns AxiomResult or None when the focal
# paper is ineligible (for example A5/A6 need cited papers).
# ---------------------------------------------------------------------------
def axiom1_self_recognition(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult:
    """Test whether adding the focal paper to the pool reduces novelty."""
    base = metric.score(fp.text, fp.base_pool)
    with_copy = metric.score(fp.text, (*fp.base_pool, fp.text))
    return AxiomResult("A1_self_recognition", with_copy < base,
                       f"base={base:.4f} with_copy={with_copy:.4f}")


def axiom2_paraphrase_invariance(
    metric: NoveltyMetric, fp: FocalPaper,
    paraphrase: Callable[[str], str] = _identity_paraphrase,
) -> AxiomResult:
    """Test whether adding a focal-paper paraphrase reduces novelty."""
    base = metric.score(fp.text, fp.base_pool)
    with_para = metric.score(fp.text, (*fp.base_pool, paraphrase(fp.text)))
    return AxiomResult("A2_paraphrase_invariance", with_para < base,
                       f"base={base:.4f} with_para={with_para:.4f}")


def axiom3_distributed_coverage(
    metric: NoveltyMetric, fp: FocalPaper,
    fragmenter: Callable[[str, int], list[str]] = _sentence_fragments,
) -> AxiomResult:
    """Test distributed coverage and its fragment-count gradient."""
    base = metric.score(fp.text, fp.base_pool)
    one = metric.score(fp.text, (*fp.base_pool, *fragmenter(fp.text, 1)))
    base_ok = one < base
    # Ax3grad: 4-fragment pool should look less novel than 2-fragment than 1-fragment.
    s4 = metric.score(fp.text, (*fp.base_pool, *fragmenter(fp.text, 4)))
    s2 = metric.score(fp.text, (*fp.base_pool, *fragmenter(fp.text, 2)))
    s1 = metric.score(fp.text, (*fp.base_pool, *fragmenter(fp.text, 1)))
    grad_ok = s4 < s2 < s1
    return AxiomResult("A3_distributed_coverage", bool(base_ok and grad_ok),
                       f"base_ok={base_ok} grad(s4<s2<s1)=({s4:.4f}<{s2:.4f}<{s1:.4f})")


def axiom4_unrelatedness(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult | None:
    """Test whether a distant pool yields greater novelty than the base pool."""
    if not fp.distant_pool:
        return None
    base = metric.score(fp.text, fp.base_pool)
    distant = metric.score(fp.text, fp.distant_pool)
    return AxiomResult("A4_unrelatedness", distant > base,
                       f"base={base:.4f} distant={distant:.4f}")


def axiom5_citation_relevance(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult | None:
    """Test whether removing cited work increases measured novelty."""
    if len(fp.cited) < 1 or len(fp.base_pool) < 20:
        return None
    cited = set(fp.cited)
    minus = tuple(p for p in fp.base_pool if p not in cited)
    base = metric.score(fp.text, fp.base_pool)
    without_cited = metric.score(fp.text, minus)
    return AxiomResult("A5_citation_relevance", without_cited > base,
                       f"base={base:.4f} minus_cited={without_cited:.4f}")


def axiom6_citation_primacy(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult | None:
    """Test whether cited work covers the focal paper better than uncited work."""
    if len(fp.cited) < 1 or len(fp.base_pool) < 20:
        return None
    cited = set(fp.cited)
    minus = tuple(p for p in fp.base_pool if p not in cited)
    cited_only = metric.score(fp.text, tuple(fp.cited))
    minus_cited = metric.score(fp.text, minus)
    return AxiomResult("A6_citation_primacy", cited_only < minus_cited,
                       f"cited_only={cited_only:.4f} minus_cited={minus_cited:.4f}")


def _temporal_slices(fp: FocalPaper) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if len(fp.pool_years) != len(fp.base_pool) or len(fp.base_pool) < 600:
        return None
    order = np.argsort(fp.pool_years)
    oldest = tuple(fp.base_pool[i] for i in order[:300])
    newest = tuple(fp.base_pool[i] for i in order[-300:])
    return oldest, newest


def axiom7_temporal_older(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult | None:
    """Test whether the oldest pool slice yields greater novelty."""
    sl = _temporal_slices(fp)
    if sl is None:
        return None
    oldest, _ = sl
    base = metric.score(fp.text, fp.base_pool)
    old = metric.score(fp.text, oldest)
    return AxiomResult("A7_temporal_older", old > base,
                       f"base={base:.4f} oldest={old:.4f}")


def axiom8_temporal_newer(metric: NoveltyMetric, fp: FocalPaper) -> AxiomResult | None:
    """Test whether the newest pool slice yields lower novelty."""
    sl = _temporal_slices(fp)
    if sl is None:
        return None
    _, newest = sl
    base = metric.score(fp.text, fp.base_pool)
    new = metric.score(fp.text, newest)
    return AxiomResult("A8_temporal_newer", new < base,
                       f"base={base:.4f} newest={new:.4f}")


# ---------------------------------------------------------------------------
# A9 No-future-leakage. A causal temporal metric must not move when documents
# dated after the focal year are added to or removed from the pool. This formally
# disqualifies globally-trained static embeddings as retrospective novelty metrics.
# ---------------------------------------------------------------------------
def axiom9_no_future_leakage(
    metric: NoveltyMetric, fp: FocalPaper, future_docs: Pool, rng: np.random.Generator,
    tol: float = 1e-9,
) -> AxiomResult | None:
    """Test invariance to documents dated after the focal year."""
    if fp.year is None or not future_docs:
        return None
    base = metric.score(fp.text, fp.base_pool)
    shuffled = list(future_docs)
    rng.shuffle(shuffled)
    with_future = metric.score(fp.text, (*fp.base_pool, *shuffled))
    passed = abs(with_future - base) <= tol
    return AxiomResult("A9_no_future_leakage", passed,
                       f"base={base:.6f} with_future={with_future:.6f} delta={with_future - base:.2e}")


# ---------------------------------------------------------------------------
# A10 Decomposition identifiability. Synthetic corpus interventions move one
# component; the estimator must attribute the response to that component and leave
# the others within a placebo noise band. ``ComponentScores`` is the estimator's
# output on a corpus; ``interventions`` produce (intervention_type, corpus) pairs.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentScores:
    """Estimator output for one corpus: each decomposition component's response."""

    lexical_emergence: float
    neighborhood_shift: float
    recombination: float

    def get(self, component: str) -> float:
        """Return one named component score."""
        return {"lexical_emergence": self.lexical_emergence,
                "neighborhood_shift": self.neighborhood_shift,
                "recombination": self.recombination}[component]


def axiom10_decomposition_identifiability(
    estimator: Callable[[object], ComponentScores],
    baseline_corpus: object,
    interventions: Sequence[tuple[str, object]],
    placebos: Sequence[object],
    component_of: dict[str, str],
) -> dict[str, float]:
    """Score decomposition identifiability via injection.

    Args:
        estimator: maps a corpus to component scores.
        baseline_corpus: the unperturbed corpus.
        interventions: (intervention_type, perturbed_corpus) pairs. Each type targets
            one component named by ``component_of`` (e.g. "relabel" -> lexical_emergence).
        placebos: perturbed corpora with no real effect, used to size the noise band.
        component_of: map from intervention type to the component it should move.

    Returns:
        A dict with the per-type accuracy and the overall identifiability score: the
        fraction of injections where the intended component responds beyond the 95th
        percentile placebo band while the others stay within it.
    """
    base = estimator(baseline_corpus)
    components = ("lexical_emergence", "neighborhood_shift", "recombination")
    # placebo noise band per component: 95th percentile of absolute deltas under placebos
    placebo_deltas: dict[str, list[float]] = {c: [] for c in components}
    for pc in placebos:
        sc = estimator(pc)
        for c in components:
            placebo_deltas[c].append(abs(sc.get(c) - base.get(c)))
    band = {c: float(np.percentile(placebo_deltas[c], 95)) if placebo_deltas[c] else 0.0
            for c in components}

    by_type: dict[str, list[bool]] = {}
    for itype, corpus in interventions:
        target = component_of[itype]
        sc = estimator(corpus)
        intended_moves = abs(sc.get(target) - base.get(target)) > band[target]
        others_quiet = all(abs(sc.get(c) - base.get(c)) <= band[c]
                           for c in components if c != target)
        by_type.setdefault(itype, []).append(bool(intended_moves and others_quiet))

    out = {f"{itype}_acc": float(np.mean(v)) for itype, v in by_type.items()}
    per_type = [np.mean(v) for v in by_type.values()]
    out["identifiability"] = float(np.mean(per_type)) if per_type else float("nan")
    return out


# ---------------------------------------------------------------------------
# Suite runner for the eight pool-relative axioms (A1-A8) plus A9.
# A10 is run separately via ``axiom10_decomposition_identifiability`` because it
# operates on the estimator, not on a pool-relative metric.
# ---------------------------------------------------------------------------
_POOL_AXIOMS: tuple[Callable[..., AxiomResult | None], ...] = (
    axiom1_self_recognition,
    axiom2_paraphrase_invariance,
    axiom3_distributed_coverage,
    axiom4_unrelatedness,
    axiom5_citation_relevance,
    axiom6_citation_primacy,
    axiom7_temporal_older,
    axiom8_temporal_newer,
)


def run_axiom_suite(
    metric: NoveltyMetric,
    focal_papers: Sequence[FocalPaper],
    metric_name: str,
    future_docs_by_focal: Sequence[Pool] | None = None,
    seed: int = 0,
) -> SuiteReport:
    """Run A1-A9 over a sample of focal papers and report per-axiom pass rates.

    Args:
        metric: the novelty metric under test.
        focal_papers: sampled focal papers with their pools and metadata.
        metric_name: label for the report.
        future_docs_by_focal: post-focal-year documents per focal paper, for A9.
        seed: RNG seed for the A9 shuffle.

    Returns:
        A ``SuiteReport`` with pass rate and eligible-count per axiom.
    """
    rng = np.random.default_rng(seed)
    passes: dict[str, list[bool]] = {}
    for i, fp in enumerate(focal_papers):
        for axiom_fn in _POOL_AXIOMS:
            res = axiom_fn(metric, fp)
            if res is not None:
                passes.setdefault(res.axiom, []).append(res.passed)
        if future_docs_by_focal is not None and i < len(future_docs_by_focal):
            res9 = axiom9_no_future_leakage(metric, fp, future_docs_by_focal[i], rng)
            if res9 is not None:
                passes.setdefault(res9.axiom, []).append(res9.passed)
    report = SuiteReport(metric_name=metric_name, n_focal=len(focal_papers))
    for axiom, vals in passes.items():
        report.pass_rate[axiom] = float(np.mean(vals)) if vals else float("nan")
        report.n_eligible[axiom] = len(vals)
    logger.info("axiom suite %s: avg pass rate %.3f over %d focal papers",
                metric_name, report.average_pass_rate, len(focal_papers))
    return report
