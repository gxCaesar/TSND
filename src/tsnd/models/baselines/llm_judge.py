"""LLM novelty-judge adapter implementing the NoveltyMetric protocol.

The adapter follows the focal-text and sampled-prior-pool elicitation pattern
of NovBench (Wu et al., arXiv:2604.11543, njust-winchy/llm4novelty).
The completion function is supplied by the caller. This optional adapter is
not connected to an inference backend or exercised in the journal analyses.
The prompt below is part of the scientific scoring implementation.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Completion = Callable[[str], str]

_PROMPT = """You are assessing the novelty of a research idea against prior work.
Rate how novel the idea is relative to the prior abstracts on a scale from 0 (not novel,
fully anticipated by prior work) to 10 (highly novel, unlike anything in prior work).
Answer with a single number.

IDEA:
{focal}

PRIOR WORK (sample):
{pool}

Novelty score (0-10):"""

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class LlmJudgeMetric:
    """Prompt-based LLM novelty judge."""

    def __init__(
        self, complete: Completion, max_pool_shown: int = 20, seed: int = 0,
    ) -> None:
        """Initialise the judge.

        Args:
            complete: maps a prompt string to the model's text response.
            max_pool_shown: how many prior abstracts to include in the prompt.
            seed: RNG seed for sampling which prior abstracts are shown.
        """
        self.complete = complete
        self.max_pool_shown = max_pool_shown
        self.rng = np.random.default_rng(seed)

    def _build_prompt(self, focal: str, pool: Sequence[str]) -> str:
        if len(pool) > self.max_pool_shown:
            idx = self.rng.choice(len(pool), self.max_pool_shown, replace=False)
            shown = [pool[i] for i in sorted(idx)]
        else:
            shown = list(pool)
        pool_text = "\n".join(f"- {p}" for p in shown) if shown else "(none)"
        return _PROMPT.format(focal=focal, pool=pool_text)

    def score(self, focal: str, pool: Sequence[str]) -> float:
        """Return the parsed novelty score in [0, 10]; 5.0 if unparseable."""
        response = self.complete(self._build_prompt(focal, pool))
        match = _NUM.search(response or "")
        if match is None:
            logger.warning("LLM judge: unparseable response %r, defaulting to 5.0", response)
            return 5.0
        return float(np.clip(float(match.group()), 0.0, 10.0))
