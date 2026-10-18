"""B3 TF-IDF novelty distance to the prior-three-year corpus.

Source paper: none; this is a trivial-strong lexical control baseline.
This control is implemented directly for the TSND front-uptake task.

The vectorizer IDF is fit only on configured training cohorts, then each
year-t front key is scored as ``1 - cosine(front_text, prior_centroid)`` where
the prior centroid is built from documents in ``[t - window, t)``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tsnd.models.baselines.citation_history import load_labels

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINE_CONFIG_KEY = "tfidf_novelty"
DEFAULT_PRIOR_WINDOW_YEARS = 3
DEFAULT_MIN_DF = 2
DEFAULT_MAX_FEATURES = None
DEFAULT_WORKS_PATTERN = "works_{year}.parquet"
DEFAULT_TEXT_COLUMNS = ("title", "abstract")

FEATURE_COLUMNS = ["kind_pair", "tfidf_novelty"]


def build_features(
    cohort_year: int,
    config: Mapping[str, Any],
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    train_years: Sequence[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build B3 TF-IDF prior-corpus novelty features.

    Args:
        cohort_year: Birth year ``t`` for front labels.
        config: Runtime configuration. Optional ``tfidf_novelty`` keys include
            ``prior_window_years``, ``min_df``, ``max_features``,
            ``works_pattern``, and ``text_columns``.
        artifact_dir: Directory containing ``labels/labels_{t}.parquet``.
        data_dir: Directory containing yearly works parquet files.
        train_years: Training cohort years used to fit IDF. Required to avoid
            test-cohort leakage.
        seed: Unused by deterministic B3 features.

    Returns:
        Per-front feature DataFrame with ``key``, ``kind_pair``, and
        ``tfidf_novelty``.
    """
    del seed
    _validate_year(cohort_year)
    art_path = Path(artifact_dir)
    data_path = Path(data_dir)
    baseline_cfg = _nested_config(config, DEFAULT_BASELINE_CONFIG_KEY)
    idf_years = _resolve_train_years(config, train_years)

    labels = load_labels(art_path, cohort_year)
    vectorizer = fit_train_vectorizer(data_path, idf_years, baseline_cfg)
    prior_texts = load_prior_corpus_texts(data_path, cohort_year, baseline_cfg)
    novelty = score_front_texts(labels["key"].astype(str).tolist(), vectorizer, prior_texts)

    out = pd.DataFrame(
        {
            "key": labels["key"].astype(str),
            "kind_pair": (labels["kind"].astype(str) == "pair").astype(int),
            "tfidf_novelty": novelty,
        }
    )
    LOGGER.info("built B3 TF-IDF novelty features for %d fronts at t=%d", len(out), cohort_year)
    return out


def fit_train_vectorizer(
    data_dir: Path,
    train_years: Sequence[int],
    config: Mapping[str, Any],
) -> Any:
    """Fit a TF-IDF vectorizer on training cohorts only.

    Args:
        data_dir: Directory containing yearly works parquet files.
        train_years: Years whose works text forms the IDF corpus.
        config: TF-IDF config.

    Returns:
        Fitted ``sklearn.feature_extraction.text.TfidfVectorizer``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = load_corpus_texts(data_dir, train_years, config)
    if not corpus:
        raise ValueError("cannot fit TF-IDF vectorizer: empty train corpus")

    min_df = int(config.get("min_df", DEFAULT_MIN_DF))
    max_features = config.get("max_features", DEFAULT_MAX_FEATURES)
    if min_df <= 0:
        raise ValueError(f"min_df must be positive, got {min_df}")
    if max_features is not None and int(max_features) <= 0:
        raise ValueError(f"max_features must be positive when provided, got {max_features}")

    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features)
    try:
        vectorizer.fit(corpus)
    except ValueError as exc:
        raise ValueError("failed to fit TF-IDF vectorizer on train corpus") from exc
    return vectorizer


def load_prior_corpus_texts(
    data_dir: Path,
    cohort_year: int,
    config: Mapping[str, Any],
) -> list[str]:
    """Load texts from the prior corpus window ``[t-window, t)``.

    Args:
        data_dir: Directory containing yearly works parquet files.
        cohort_year: Cohort year ``t``.
        config: TF-IDF config containing ``prior_window_years``.

    Returns:
        List of document texts in the prior window.
    """
    _validate_year(cohort_year)
    prior_window = int(config.get("prior_window_years", DEFAULT_PRIOR_WINDOW_YEARS))
    if prior_window <= 0:
        raise ValueError(f"prior_window_years must be positive, got {prior_window}")
    years = list(range(cohort_year - prior_window, cohort_year))
    return load_corpus_texts(data_dir, years, config)


def load_corpus_texts(
    data_dir: Path,
    years: Sequence[int],
    config: Mapping[str, Any],
) -> list[str]:
    """Load yearly works text for TF-IDF.

    Args:
        data_dir: Directory containing yearly works parquet files.
        years: Years to load.
        config: TF-IDF config with ``works_pattern`` and ``text_columns``.

    Returns:
        Nonempty document strings from available yearly parquet files.
    """
    works_pattern = str(config.get("works_pattern", DEFAULT_WORKS_PATTERN))
    text_columns = tuple(config.get("text_columns", DEFAULT_TEXT_COLUMNS))
    if not text_columns:
        raise ValueError("text_columns must contain at least one column name")

    texts: list[str] = []
    for year in years:
        _validate_year(int(year))
        works_path = data_dir / works_pattern.format(year=int(year))
        if not works_path.exists():
            LOGGER.warning("works parquet missing for TF-IDF corpus: %s", works_path)
            continue
        available_columns = pd.read_parquet(works_path).columns
        columns = [column for column in text_columns if column in available_columns]
        if not columns:
            LOGGER.warning("works parquet has none of the requested text columns: %s", works_path)
            continue
        frame = pd.read_parquet(works_path, columns=columns)
        texts.extend(_combine_text_columns(frame, columns))
    return texts


def score_front_texts(
    keys: Sequence[str],
    vectorizer: Any,
    prior_texts: Sequence[str],
) -> npt.NDArray[np.float64]:
    """Score front-key novelty against a prior-corpus centroid.

    Args:
        keys: Front keys to score.
        vectorizer: Fitted TF-IDF vectorizer.
        prior_texts: Prior document texts used to form the centroid.

    Returns:
        Float array with shape ``(n_fronts,)`` containing novelty scores.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not prior_texts:
        return np.full(len(keys), np.nan, dtype=np.float64)

    prior_matrix = vectorizer.transform(list(prior_texts))
    if prior_matrix.shape[0] == 0:
        return np.full(len(keys), np.nan, dtype=np.float64)
    centroid = np.asarray(prior_matrix.mean(axis=0))
    front_matrix = vectorizer.transform([front_key_to_text(key) for key in keys])
    similarity = cosine_similarity(front_matrix, centroid).ravel()
    return np.asarray(1.0 - similarity, dtype=np.float64)


def front_key_to_text(key: str) -> str:
    """Convert a front key into lexical text, matching the pilot transform."""
    return key.replace("|", " ").replace("_", " ")


def _combine_text_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Combine configured text columns into one document string per row."""
    texts: list[str] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        parts = [str(value) for value in row if isinstance(value, str) and value.strip()]
        if parts:
            texts.append(" ".join(parts))
    return texts


def _resolve_train_years(
    config: Mapping[str, Any],
    train_years: Sequence[int] | None,
) -> list[int]:
    """Resolve and validate train years for IDF fitting."""
    raw_years: Sequence[int] | None
    if train_years is not None:
        raw_years = train_years
    else:
        raw_value = config.get("train_years")
        if raw_value is None:
            raise ValueError("B3 requires train_years or config['train_years'] for IDF fitting")
        if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
            raise TypeError("config['train_years'] must be a sequence of years")
        raw_years = raw_value

    years = sorted({int(year) for year in raw_years})
    if not years:
        raise ValueError("train_years must not be empty")
    for year in years:
        _validate_year(year)
    return years


def _nested_config(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a nested config mapping or an empty mapping."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config[{key!r}] must be a mapping when provided")
    return value


def _validate_year(year: int) -> None:
    """Validate a positive integer year."""
    if not isinstance(year, int):
        raise TypeError(f"year must be int, got {type(year).__name__}")
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")
