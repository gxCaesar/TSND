"""Embedding models for TSND.

This module ports the validated pilot yearly SGNS training logic and adds
sentence-transformer embedders used by static-density and semantic baselines.
Yearly SGNS artifacts preserve the pilot on-disk schema exactly:
``units`` with dtype object and ``mat`` with dtype float32.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

LOG = logging.getLogger(__name__)

Config = Mapping[str, Any]
FloatMatrix = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]
PILOT_SKIP_GRAM_MODE = 1
PILOT_NEGATIVE_SAMPLES = 5
BGE_M3_MODEL_ID = "BAAI/bge-m3"


class ConfigError(ValueError):
    """Raised when a TSND embedding configuration is missing or invalid."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with ``yaml.safe_load``.

    Args:
        path: YAML config path.

    Returns:
        Parsed config mapping.

    Raises:
        ConfigError: If the config is empty or not a mapping.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"config must be a mapping: {cfg_path}")
    return loaded


def resolve_artifact_dir(cfg: Config, config_path: str | Path | None = None) -> Path:
    """Resolve the artifact directory used by pilot-compatible stages.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path used to infer the project root when
            ``_art_dir`` is absent.

    Returns:
        Artifact directory path.
    """
    raw_art_dir = cfg.get("_art_dir")
    if raw_art_dir is not None:
        return Path(str(raw_art_dir))
    root = _resolve_project_root(cfg, config_path)
    artifacts_cfg = cfg.get("artifacts")
    if isinstance(artifacts_cfg, Mapping) and artifacts_cfg.get("out_dir"):
        return root / str(artifacts_cfg["out_dir"])
    tag = str(cfg.get("run_tag", "main"))
    return root / f"artifacts_{tag}"


def resolve_data_dir(cfg: Config, config_path: str | Path | None = None) -> Path:
    """Resolve the works parquet data directory.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path used to infer the project root when
            ``_data_dir`` is absent.

    Returns:
        Data directory path containing ``works_{year}.parquet`` files.
    """
    raw_data_dir = cfg.get("_data_dir")
    if raw_data_dir is not None:
        return Path(str(raw_data_dir))
    data_cfg = _require_mapping(cfg, "data")
    out_dir = _require_str(data_cfg, "out_dir")
    return _resolve_project_root(cfg, config_path) / out_dir


def chunked_matmul_nn_distance(
    mat_a: npt.ArrayLike,
    mat_b: npt.ArrayLike,
    chunk_size: int,
) -> tuple[IndexArray, FloatMatrix]:
    """Compute nearest-neighbor cosine distances by chunked matrix multiply.

    Args:
        mat_a: Query matrix with shape ``(n_query, dim)``.
        mat_b: Candidate matrix with shape ``(n_candidate, dim)``.
        chunk_size: Number of query rows per matrix-multiply chunk.

    Returns:
        Pair ``(indices, distances)`` where ``indices`` has shape
        ``(n_query,)`` and dtype int64, and ``distances`` has shape
        ``(n_query,)`` and dtype float32. Distance is ``1 - max_cosine``.

    Raises:
        ValueError: If shapes, ranges, or numeric values are invalid.
    """
    a = _as_float_matrix(mat_a, "mat_a")
    b = _as_float_matrix(mat_b, "mat_b")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: mat_a dim={a.shape[1]}, mat_b dim={b.shape[1]}")
    if b.shape[0] == 0:
        raise ValueError("mat_b must contain at least one candidate row")
    if a.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    a_norm = _normalize_rows(a)
    b_norm = _normalize_rows(b)
    indices = np.empty(a_norm.shape[0], dtype=np.int64)
    distances = np.empty(a_norm.shape[0], dtype=np.float32)
    for start in range(0, a_norm.shape[0], chunk_size):
        end = min(start + chunk_size, a_norm.shape[0])
        sims = a_norm[start:end] @ b_norm.T
        best = np.argmax(sims, axis=1)
        best_sims = sims[np.arange(end - start), best]
        indices[start:end] = best.astype(np.int64)
        distances[start:end] = (1.0 - np.clip(best_sims, -1.0, 1.0)).astype(np.float32)
    return indices, distances


class YearlySgnsEmbedder:
    """Train pilot-compatible yearly SGNS embeddings.

    Each paper's unit set is one SGNS sentence. For year ``t``, training uses
    the sliding window ``D_{t-w+1..t}``, then restricts output rows to that
    year's vocabulary ``V_t``. Saved matrices have shape ``(n_units, dim)``.
    """

    def __init__(self, cfg: Config, terms_dir: Path, out_dir: Path, workers: int) -> None:
        """Initialize the SGNS embedder.

        Args:
            cfg: TSND config containing ``embed`` and ``seed_list`` sections.
            terms_dir: Directory containing ``year_{t}.pkl`` graph artifacts.
            out_dir: Directory for ``emb_{t}_s{seed}.npz`` artifacts.
            workers: Word2Vec worker count.

        Raises:
            ConfigError: If required config values are missing or invalid.
        """
        self.cfg = cfg
        self.embed_cfg = _require_mapping(cfg, "embed")
        self.terms_dir = terms_dir
        self.out_dir = out_dir
        self.workers = workers
        _require_positive_int(self.embed_cfg, "sgns_dim")
        _require_positive_int(self.embed_cfg, "window")
        _require_non_negative_int(self.embed_cfg, "min_count")
        _require_positive_int(self.embed_cfg, "epochs")
        _require_positive_int(self.embed_cfg, "train_window_years")
        _sgns_mode(self.embed_cfg)
        _negative_samples(self.embed_cfg)
        _require_positive_int_value(workers, "workers")

    def train_year(self, year: int, seed: int) -> tuple[list[str], FloatMatrix, int]:
        """Train one SGNS model for a year and seed.

        Args:
            year: Target year ``t``.
            seed: Random seed for Word2Vec and NumPy/Python RNGs.

        Returns:
            Tuple ``(units, mat, n_sentences)``. ``units`` is the row vocabulary,
            and ``mat`` has shape ``(n_units, dim)`` with dtype float32.

        Raises:
            RuntimeError: If no target-year vocabulary survives SGNS training.
        """
        from gensim.models import Word2Vec

        _require_positive_int_value(year, "year")
        _require_non_negative_int_value(seed, "seed")
        train_window_years = _require_positive_int(self.embed_cfg, "train_window_years")
        sentences: list[list[str]] = []
        for window_year in range(year - train_window_years + 1, year + 1):
            year_path = self.terms_dir / f"year_{window_year}.pkl"
            if year_path.exists():
                year_graph = _load_year_graph(year_path)
                sentences.extend(_require_paper_units(year_graph, year_path))
        if not sentences:
            raise RuntimeError(f"year {year} seed {seed}: no SGNS sentences found")

        _set_seed(seed)
        model = Word2Vec(
            sentences=sentences,
            vector_size=_require_positive_int(self.embed_cfg, "sgns_dim"),
            window=_require_positive_int(self.embed_cfg, "window"),
            min_count=_require_non_negative_int(self.embed_cfg, "min_count"),
            sg=_sgns_mode(self.embed_cfg),
            negative=_negative_samples(self.embed_cfg),
            epochs=_require_positive_int(self.embed_cfg, "epochs"),
            workers=self.workers,
            seed=seed,
        )
        target_graph = _load_year_graph(self.terms_dir / f"year_{year}.pkl")
        vocab = _require_str_list(target_graph, "vocab")
        units = [unit for unit in vocab if unit in model.wv]
        if not units:
            raise RuntimeError(f"year {year} seed {seed}: no vocab survived min_count in SGNS")
        mat = np.vstack([model.wv[unit] for unit in units]).astype(np.float32)
        _validate_embedding_output(units, mat)
        return units, mat, len(sentences)

    def save_year(self, year: int, seed: int, units: Sequence[str], mat: npt.ArrayLike) -> Path:
        """Save one pilot-compatible yearly embedding artifact.

        Args:
            year: Target year ``t``.
            seed: SGNS random seed.
            units: Row vocabulary with length ``n_units``.
            mat: Embedding matrix with shape ``(n_units, dim)``.

        Returns:
            Path to ``emb_{year}_s{seed}.npz``.
        """
        _require_positive_int_value(year, "year")
        _require_non_negative_int_value(seed, "seed")
        mat32 = _as_float_matrix(mat, "mat")
        _validate_embedding_output(units, mat32)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"emb_{year}_s{seed}.npz"
        np.savez_compressed(out_path, units=np.array(list(units), dtype=object), mat=mat32.astype(np.float32))
        return out_path

    def run(self, year: int | None = None) -> None:
        """Train and save SGNS embeddings for configured years and seeds.

        Args:
            year: Optional single target year. If absent, all configured years
                from ``data.year_start`` through ``data.year_end`` are processed.
        """
        years = _years_from_config(self.cfg, year)
        seeds = _seed_list(self.cfg)
        for target_year in years:
            if not (self.terms_dir / f"year_{target_year}.pkl").exists():
                LOG.warning("missing terms for %d, skip", target_year)
                continue
            for seed in seeds:
                units, mat, n_sentences = self.train_year(target_year, seed)
                out_path = self.save_year(target_year, seed, units, mat)
                LOG.info(
                    "year %d seed %d: %d units, dim %d, %d sentences -> %s",
                    target_year,
                    seed,
                    len(units),
                    mat.shape[1],
                    n_sentences,
                    out_path,
                )


class SpecterEmbedder:
    """SPECTER sentence-transformer embedder.

    The pilot used ``allenai-specter`` via sentence-transformers. The full run
    uses SPECTER2 (``allenai/specter2_base`` + proximity adapter). Output
    matrices are batched, L2-normalized, and have shape ``(n_texts, dim)``.
    """

    def __init__(self, model_id: str, batch_size: int, device: str | None = None) -> None:
        """Initialize a SPECTER embedder.

        Args:
            model_id: Sentence-transformers model id from config.
            batch_size: Encoding batch size.
            device: Optional sentence-transformers device string.
        """
        if not model_id:
            raise ValueError("model_id must be non-empty")
        _require_positive_int_value(batch_size, "batch_size")
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> SpecterEmbedder:
        """Build a SPECTER embedder from ``baseline_specter`` config.

        Args:
            cfg: TSND config mapping.

        Returns:
            Configured ``SpecterEmbedder``.
        """
        specter_cfg = _require_mapping(cfg, "baseline_specter")
        return cls(
            model_id=_require_str(specter_cfg, "model"),
            batch_size=_require_positive_int(specter_cfg, "batch_size"),
            device=_optional_str(specter_cfg, "device"),
        )

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts into normalized SPECTER vectors.

        Args:
            texts: Text inputs with length ``n_texts``.

        Returns:
            Float32 embedding matrix with shape ``(n_texts, dim)``.
        """
        _validate_texts(texts)
        model = self._load_model()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _as_float_matrix(vecs, "specter_embeddings")

    def _load_model(self) -> Any:
        """Load and cache the sentence-transformers model.

        Returns:
            Loaded ``SentenceTransformer`` instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model


class BgeM3Embedder:
    """BGE-M3 sentence-transformer embedder.

    Uses ``BAAI/bge-m3`` by default unless a config supplies another model id.
    Output matrices are batched, L2-normalized, and have shape
    ``(n_texts, dim)``.
    """

    def __init__(self, model_id: str, batch_size: int, device: str | None = None) -> None:
        """Initialize a BGE-M3 embedder.

        Args:
            model_id: Sentence-transformers model id.
            batch_size: Encoding batch size.
            device: Optional sentence-transformers device string.
        """
        if not model_id:
            raise ValueError("model_id must be non-empty")
        _require_positive_int_value(batch_size, "batch_size")
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> BgeM3Embedder:
        """Build a BGE-M3 embedder from ``baseline_bge_m3`` config.

        Args:
            cfg: TSND config mapping.

        Returns:
            Configured ``BgeM3Embedder``.
        """
        bge_cfg = _require_mapping(cfg, "baseline_bge_m3")
        raw_model_id = bge_cfg.get("model", BGE_M3_MODEL_ID)
        if not isinstance(raw_model_id, str) or not raw_model_id:
            raise ConfigError("baseline_bge_m3.model must be a non-empty string")
        return cls(
            model_id=raw_model_id,
            batch_size=_require_positive_int(bge_cfg, "batch_size"),
            device=_optional_str(bge_cfg, "device"),
        )

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts into normalized BGE-M3 vectors.

        Args:
            texts: Text inputs with length ``n_texts``.

        Returns:
            Float32 embedding matrix with shape ``(n_texts, dim)``.
        """
        _validate_texts(texts)
        model = self._load_model()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _as_float_matrix(vecs, "bge_m3_embeddings")

    def _load_model(self) -> Any:
        """Load and cache the sentence-transformers model.

        Returns:
            Loaded ``SentenceTransformer`` instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model


def embed_works_parquet(
    embedder: SpecterEmbedder | BgeM3Embedder,
    parquet_path: Path,
    out_path: Path,
    text_column: str,
) -> FloatMatrix:
    """Embed one yearly works parquet file.

    Args:
        embedder: Static sentence-transformer embedder.
        parquet_path: Input ``works_{year}.parquet`` path.
        out_path: Output ``.npy`` path.
        text_column: Text column to embed.

    Returns:
        Float32 matrix with shape ``(n_papers, dim)``.
    """
    if not text_column:
        raise ValueError("text_column must be non-empty")
    if out_path.exists():
        return _as_float_matrix(np.load(out_path), "cached_embeddings")
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=[text_column])
    texts = [str(text) for text in df[text_column].fillna("").tolist()]
    vecs = embedder.encode(texts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vecs.astype(np.float32))
    return vecs


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for pilot-compatible SGNS embedding training.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_yaml_config(args.config)
    art_dir = resolve_artifact_dir(cfg, args.config)
    terms_dir = art_dir / "terms"
    out_dir = art_dir / "embed"
    embedder = YearlySgnsEmbedder(cfg=cfg, terms_dir=terms_dir, out_dir=out_dir, workers=args.workers)
    embedder.run(args.year)
    return 0


def _resolve_project_root(cfg: Config, config_path: str | Path | None) -> Path:
    """Resolve the project root for path derivation.

    Args:
        cfg: TSND config mapping.
        config_path: Optional config path.

    Returns:
        Project root path.
    """
    raw_root = cfg.get("_root")
    if raw_root is not None:
        return Path(str(raw_root))
    if config_path is not None:
        cfg_path = Path(config_path).resolve()
        if cfg_path.parent.name == "configs":
            return cfg_path.parent.parent
        return cfg_path.parent
    return Path.cwd()


def _load_year_graph(path: Path) -> dict[str, Any]:
    """Load a trusted yearly term graph pickle.

    Args:
        path: Path to a local TSND ``year_{t}.pkl`` artifact.

    Returns:
        Graph dictionary.
    """
    _require_trusted_artifact_path(path, ".pkl")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, dict):
        raise ValueError(f"year graph must be a dict: {path}")
    return graph


def _require_trusted_artifact_path(path: Path, suffix: str) -> None:
    """Validate a local TSND artifact path before pickle-capable loading.

    Args:
        path: Artifact path.
        suffix: Required file suffix.

    Raises:
        ValueError: If the path suffix is not the expected artifact type.
    """
    if path.suffix != suffix:
        raise ValueError(f"expected a {suffix} TSND artifact, got {path}")


def _require_paper_units(graph: Mapping[str, Any], path: Path) -> list[list[str]]:
    """Validate and return paper-unit sentences.

    Args:
        graph: Year graph mapping.
        path: Source path for error messages.

    Returns:
        List of SGNS sentences.
    """
    raw = graph.get("paper_units")
    if not isinstance(raw, list):
        raise ValueError(f"paper_units must be a list: {path}")
    sentences: list[list[str]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"paper_units rows must be sequences: {path}")
        sentences.append([str(unit) for unit in row])
    return sentences


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash seeding for deterministic SGNS runs.

    Args:
        seed: Non-negative random seed.
    """
    _require_non_negative_int_value(seed, "seed")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _normalize_rows(mat: FloatMatrix) -> FloatMatrix:
    """L2-normalize rows of a matrix.

    Args:
        mat: Matrix with shape ``(n_rows, dim)``.

    Returns:
        Row-normalized float32 matrix with shape ``(n_rows, dim)``.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(mat / norms, dtype=np.float32)


def _as_float_matrix(value: npt.ArrayLike, name: str) -> FloatMatrix:
    """Validate and coerce an array-like object to a finite float32 matrix.

    Args:
        value: Array-like value.
        name: Name used in error messages.

    Returns:
        Float32 matrix with shape ``(n_rows, dim)``.
    """
    mat = np.asarray(value, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {mat.shape}")
    if mat.shape[1] <= 0:
        raise ValueError(f"{name} must have positive embedding dimension")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} contains non-finite values")
    return mat


def _validate_embedding_output(units: Sequence[str], mat: FloatMatrix) -> None:
    """Validate row vocabulary and embedding matrix compatibility.

    Args:
        units: Row vocabulary.
        mat: Matrix with shape ``(len(units), dim)``.
    """
    if len(units) == 0:
        raise ValueError("units must be non-empty")
    if len(units) != mat.shape[0]:
        raise ValueError(f"units/mat row mismatch: {len(units)} vs {mat.shape[0]}")
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise ValueError("all units must be non-empty strings")


def _validate_texts(texts: Sequence[str]) -> None:
    """Validate text inputs for sentence-transformer encoding.

    Args:
        texts: Text sequence.
    """
    if len(texts) == 0:
        raise ValueError("texts must be non-empty")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("all texts must be strings")


def _years_from_config(cfg: Config, year: int | None) -> list[int]:
    """Resolve target years from config and optional CLI year.

    Args:
        cfg: TSND config mapping.
        year: Optional single year.

    Returns:
        Ordered target years.
    """
    if year is not None:
        _require_positive_int_value(year, "year")
        return [year]
    data_cfg = _require_mapping(cfg, "data")
    year_start = _require_positive_int(data_cfg, "year_start")
    year_end = _require_positive_int(data_cfg, "year_end")
    if year_end < year_start:
        raise ConfigError("data.year_end must be >= data.year_start")
    return list(range(year_start, year_end + 1))


def _seed_list(cfg: Config) -> list[int]:
    """Read and validate ``seed_list`` from config.

    Args:
        cfg: TSND config mapping.

    Returns:
        Non-empty seed list.
    """
    raw = cfg.get("seed_list")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("seed_list must be a non-empty list")
    seeds = [_coerce_int(seed, "seed_list item") for seed in raw]
    for seed in seeds:
        _require_non_negative_int_value(seed, "seed_list item")
    return seeds


def _require_mapping(cfg: Config | None, key: str) -> Mapping[str, Any]:
    """Read a required mapping config section.

    Args:
        cfg: Parent mapping.
        key: Section key.

    Returns:
        Required mapping section.
    """
    if cfg is None:
        raise ConfigError(f"{key} config section is required")
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _optional_mapping(cfg: Config, key: str) -> Mapping[str, Any] | None:
    """Read an optional mapping config section.

    Args:
        cfg: Parent mapping.
        key: Section key.

    Returns:
        Optional mapping section.
    """
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_str(cfg: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string config value.

    Args:
        cfg: Config mapping.
        key: Config key.

    Returns:
        String value.
    """
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_str(cfg: Mapping[str, Any] | None, key: str) -> str | None:
    """Read an optional string config value.

    Args:
        cfg: Config mapping.
        key: Config key.

    Returns:
        Optional string value.
    """
    if cfg is None:
        return None
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _require_str_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    """Read a required string-list config or artifact value.

    Args:
        cfg: Mapping.
        key: Key to read.

    Returns:
        List of strings.
    """
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    out = [str(item) for item in value]
    if any(not item for item in out):
        raise ValueError(f"{key} entries must be non-empty")
    return out


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> int:
    """Read a required positive integer config value.

    Args:
        cfg: Config mapping.
        key: Config key.

    Returns:
        Positive integer.
    """
    return _require_positive_int_value(_coerce_int(cfg.get(key), key), key)


def _require_non_negative_int(cfg: Mapping[str, Any], key: str) -> int:
    """Read a required non-negative integer config value.

    Args:
        cfg: Config mapping.
        key: Config key.

    Returns:
        Non-negative integer.
    """
    return _require_non_negative_int_value(_coerce_int(cfg.get(key), key), key)


def _sgns_mode(cfg: Mapping[str, Any]) -> int:
    """Read the SGNS mode while preserving the pilot skip-gram setting.

    Args:
        cfg: Config mapping.

    Returns:
        Gensim ``sg`` mode. The pilot used skip-gram.
    """
    value = _coerce_int(cfg.get("sg", PILOT_SKIP_GRAM_MODE), "embed.sg")
    if value != PILOT_SKIP_GRAM_MODE:
        raise ConfigError("embed.sg must be 1 for skip-gram SGNS")
    return value


def _negative_samples(cfg: Mapping[str, Any]) -> int:
    """Read the negative-sampling count with pilot compatibility.

    Args:
        cfg: Config mapping.

    Returns:
        Positive negative-sampling count.
    """
    value = _coerce_int(cfg.get("negative", PILOT_NEGATIVE_SAMPLES), "embed.negative")
    if value <= 0:
        raise ConfigError("embed.negative must be positive")
    return value


def _require_positive_int_value(value: int, name: str) -> int:
    """Validate a positive integer value.

    Args:
        value: Integer value.
        name: Name used in error messages.

    Returns:
        Positive integer.
    """
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def _require_non_negative_int_value(value: int, name: str) -> int:
    """Validate a non-negative integer value.

    Args:
        value: Integer value.
        name: Name used in error messages.

    Returns:
        Non-negative integer.
    """
    if value < 0:
        raise ConfigError(f"{name} must be non-negative")
    return value


def _coerce_int(value: Any, name: str) -> int:
    """Coerce a config value to int without accepting booleans.

    Args:
        value: Config value.
        name: Name used in error messages.

    Returns:
        Integer value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
