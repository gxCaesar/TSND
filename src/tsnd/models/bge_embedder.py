"""Standalone embedders for density-baseline text inputs."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)

DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_DIM = 1024
DEFAULT_BATCH_SIZE = 128
DEFAULT_DEVICE = "cuda"
MAX_CACHE_TEXTS = 200_000
_MISSING = object()
FloatMatrix = npt.NDArray[np.float32]


class BgeEmbedder:
    """BGE sentence-transformer callable matching the density metric embedder contract.

    Output matrices are float32 and have shape ``(n_texts, dim)``. The default model is
    ``BAAI/bge-m3`` with dense embedding dimension 1024.
    """

    def __init__(
        self,
        model: str = DEFAULT_BGE_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
        cache: bool = True,
    ) -> None:
        """Initialize the lazy BGE embedder.

        Args:
            model: Sentence-transformers model id.
            device: Device string passed to ``SentenceTransformer``.
            batch_size: Batch size passed to ``SentenceTransformer.encode``.
            normalize: Whether to L2-normalize embeddings in sentence-transformers.
            cache: Whether to cache embeddings by text hash across calls.

        Raises:
            ValueError: If ``model``, ``device``, or ``batch_size`` is invalid.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if not device:
            raise ValueError("device must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_enabled = cache
        self._model: Any | None = None
        self._dim: int | None = DEFAULT_BGE_DIM if model == DEFAULT_BGE_MODEL else None
        self._cache: OrderedDict[str, FloatMatrix] = OrderedDict()

    @classmethod
    def from_config(cls, config: object) -> BgeEmbedder:
        """Build an embedder from ``config.baseline_bge``.

        Args:
            config: Mapping or object with ``baseline_bge.{model,batch_size,device}``.

        Returns:
            Configured ``BgeEmbedder``.

        Raises:
            ValueError: If ``baseline_bge`` or any configured field is invalid.
        """
        bge_config = _require_section(config, "baseline_bge")
        return cls(
            model=_read_non_empty_str(bge_config, "model", DEFAULT_BGE_MODEL),
            batch_size=_read_positive_int(bge_config, "batch_size", DEFAULT_BATCH_SIZE),
            device=_read_non_empty_str(bge_config, "device", DEFAULT_DEVICE),
        )

    def __call__(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts into a float32 embedding matrix.

        Args:
            texts: Text inputs with length ``n_texts``.

        Returns:
            Float32 embedding matrix with shape ``(n_texts, dim)``.
        """
        text_list = _validate_texts(texts)
        if not text_list:
            return np.empty((0, self._known_dim()), dtype=np.float32)
        if not self.cache_enabled:
            return self._encode_uncached(text_list)
        return self._encode_with_cache(text_list)

    def _encode_with_cache(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts while reusing cached rows keyed by stable text hashes."""
        keys = [_text_hash(text) for text in texts]
        vectors_by_key: dict[str, FloatMatrix] = {}
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        queued_missing: set[str] = set()

        for key, text in zip(keys, texts, strict=True):
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                vectors_by_key[key] = cached
            elif key not in queued_missing:
                queued_missing.add(key)
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            encoded = self._encode_uncached(missing_texts)
            for key, row in zip(missing_keys, encoded, strict=True):
                vector = np.asarray(row, dtype=np.float32)
                vectors_by_key[key] = vector
                self._cache[key] = vector
                self._cache.move_to_end(key)
            self._trim_cache()

        rows = [vectors_by_key[key] for key in keys]
        return _as_float_matrix(np.vstack(rows), expected_rows=len(texts))

    def _encode_uncached(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts directly with sentence-transformers."""
        model = self._load_model()
        LOGGER.debug("encoding %d texts with %s", len(texts), self.model)
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        matrix = _as_float_matrix(vecs, expected_rows=len(texts))
        self._set_dim(matrix.shape[1])
        return matrix

    def _load_model(self) -> Any:
        """Load and cache the sentence-transformers model."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                LOGGER.warning("requested device %s, but torch.cuda.is_available() is false", self.device)
            LOGGER.info("loading SentenceTransformer model %s on %s", self.model, self.device)
            self._model = SentenceTransformer(self.model, device=self.device)
            inferred_dim = self._model.get_sentence_embedding_dimension()
            if inferred_dim is not None:
                self._set_dim(int(inferred_dim))
        return self._model

    def _known_dim(self) -> int:
        """Return the known embedding dimension, loading a custom model if needed."""
        if self._dim is None:
            model = self._load_model()
            inferred_dim = model.get_sentence_embedding_dimension()
            if inferred_dim is None:
                raise RuntimeError(f"could not infer embedding dimension for {self.model}")
            self._set_dim(int(inferred_dim))
        if self._dim is None:
            raise RuntimeError(f"could not infer embedding dimension for {self.model}")
        return self._dim

    def _set_dim(self, dim: int) -> None:
        """Record the embedding dimension after validating it."""
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        if self._dim is not None and self._dim != dim:
            LOGGER.warning("embedding dimension changed from %d to %d for %s", self._dim, dim, self.model)
        self._dim = dim

    def _trim_cache(self) -> None:
        """Keep only the most recent cached text embeddings."""
        while len(self._cache) > MAX_CACHE_TEXTS:
            self._cache.popitem(last=False)


class HashingEmbedder:
    """Hashing-trick bag-of-words embedder for smoke tests only.

    This fallback is deterministic and CPU-only, but it is not a semantic model and must not
    be used for reported numbers. Output matrices are float32 and have shape
    ``(n_texts, dim)``.
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialize the hashing embedder.

        Args:
            dim: Output embedding dimension.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode texts as normalized hashing-trick bag-of-words vectors.

        Args:
            texts: Text inputs with length ``n_texts``.

        Returns:
            Float32 matrix with shape ``(n_texts, dim)``.
        """
        text_list = _validate_texts(texts)
        matrix = np.zeros((len(text_list), self.dim), dtype=np.float32)
        for row_index, text in enumerate(text_list):
            for token in text.lower().split():
                index, sign = _hash_token(token, self.dim)
                matrix[row_index, index] += sign
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0.0
        matrix[nonzero] /= norms[nonzero, None]
        return matrix.astype(np.float32, copy=False)


def _validate_texts(texts: Sequence[str]) -> list[str]:
    """Validate and materialize a text sequence."""
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    text_list = list(texts)
    for index, text in enumerate(text_list):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be str, got {type(text).__name__}")
    return text_list


def _text_hash(text: str) -> str:
    """Return a stable SHA-256 hash for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    """Map one token to a signed hashing-trick feature."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, byteorder="big", signed=False)
    index = raw % dim
    sign = 1.0 if (raw >> 63) == 0 else -1.0
    return index, sign


def _as_float_matrix(values: object, expected_rows: int) -> FloatMatrix:
    """Convert values to a float32 matrix and validate its row count."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must have shape (n_texts, dim), got {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} embedding rows, got {matrix.shape[0]}")
    if matrix.shape[1] <= 0:
        raise ValueError("embedding dimension must be positive")
    return np.asarray(np.ascontiguousarray(matrix), dtype=np.float32)


def _require_section(config: object, key: str) -> object:
    """Read a required config section from a mapping or object."""
    value = _read_field(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"config.{key} is required")
    return value


def _read_non_empty_str(config: object, key: str, default: str) -> str:
    """Read a non-empty string field from a mapping or object."""
    value = _read_field(config, key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _read_positive_int(config: object, key: str, default: int) -> int:
    """Read a positive integer field from a mapping or object."""
    value = _read_field(config, key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _read_field(config: object, key: str, default: object) -> object:
    """Read a field from a mapping or object, returning ``default`` if absent."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
