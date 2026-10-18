"""Standalone SPECTER2 paper embedding artifact writer for TSND.

This module writes the B5 static SPECTER nearest-neighbor baseline artifacts
consumed by ``tsnd.models.baselines.specter_nn``. Rows follow
``works_{year}.parquet`` order, ``units`` stores the paper ids from that same
row order, and ``mat`` stores float32 CLS/text embeddings with shape
``(n_papers, dim)``.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

LOG = logging.getLogger(__name__)

Config = Mapping[str, Any]
FloatMatrix = npt.NDArray[np.float32]

DEFAULT_BASELINE_CONFIG_KEY = "specter_nn"
DEFAULT_SPECTER_CONFIG_KEY = "baseline_specter"
DEFAULT_EMBEDDING_DIR = "specter"
DEFAULT_EMBEDDING_PATTERN = "paper_emb_{year}.npz"
DEFAULT_LEGACY_NPY_PATTERN = "paper_emb_{year}.npy"
DEFAULT_WORKS_PATTERN = "works_{year}.parquet"
DEFAULT_MAX_LENGTH = 512
REQUIRED_WORK_COLUMNS = ("id", "abstract")


class ConfigError(ValueError):
    """Raised when the SPECTER embedding config is missing or invalid."""


@dataclass(frozen=True)
class _YearTexts:
    """Paper ids and abstract texts loaded for one year."""

    units: list[str]
    texts: list[str]


class _AdapterSpecterEmbedder:
    """SPECTER2 encoder using the adapters package and CLS token embeddings."""

    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        batch_size: int,
        device: str,
        max_length: int,
    ) -> None:
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode abstracts with SPECTER2 CLS embeddings.

        Args:
            texts: Abstract-only text inputs with length ``n_papers``.

        Returns:
            Float32 embedding matrix with shape ``(n_papers, dim)``.
        """
        _validate_texts(texts)
        torch, tokenizer, model = self._load_model()
        matrices: list[FloatMatrix] = []
        for batch in _batched_texts(texts, self.batch_size):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                hidden_state = outputs[0]
            cls_embeddings = hidden_state[:, 0, :].detach().cpu().float().numpy()
            matrices.append(_as_float_matrix(cls_embeddings, "specter_cls_embeddings"))
        return _as_float_matrix(np.vstack(matrices), "specter_embeddings")

    def _load_model(self) -> tuple[Any, Any, Any]:
        """Load tokenizer, adapters model, and torch lazily."""
        if self._model is None or self._tokenizer is None or self._torch is None:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
            model = AutoAdapterModel.from_pretrained(self.model_id)
            model.load_adapter(self.adapter_id, source="hf", set_active=True)
            model.to(self.device)
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
        return self._torch, self._tokenizer, self._model


class _SentenceTransformerSpecterEmbedder:
    """SPECTER encoder using sentence-transformers."""

    def __init__(self, model_id: str, batch_size: int, device: str) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Encode abstracts with sentence-transformers.

        Args:
            texts: Abstract-only text inputs with length ``n_papers``.

        Returns:
            Float32 embedding matrix with shape ``(n_papers, dim)``.
        """
        _validate_texts(texts)
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            device=self.device,
            show_progress_bar=True,
        )
        return _as_float_matrix(embeddings, "specter_embeddings")

    def _load_model(self) -> Any:
        """Load the sentence-transformers model lazily."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model


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


def generate_specter_embeddings(
    cfg: Config,
    artifact_dir: str | Path,
    data_dir: str | Path,
    *,
    year: int | None = None,
    device: str | None = None,
) -> None:
    """Generate SPECTER paper embedding artifacts for configured years.

    Args:
        cfg: TSND config mapping containing ``data`` and ``baseline_specter``.
        artifact_dir: Artifact root. By default this resolves to
            ``artifacts_{run_tag}``.
        data_dir: Directory containing ``works_{year}.parquet`` inputs.
        year: Optional single target year. If absent, all years from
            ``data.year_start`` through ``data.year_end`` are generated.
        device: Optional device override. If absent, config/default resolution
            chooses ``cuda`` when available, otherwise ``cpu``.
    """
    specter_cfg = _require_mapping(cfg, DEFAULT_SPECTER_CONFIG_KEY)
    baseline_cfg = _optional_mapping(cfg, DEFAULT_BASELINE_CONFIG_KEY)
    resolved_device = _resolve_device(specter_cfg, device)
    embedder = _build_embedder(specter_cfg, resolved_device)
    out_dir = _resolve_embedding_dir(Path(artifact_dir), baseline_cfg)
    pattern = str(baseline_cfg.get("embedding_pattern", DEFAULT_EMBEDDING_PATTERN))
    target_years = _years_from_config(cfg, year)

    LOG.info(
        "writing SPECTER artifacts to %s with pattern %s; legacy loader fallback pattern is %s",
        out_dir,
        pattern,
        DEFAULT_LEGACY_NPY_PATTERN,
    )
    for target_year in target_years:
        out_path = out_dir / pattern.format(year=target_year)
        if out_path.exists():
            LOG.info("skipping year %d, already exists: %s", target_year, out_path)
            continue
        papers = _read_year_texts(Path(data_dir), target_year)
        embeddings = embedder.encode(papers.texts)
        save_embedding_artifact(out_path, papers.units, embeddings)
        LOG.info(
            "wrote year %d SPECTER embeddings with shape %s to %s",
            target_year,
            embeddings.shape,
            out_path,
        )


def save_embedding_artifact(out_path: str | Path, units: Sequence[str], mat: npt.ArrayLike) -> None:
    """Atomically save one SPECTER embedding artifact.

    Args:
        out_path: Target ``paper_emb_{year}.npz`` path.
        units: Paper id array with shape ``(n_papers,)`` in works parquet row
            order.
        mat: Embedding matrix with shape ``(n_papers, dim)``.

    Raises:
        ValueError: If ``units`` and ``mat`` do not have compatible shapes.
    """
    final_path = Path(out_path)
    mat32 = _as_float_matrix(mat, "mat")
    _validate_units(units, expected_rows=mat32.shape[0])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, units=np.array(list(units), dtype=object), mat=mat32)
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main(argv: list[str] | None = None) -> None:
    """Run the standalone SPECTER artifact writer.

    Args:
        argv: Optional CLI argument list.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TSND YAML config.")
    parser.add_argument("--year", type=int, default=None, help="Optional single year to embed.")
    parser.add_argument("--artifact-dir", default=None, help="Override artifact root.")
    parser.add_argument("--data-dir", default=None, help="Override works parquet directory.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_yaml_config(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else resolve_artifact_dir(cfg, args.config)
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir(cfg, args.config)
    generate_specter_embeddings(
        cfg,
        artifact_dir,
        data_dir,
        year=args.year,
        device=args.device,
    )


def _build_embedder(
    specter_cfg: Mapping[str, Any],
    device: str,
) -> _AdapterSpecterEmbedder | _SentenceTransformerSpecterEmbedder:
    model_id = _require_str(specter_cfg, "model")
    batch_size = _require_positive_int(specter_cfg, "batch_size")
    adapter_id = _adapter_from_config(specter_cfg)
    if adapter_id is not None:
        max_length = _optional_positive_int(specter_cfg, "max_length", DEFAULT_MAX_LENGTH)
        LOG.info(
            "using adapters SPECTER backend: model=%s adapter=%s device=%s",
            model_id,
            adapter_id,
            device,
        )
        return _AdapterSpecterEmbedder(
            model_id=model_id,
            adapter_id=adapter_id,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )
    LOG.info("using sentence-transformers SPECTER backend: model=%s device=%s", model_id, device)
    return _SentenceTransformerSpecterEmbedder(model_id=model_id, batch_size=batch_size, device=device)


def _read_year_texts(data_dir: Path, year: int) -> _YearTexts:
    import pandas as pd

    _require_positive_int_value(year, "year")
    works_path = data_dir / DEFAULT_WORKS_PATTERN.format(year=year)
    if not works_path.exists():
        raise FileNotFoundError(f"works parquet not found for SPECTER embedding: {works_path}")
    try:
        frame = pd.read_parquet(works_path, columns=list(REQUIRED_WORK_COLUMNS))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"works parquet must contain columns {REQUIRED_WORK_COLUMNS}: {works_path}") from exc
    missing = [column for column in REQUIRED_WORK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"works parquet missing required columns {missing}: {works_path}")
    if frame["id"].isna().any():
        raise ValueError(f"works parquet contains null paper ids: {works_path}")

    LOG.warning("title unavailable for year %d; using abstract text only", year)
    units = [str(value) for value in frame["id"].tolist()]
    _validate_units(units, expected_rows=len(frame))
    texts = [str(value) for value in frame["abstract"].fillna("").tolist()]
    LOG.info("loaded %d papers for year %d from %s; rows follow parquet order", len(texts), year, works_path)
    return _YearTexts(units=units, texts=texts)


def _resolve_embedding_dir(artifact_dir: Path, baseline_cfg: Mapping[str, Any]) -> Path:
    raw_dir = baseline_cfg.get("embedding_dir")
    if raw_dir is None:
        return artifact_dir / DEFAULT_EMBEDDING_DIR
    path = Path(str(raw_dir))
    return path if path.is_absolute() else artifact_dir / path


def _resolve_device(specter_cfg: Mapping[str, Any], cli_device: str | None) -> str:
    if cli_device is not None:
        return _require_str_value(cli_device, "device")
    raw_device = specter_cfg.get("device")
    if raw_device is not None:
        return _require_str_value(raw_device, "baseline_specter.device")
    return _default_device()


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        LOG.warning("torch unavailable while selecting default device; using cpu")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _adapter_from_config(specter_cfg: Mapping[str, Any]) -> str | None:
    if "adapter" not in specter_cfg or specter_cfg["adapter"] is None:
        return None
    return _require_str_value(specter_cfg["adapter"], "baseline_specter.adapter")


def _years_from_config(cfg: Config, year: int | None) -> list[int]:
    if year is not None:
        _require_positive_int_value(year, "year")
        return [year]
    data_cfg = _require_mapping(cfg, "data")
    year_start = _require_positive_int(data_cfg, "year_start")
    year_end = _require_positive_int(data_cfg, "year_end")
    if year_end < year_start:
        raise ConfigError("data.year_end must be >= data.year_start")
    return list(range(year_start, year_end + 1))


def _resolve_project_root(cfg: Config, config_path: str | Path | None) -> Path:
    raw_root = cfg.get("_root")
    if raw_root is not None:
        return Path(str(raw_root))
    if config_path is not None:
        cfg_path = Path(config_path).resolve()
        if cfg_path.parent.name == "configs":
            return cfg_path.parent.parent
        return cfg_path.parent
    return Path.cwd()


def _batched_texts(texts: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(texts), batch_size):
        yield list(texts[start : start + batch_size])


def _as_float_matrix(value: npt.ArrayLike, name: str) -> FloatMatrix:
    mat = np.asarray(value, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {mat.shape}")
    if mat.shape[0] <= 0 or mat.shape[1] <= 0:
        raise ValueError(f"{name} must have positive shape, got {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} contains non-finite values")
    return mat.astype(np.float32, copy=False)


def _validate_texts(texts: Sequence[str]) -> None:
    if len(texts) == 0:
        raise ValueError("texts must be non-empty")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts must be strings")


def _validate_units(units: Sequence[str], *, expected_rows: int) -> None:
    if len(units) != expected_rows:
        raise ValueError(f"units length must match mat rows: {len(units)} vs {expected_rows}")
    if len(units) == 0:
        raise ValueError("units must be non-empty")
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise ValueError("all units must be non-empty strings")


def _optional_mapping(cfg: Config, key: str) -> Mapping[str, Any]:
    value = cfg.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_mapping(cfg: Config | None, key: str) -> Mapping[str, Any]:
    if cfg is None:
        raise ConfigError(f"{key} config section is required")
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_str(cfg: Mapping[str, Any], key: str) -> str:
    return _require_str_value(cfg.get(key), key)


def _require_str_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> int:
    return _require_positive_int_value(_coerce_int(cfg.get(key), key), key)


def _optional_positive_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    raw_value = cfg.get(key, default)
    return _require_positive_int_value(_coerce_int(raw_value, key), key)


def _require_positive_int_value(value: int, name: str) -> int:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return int(value)


if __name__ == "__main__":
    main()
