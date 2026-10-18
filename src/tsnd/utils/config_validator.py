"""Validated TSND configuration models.

The schema mirrors `pilot/configs/pilot.yaml`. Downstream artifact readers should
preserve pilot formats: parquet shards per year; `.npz` files with `units` as an
object array shaped `(n_units,)` and `mat` as `float32` shaped `(n_units, dim)`;
and `.pkl` dictionaries with keys `vocab`, `uidx`, `unit_df`, `paper_units`,
`edges`, and `n_papers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
OpenUnitInterval = Annotated[float, Field(gt=0.0, lt=1.0)]


def _strip_non_empty(value: str, field_name: str) -> str:
    """Strip and validate a required string config value."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty.")
    return stripped


def aslist(x: Any) -> list[Any]:
    """Normalize a scalar or missing value to a list.

    Args:
        x: Input value from YAML or artifact metadata.

    Returns:
        `x` unchanged when it is already a list, an empty list for `None`, or a
        one-item list wrapping `x`.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


class BaseTSNDConfig(BaseModel):
    """Common Pydantic options for TSND config sections."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseTSNDConfig):
    """Data acquisition and storage configuration."""

    source: str
    subfield_search: str
    year_start: PositiveInt
    year_end: PositiveInt
    has_abstract: bool
    per_page: PositiveInt
    mailto: str | None = None
    max_papers: PositiveInt
    out_dir: str

    @field_validator("source", "subfield_search", "out_dir")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        """Validate required non-empty string fields."""
        return _strip_non_empty(value, "data string field")

    @field_validator("mailto")
    @classmethod
    def _validate_optional_mailto(cls, value: str | None) -> str | None:
        """Normalize an optional OpenAlex mailto value."""
        if value is None:
            return None
        return _strip_non_empty(value, "mailto")

    @model_validator(mode="after")
    def _validate_year_range(self) -> Self:
        """Validate that the configured data window is ordered."""
        if self.year_end < self.year_start:
            raise ValueError("data.year_end must be greater than or equal to data.year_start.")
        return self


class TermsConfig(BaseTSNDConfig):
    """Term extraction configuration."""

    min_df_per_year: PositiveInt
    top_terms: PositiveInt
    use_spacy_nounchunks: bool
    use_rake: bool


class EmbedConfig(BaseTSNDConfig):
    """Embedding model training configuration."""

    sgns_dim: PositiveInt
    window: PositiveInt
    min_count: PositiveInt
    train_window_years: PositiveInt
    epochs: PositiveInt


class AlignConfig(BaseTSNDConfig):
    """Temporal embedding alignment configuration."""

    k_neighbors: PositiveInt
    tau_sweep: list[UnitInterval] = Field(min_length=1)
    ref_year: PositiveInt

    @field_validator("tau_sweep", mode="before")
    @classmethod
    def _coerce_tau_sweep(cls, value: Any) -> list[Any]:
        """Coerce a scalar stability threshold to a list."""
        return aslist(value)


class NoveltyConfig(BaseTSNDConfig):
    """Novelty score and null-model configuration."""

    delta_relabel: UnitInterval
    n_permutations: PositiveInt
    prox_bins: PositiveInt


class LabelConfig(BaseTSNDConfig):
    """Forward uptake label configuration."""

    horizon: PositiveInt
    breadth_channels: list[str] = Field(min_length=1)
    positive_quantile: OpenUnitInterval

    @field_validator("breadth_channels", mode="before")
    @classmethod
    def _coerce_breadth_channels(cls, value: Any) -> list[Any]:
        """Coerce a single breadth channel to a list."""
        return aslist(value)

    @field_validator("breadth_channels")
    @classmethod
    def _validate_breadth_channels(cls, values: list[str]) -> list[str]:
        """Validate that breadth channel names are non-empty."""
        return [_strip_non_empty(value, "breadth channel") for value in values]


class A10Config(BaseTSNDConfig):
    """Axiom A10 intervention audit configuration."""

    n_injections: PositiveInt
    n_placebo: PositiveInt
    pass_threshold: UnitInterval
    iterate_floor: UnitInterval


class BaselineSpecterConfig(BaseTSNDConfig):
    """SPECTER baseline configuration."""

    model: str
    batch_size: PositiveInt
    device: str

    @field_validator("model", "device")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        """Validate required non-empty string fields."""
        return _strip_non_empty(value, "baseline_specter string field")


class ComputeConfig(BaseTSNDConfig):
    """Compute budget and GPU placement configuration."""

    gpu_index: NonNegativeInt
    gpu_hour_budget: PositiveInt


class Config(BaseTSNDConfig):
    """Top-level TSND run configuration."""

    run_tag: str
    seed_list: list[int] = Field(min_length=1)
    data: DataConfig
    terms: TermsConfig
    embed: EmbedConfig
    align: AlignConfig
    novelty: NoveltyConfig
    label: LabelConfig
    a10: A10Config
    baseline_specter: BaselineSpecterConfig
    compute: ComputeConfig
    domains: list[str] = Field(default_factory=list)
    use_snapshot: bool = False

    @field_validator("run_tag")
    @classmethod
    def _validate_run_tag(cls, value: str) -> str:
        """Validate the run tag."""
        return _strip_non_empty(value, "run_tag")

    @field_validator("seed_list", "domains", mode="before")
    @classmethod
    def _coerce_top_level_lists(cls, value: Any) -> list[Any]:
        """Coerce scalar top-level list fields to lists."""
        return aslist(value)

    @field_validator("domains")
    @classmethod
    def _validate_domains(cls, values: list[str]) -> list[str]:
        """Validate optional domain names."""
        return [_strip_non_empty(value, "domain") for value in values]


def load_config(path: str | Path) -> Config:
    """Load and validate a TSND YAML config file.

    Args:
        path: Path to a YAML config file with the pilot-compatible schema.

    Returns:
        Validated `Config` instance.

    Raises:
        ValueError: If the file cannot be read, parsed, or validated.
    """
    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Failed to read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config file {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file {config_path} must contain a top-level mapping.")

    try:
        return Config.model_validate(cast(dict[str, Any], raw_config))
    except ValidationError as exc:
        raise ValueError(f"Invalid TSND config file {config_path}: {exc}") from exc
