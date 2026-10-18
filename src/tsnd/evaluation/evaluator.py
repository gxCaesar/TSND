"""Evaluation harness for TSND front-forecasting experiments."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tsnd.evaluation.labels import build_labels
from tsnd.models.decomposition import (
    lexical_emergence as LE,
)
from tsnd.models.decomposition import (
    neighborhood_shift as NS,
)
from tsnd.models.decomposition import (
    recombination as RC,
)
from tsnd.training.forecaster import Forecaster, ForecasterSettings

LOGGER = logging.getLogger(__name__)


def _format_path(path: Sequence[str]) -> str:
    """Format a config path.

    Args:
        path: Config path components.

    Returns:
        Dot-delimited config path.
    """
    return ".".join(path)


def _lookup_config(config: object, path: Sequence[str]) -> Any:
    """Look up a nested config value in mappings or objects.

    Args:
        config: Mapping-style or object-style config.
        path: Nested path components.

    Returns:
        Config value.

    Raises:
        KeyError: If a path component is missing.
    """
    current: Any = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                raise KeyError(_format_path(path))
            current = current[key]
        else:
            if not hasattr(current, key):
                raise KeyError(_format_path(path))
            current = getattr(current, key)
    return current


def _first_config_value(config: object, paths: Sequence[Sequence[str]]) -> Any:
    """Return the first available config value.

    Args:
        config: Mapping-style or object-style config.
        paths: Candidate config paths.

    Returns:
        First value found.

    Raises:
        KeyError: If no candidate path exists.
    """
    for path in paths:
        try:
            return _lookup_config(config, path)
        except KeyError:
            continue
    joined = " or ".join(_format_path(path) for path in paths)
    raise KeyError(f"Missing required config value: {joined}")


def _as_str(value: Any, name: str) -> str:
    """Coerce a config value to string.

    Args:
        value: Config value.
        name: Human-readable config path.

    Returns:
        String value.

    Raises:
        TypeError: If the value is not a string.
    """
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a string; got {type(value).__name__}.")


def _as_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Coerce a config value to a string tuple.

    Args:
        value: Sequence config value.
        name: Human-readable config path.

    Returns:
        Tuple of strings.

    Raises:
        TypeError: If the value is not a non-string sequence.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    return tuple(str(item) for item in value)


def _as_str_mapping(value: Any, name: str) -> dict[str, str]:
    """Coerce a config value to a string dictionary.

    Args:
        value: Mapping config value.
        name: Human-readable config path.

    Returns:
        Dictionary with string keys and values.

    Raises:
        TypeError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}


@dataclass(frozen=True)
class EvaluatorSettings:
    """Resolved evaluator settings.

    Attributes:
        merge_columns: Columns used to join fronts, labels, controls, and
            baselines. Shape is (n_merge_columns,).
        key_column: Stable front identifier column.
        label_column: Binary uptake label column.
        decomposition_columns: Mapping from ``le``, ``ns``, and ``rc`` to output
            column names.
        control_columns: Control feature columns required in the assembled
            table.
        baseline_columns: Baseline feature columns required in the assembled
            table.
    """

    merge_columns: tuple[str, ...]
    key_column: str
    label_column: str
    decomposition_columns: dict[str, str]
    control_columns: tuple[str, ...]
    baseline_columns: tuple[str, ...]

    @classmethod
    def from_config(cls, config: object) -> EvaluatorSettings:
        """Resolve evaluator settings from config.

        Args:
            config: Mapping-style or object-style config.

        Returns:
            Resolved evaluator settings.
        """
        merge_columns = _as_str_tuple(
            _first_config_value(config, (("evaluation", "merge_columns"),)),
            "evaluation.merge_columns",
        )
        if not merge_columns:
            raise ValueError("evaluation.merge_columns must not be empty.")
        key_column = _as_str(
            _first_config_value(config, (("evaluation", "key_column"),)),
            "evaluation.key_column",
        )
        label_column = _as_str(
            _first_config_value(config, (("forecaster", "label_column"),)),
            "forecaster.label_column",
        )
        decomposition_columns = _as_str_mapping(
            _first_config_value(config, (("evaluation", "decomposition_columns"),)),
            "evaluation.decomposition_columns",
        )
        missing_components = {"le", "ns", "rc"} - set(decomposition_columns)
        if missing_components:
            raise ValueError(
                "evaluation.decomposition_columns must define le/ns/rc; "
                f"missing {sorted(missing_components)}."
            )
        control_columns = _as_str_tuple(
            _first_config_value(config, (("evaluation", "control_columns"),)),
            "evaluation.control_columns",
        )
        baseline_columns = _as_str_tuple(
            _first_config_value(config, (("evaluation", "baseline_columns"),)),
            "evaluation.baseline_columns",
        )
        return cls(
            merge_columns=merge_columns,
            key_column=key_column,
            label_column=label_column,
            decomposition_columns=decomposition_columns,
            control_columns=control_columns,
            baseline_columns=baseline_columns,
        )


def _missing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Find missing columns in a DataFrame.

    Args:
        frame: DataFrame with shape (n_fronts, n_columns).
        columns: Required column names.

    Returns:
        Missing column names.
    """
    return [column for column in columns if column not in frame.columns]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    """Require columns in a DataFrame.

    Args:
        frame: DataFrame with shape (n_fronts, n_columns).
        columns: Required column names.
        context: Human-readable table context.

    Raises:
        ValueError: If any column is missing.
    """
    missing = _missing_columns(frame, columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}.")


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    positional: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Call a function with only supported keyword arguments.

    Args:
        func: Callable to invoke.
        positional: Positional fallback arguments.
        kwargs: Candidate keyword arguments.

    Returns:
        Callable result.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_kwargs:
        return func(**kwargs)

    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    required_positionals = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Signature.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in accepted
    ]
    if required_positionals:
        return func(*positional, **accepted)
    return func(**accepted)


def _invoke_component(
    component: Any,
    fronts: pd.DataFrame,
    config: object,
    seed: int | None,
    component_name: str,
    component_inputs: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke an imported decomposition component.

    Args:
        component: Imported LE/NS/RC function, class, or object.
        fronts: Front table with shape (n_fronts, n_columns).
        config: Mapping-style or object-style config.
        seed: Optional seed for seed-dependent components.
        component_name: Component name used for diagnostics.
        component_inputs: Optional keyword inputs for the imported decomposition
            function. For the existing TSND module these are arrays such as
            ``new_vecs_aligned`` and ``prior_matrix``.

    Returns:
        Raw component output.

    Raises:
        TypeError: If the component cannot be called.
    """
    kwargs: dict[str, Any] = {
        "fronts": fronts,
        "frame": fronts,
        "table": fronts,
        "config": config,
        "seed": seed,
        "component": component_name,
    }
    if component_inputs is not None:
        kwargs.update(component_inputs)
    if isinstance(component, type):
        instance = _call_with_supported_kwargs(component, (), kwargs)
        return _invoke_component(instance, fronts, config, seed, component_name)

    for method_name in ("transform", "score", "compute", "build"):
        method = getattr(component, method_name, None)
        if callable(method):
            return _call_with_supported_kwargs(method, (fronts,), kwargs)

    if callable(component):
        return _call_with_supported_kwargs(component, (fronts,), kwargs)
    raise TypeError(f"Decomposition component {component_name} is not callable.")


def _series_from_component_output(
    output: Any,
    fronts: pd.DataFrame,
    column_name: str,
    settings: EvaluatorSettings,
) -> pd.Series:
    """Normalize a component output to a front-aligned Series.

    Args:
        output: Component output as Series, DataFrame, mapping, or array.
        fronts: Front table with shape (n_fronts, n_columns).
        column_name: Desired output column name.
        settings: Resolved evaluator settings.

    Returns:
        Series aligned to ``fronts.index`` with shape (n_fronts,).

    Raises:
        ValueError: If the output cannot be aligned.
    """
    if isinstance(output, pd.Series):
        if output.index.equals(fronts.index):
            return output.astype(float)
        if len(output) == len(fronts):
            return pd.Series(output.to_numpy(dtype=float), index=fronts.index, name=column_name)
        raise ValueError(f"Component output Series for {column_name} is not front-aligned.")

    if isinstance(output, pd.DataFrame):
        if column_name not in output.columns:
            numeric_columns = list(output.select_dtypes(include=[np.number]).columns)
            if len(numeric_columns) != 1:
                raise ValueError(
                    f"Component DataFrame for {column_name} must contain that column "
                    "or exactly one numeric score column."
                )
            column_name = str(numeric_columns[0])
        if set(settings.merge_columns).issubset(output.columns) and set(
            settings.merge_columns
        ).issubset(fronts.columns):
            merged = fronts.loc[:, list(settings.merge_columns)].merge(
                output.loc[:, list(settings.merge_columns) + [column_name]],
                on=list(settings.merge_columns),
                how="left",
                validate="one_to_one",
            )
            return pd.Series(merged[column_name].to_numpy(dtype=float), index=fronts.index)
        if len(output) == len(fronts):
            return pd.Series(output[column_name].to_numpy(dtype=float), index=fronts.index)
        raise ValueError(f"Component DataFrame for {column_name} cannot be aligned.")

    if isinstance(output, Mapping):
        _require_columns(fronts, [settings.key_column], "front table")
        mapped = fronts[settings.key_column].map(output)
        return pd.Series(mapped.to_numpy(dtype=float), index=fronts.index, name=column_name)

    array = np.asarray(output)
    if array.ndim != 1:
        raise ValueError(f"Component output for {column_name} must be one-dimensional.")
    if array.shape[0] != len(fronts):
        raise ValueError(
            f"Component output length for {column_name} must match fronts; "
            f"got {array.shape[0]} and {len(fronts)}."
        )
    return pd.Series(array.astype(float), index=fronts.index, name=column_name)


def _merge_feature_block(
    table: pd.DataFrame,
    block: pd.DataFrame | None,
    required_columns: Sequence[str],
    settings: EvaluatorSettings,
    context: str,
) -> pd.DataFrame:
    """Merge or validate a control/baseline feature block.

    Args:
        table: Current feature table with shape (n_fronts, n_columns).
        block: Optional block table with shape (n_fronts, n_block_columns).
        required_columns: Feature columns expected after assembly.
        settings: Resolved evaluator settings.
        context: Human-readable block name.

    Returns:
        Updated feature table.

    Raises:
        ValueError: If required columns cannot be found or merged.
    """
    if not required_columns:
        return table
    if block is None:
        _require_columns(table, required_columns, context)
        return table

    _require_columns(block, list(settings.merge_columns) + list(required_columns), context)
    merged = table.merge(
        block.loc[:, list(settings.merge_columns) + list(required_columns)],
        on=list(settings.merge_columns),
        how="left",
        validate="one_to_one",
    )
    return merged


def _attach_labels(
    table: pd.DataFrame,
    labels: pd.DataFrame | pd.Series | ArrayLike,
    settings: EvaluatorSettings,
) -> pd.DataFrame:
    """Attach binary labels to an assembled feature table.

    Args:
        table: Feature table with shape (n_fronts, n_columns).
        labels: Labels as DataFrame, Series, or array with shape (n_fronts,).
        settings: Resolved evaluator settings.

    Returns:
        Feature table containing ``settings.label_column``.

    Raises:
        ValueError: If labels cannot be aligned.
    """
    if isinstance(labels, pd.DataFrame):
        _require_columns(labels, [settings.label_column], "labels")
        if set(settings.merge_columns).issubset(labels.columns):
            return table.merge(
                labels.loc[:, list(settings.merge_columns) + [settings.label_column]],
                on=list(settings.merge_columns),
                how="left",
                validate="one_to_one",
            )
        if len(labels) == len(table):
            updated = table.copy()
            updated[settings.label_column] = labels[settings.label_column].to_numpy()
            return updated
        raise ValueError("Label DataFrame cannot be aligned to feature table.")

    if isinstance(labels, pd.Series):
        if labels.index.equals(table.index):
            updated = table.copy()
            updated[settings.label_column] = labels.to_numpy()
            return updated
        if len(labels) == len(table):
            updated = table.copy()
            updated[settings.label_column] = labels.to_numpy()
            return updated
        raise ValueError("Label Series cannot be aligned to feature table.")

    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != len(table):
        raise ValueError("Label array must have shape (n_fronts,).")
    updated = table.copy()
    updated[settings.label_column] = array
    return updated


def _call_build_labels(fronts: pd.DataFrame, config: object, seed: int | None) -> Any:
    """Call the imported label builder with supported arguments.

    Args:
        fronts: Front table with shape (n_fronts, n_columns).
        config: Mapping-style or object-style config.
        seed: Optional seed.

    Returns:
        Raw label builder output.
    """
    return _call_with_supported_kwargs(
        build_labels,
        (fronts,),
        {"fronts": fronts, "frame": fronts, "table": fronts, "config": config, "seed": seed},
    )


def _value_for_seed(value: Any, seed: int) -> Any:
    """Select a seed-specific value when a mapping is provided.

    Args:
        value: Plain value or mapping keyed by seed.
        seed: Random seed.

    Returns:
        Seed-specific value or the original value.

    Raises:
        KeyError: If a seed mapping lacks the requested seed.
    """
    if isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
        if seed not in value:
            raise KeyError(f"Missing seed {seed} in seed-specific input.")
        return value[seed]
    return value


class Evaluator:
    """Assemble TSND feature tables and run the configured forecaster."""

    def __init__(self, config: object) -> None:
        """Create an evaluator.

        Args:
            config: Mapping-style or object-style config.
        """
        self.config = config
        self.settings = EvaluatorSettings.from_config(config)
        self.forecaster_settings = ForecasterSettings.from_config(config)
        self.forecaster = Forecaster(config)

    def assemble_feature_table(
        self,
        fronts: pd.DataFrame,
        *,
        controls: pd.DataFrame | None = None,
        baselines: pd.DataFrame | None = None,
        labels: pd.DataFrame | pd.Series | ArrayLike | None = None,
        decomposition_inputs: Mapping[str, Mapping[str, Any]] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Assemble a front-level feature table.

        Args:
            fronts: Candidate front table with shape (n_fronts, n_columns). It
                must contain the merge/key columns and any precomputed feature
                columns not supplied by ``controls`` or ``baselines``.
            controls: Optional control feature block with shape
                (n_fronts, n_control_columns).
            baselines: Optional baseline feature block with shape
                (n_fronts, n_baseline_columns).
            labels: Optional labels. If omitted, ``build_labels`` is called.
            decomposition_inputs: Optional mapping from ``le``, ``ns``, and
                ``rc`` to keyword inputs accepted by the imported decomposition
                functions. If omitted, decomposition columns must already exist
                in ``fronts``.
            seed: Optional seed for seed-dependent decomposition calls.

        Returns:
            Assembled table with shape (n_fronts, n_features + metadata).
        """
        _require_columns(fronts, list(self.settings.merge_columns), "front table")
        table = fronts.copy()

        components = {
            "le": LE,
            "ns": NS,
            "rc": RC,
        }
        for component_name, component in components.items():
            output_column = self.settings.decomposition_columns[component_name]
            if output_column in table.columns:
                continue
            if decomposition_inputs is None or component_name not in decomposition_inputs:
                raise ValueError(
                    f"Missing decomposition column {output_column!r}. Provide it in fronts "
                    f"or pass decomposition_inputs[{component_name!r}] for the imported "
                    "tsnd.models.decomposition function."
                )
            raw_scores = _invoke_component(
                component,
                table,
                self.config,
                seed,
                component_name,
                decomposition_inputs[component_name],
            )
            table[output_column] = _series_from_component_output(
                raw_scores,
                table,
                output_column,
                self.settings,
            )

        table = _merge_feature_block(
            table,
            controls,
            self.settings.control_columns,
            self.settings,
            "control features",
        )
        table = _merge_feature_block(
            table,
            baselines,
            self.settings.baseline_columns,
            self.settings,
            "baseline features",
        )

        if labels is None:
            if self.settings.label_column in table.columns:
                labels = table[self.settings.label_column]
            else:
                labels = _call_build_labels(table, self.config, seed)
        table = _attach_labels(table, labels, self.settings)
        _require_columns(
            table,
            list(self.settings.decomposition_columns.values())
            + list(self.settings.control_columns)
            + list(self.settings.baseline_columns)
            + [self.settings.label_column],
            "assembled feature table",
        )
        return table

    def evaluate(
        self,
        train_fronts: pd.DataFrame | Mapping[int, pd.DataFrame],
        test_fronts: pd.DataFrame | Mapping[int, pd.DataFrame],
        *,
        train_controls: pd.DataFrame | Mapping[int, pd.DataFrame] | None = None,
        test_controls: pd.DataFrame | Mapping[int, pd.DataFrame] | None = None,
        train_baselines: pd.DataFrame | Mapping[int, pd.DataFrame] | None = None,
        test_baselines: pd.DataFrame | Mapping[int, pd.DataFrame] | None = None,
        train_labels: pd.DataFrame | pd.Series | ArrayLike | Mapping[int, Any] | None = None,
        test_labels: pd.DataFrame | pd.Series | ArrayLike | Mapping[int, Any] | None = None,
        train_decomposition_inputs: Mapping[int, Mapping[str, Mapping[str, Any]]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
        test_decomposition_inputs: Mapping[int, Mapping[str, Mapping[str, Any]]]
        | Mapping[str, Mapping[str, Any]]
        | None = None,
    ) -> dict[str, Any]:
        """Assemble train/test tables per seed and run the forecaster.

        Args:
            train_fronts: Training fronts, either one table or a seed-keyed
                mapping. Table shape is (n_train_fronts, n_columns).
            test_fronts: Test fronts, either one table or a seed-keyed mapping.
                Table shape is (n_test_fronts, n_columns).
            train_controls: Optional training control features.
            test_controls: Optional test control features.
            train_baselines: Optional training baseline features.
            test_baselines: Optional test baseline features.
            train_labels: Optional training labels.
            test_labels: Optional test labels.
            train_decomposition_inputs: Optional train decomposition function
                inputs, either one mapping or a seed-keyed mapping.
            test_decomposition_inputs: Optional test decomposition function
                inputs, either one mapping or a seed-keyed mapping.

        Returns:
            Results dictionary emitted by ``tsnd.training.forecaster``.
        """
        table_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

        def provider(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Build or retrieve seed-specific train/test tables.

            Args:
                seed: Random seed.

            Returns:
                ``(train, test)`` feature tables.
            """
            if seed not in table_cache:
                train_table = self.assemble_feature_table(
                    _value_for_seed(train_fronts, seed),
                    controls=_value_for_seed(train_controls, seed)
                    if train_controls is not None
                    else None,
                    baselines=_value_for_seed(train_baselines, seed)
                    if train_baselines is not None
                    else None,
                    labels=_value_for_seed(train_labels, seed) if train_labels is not None else None,
                    decomposition_inputs=_value_for_seed(train_decomposition_inputs, seed)
                    if train_decomposition_inputs is not None
                    else None,
                    seed=seed,
                )
                test_table = self.assemble_feature_table(
                    _value_for_seed(test_fronts, seed),
                    controls=_value_for_seed(test_controls, seed) if test_controls is not None else None,
                    baselines=_value_for_seed(test_baselines, seed)
                    if test_baselines is not None
                    else None,
                    labels=_value_for_seed(test_labels, seed) if test_labels is not None else None,
                    decomposition_inputs=_value_for_seed(test_decomposition_inputs, seed)
                    if test_decomposition_inputs is not None
                    else None,
                    seed=seed,
                )
                table_cache[seed] = (train_table, test_table)
            return table_cache[seed]

        results = self.forecaster.fit_evaluate_by_seed(provider)
        results["assembled_feature_columns"] = {
            "decomposition": dict(self.settings.decomposition_columns),
            "controls": list(self.settings.control_columns),
            "baselines": list(self.settings.baseline_columns),
        }
        return results

    def run(
        self,
        train_fronts: pd.DataFrame | Mapping[int, pd.DataFrame],
        test_fronts: pd.DataFrame | Mapping[int, pd.DataFrame],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for ``evaluate``.

        Args:
            train_fronts: Training fronts with shape (n_train_fronts, n_columns)
                or a seed-keyed mapping of such tables.
            test_fronts: Test fronts with shape (n_test_fronts, n_columns) or a
                seed-keyed mapping of such tables.
            **kwargs: Optional feature and label blocks accepted by
                ``evaluate``.

        Returns:
            Results dictionary emitted by ``evaluate``.
        """
        return self.evaluate(train_fronts, test_fronts, **kwargs)
