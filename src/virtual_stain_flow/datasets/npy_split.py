"""Metadata-only, replicate-held-out splits for NumPy tile exports."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union
import warnings

import numpy as np
import pandas as pd

from .ds_engine.npy_manifest import NpyIndices, read_npy_tiles, validate_npy_indices


class NpyDatasetSplit(NamedTuple):
    """Four lists of root parquet positions; supports tuple unpacking."""

    train_indices: list[int]
    test_indices: list[int]
    train_plot_indices: list[int]
    test_plot_indices: list[int]


def _columns(value, name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if (not isinstance(value, Sequence) or not value
            or any(not isinstance(c, str) or not c.strip() for c in value)):
        raise ValueError(f"{name} must contain nonempty column names.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate column names.")
    return list(value)


def _positive_integer(value, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer" + (" or None." if optional else "."))


def _sample(pool: list[int], size: Optional[int], rng: np.random.Generator) -> list[int]:
    if size is None or size >= len(pool):
        return list(pool)
    return [int(i) for i in rng.choice(pool, size=int(size), replace=False)]


def split_npy_dataset(
    root_dir: Union[str, Path],
    *,
    group_columns: Sequence[str] = ("seeding_density", "cell_line"),
    replicate_columns: Union[str, Sequence[str]] = "well",
    n_holdout: int = 1,
    seed: int = 42,
    indices: Optional[NpyIndices] = None,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    plot_size: Optional[int] = 5,
    plot_column: Optional[str] = "all_fluorescence_nonconstant",
    plot_filters: Optional[Mapping[str, Any]] = None,
) -> NpyDatasetSplit:
    """Hold out replicates within each metadata group, without loading arrays.

    All indices are root parquet positions, never tile IDs or local tile_index.
    Each group needs at least n_holdout + 1 unique replicate tuples. With the
    default, matching well labels across plates are one replicate; explicitly
    pass ["plate", "well"] for plate-specific wells. Shared images_path values
    across full assignments raise, even if size caps would hide the leakage.

    Train/test sizes are positive integer caps (None means all). Plot pools use
    full assignments, strict boolean plot_column and scalar-equality filters;
    None disables plot_column. Empty plot pools warn and return []. Five RNG
    streams isolate holdout, train, test and the two plot selections. Sorted
    group/replicate keys and root positions give reproducible selections for an
    unchanged parquet; metadata keys must be mutually sortable within columns.
    """
    groups = _columns(group_columns, "group_columns")
    replicates = _columns(replicate_columns, "replicate_columns")
    if set(groups) & set(replicates):
        raise ValueError("group_columns and replicate_columns must not overlap.")
    _positive_integer(n_holdout, "n_holdout")
    for name, size in (("train_size", train_size), ("test_size", test_size), ("plot_size", plot_size)):
        _positive_integer(size, name, optional=True)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer (not bool).")
    if plot_column is not None and (not isinstance(plot_column, str) or not plot_column.strip()):
        raise ValueError("plot_column must be a nonempty column name or None.")
    if plot_filters is None:
        plot_filters = {}
    if not isinstance(plot_filters, Mapping):
        raise ValueError("plot_filters must be a mapping of metadata names to scalar values.")
    for column, value in plot_filters.items():
        if (not isinstance(column, str) or not column.strip()
                or not pd.api.types.is_scalar(value)):
            raise ValueError("plot_filters must map nonempty metadata names to scalar values.")

    root = Path(root_dir).expanduser().resolve()
    tiles = read_npy_tiles(root)
    positions = np.asarray(sorted(validate_npy_indices(indices, len(tiles))), dtype=np.int64)
    tiles = tiles.iloc[positions]
    required = set(groups + replicates + ["images_path"] + list(plot_filters))
    if plot_column is not None:
        required.add(plot_column)
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"tiles.parquet is missing required columns: {sorted(missing)}")
    for column in groups + replicates:
        if tiles[column].isna().any():
            raise ValueError(f"Grouping/replicate column {column!r} contains missing/NA values.")
    if any(not isinstance(p, (str, Path)) or not str(p) for p in tiles["images_path"]):
        raise ValueError("images_path must contain nonempty paths.")
    # Resolve aliases once per distinct FOV, without reading its image array.
    paths = {p: (root / p).resolve() for p in tiles["images_path"].unique()}
    image_paths = tiles["images_path"].map(paths)

    plot_mask = np.ones(len(tiles), dtype=bool)
    if plot_column is not None:
        flags = tiles[plot_column]
        if not pd.api.types.is_bool_dtype(flags.dtype) or flags.isna().any():
            raise ValueError(f"plot_column {plot_column!r} must contain non-null boolean values.")
        plot_mask &= flags.to_numpy(dtype=bool)
    for column, value in plot_filters.items():
        plot_mask &= tiles[column].eq(value).fillna(False).to_numpy(dtype=bool)

    holdout_rng, train_rng, test_rng, train_plot_rng, test_plot_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(5)
    )
    test_mask = np.zeros(len(tiles), dtype=bool)
    try:
        grouped = tiles.groupby(groups, sort=True, observed=True).indices
        for group in sorted(grouped):
            local_positions = np.asarray(grouped[group], dtype=np.int64)
            replicate_rows = tiles.iloc[local_positions].groupby(
                replicates, sort=True, observed=True,
            ).indices
            keys = sorted(replicate_rows)
            if len(keys) < n_holdout + 1:
                raise ValueError(
                    f"Group {group!r} has {len(keys)} unique replicates; "
                    f"requires at least {n_holdout + 1} for n_holdout={n_holdout}."
                )
            for choice in holdout_rng.choice(len(keys), size=int(n_holdout), replace=False):
                test_mask[local_positions[replicate_rows[keys[int(choice)]]]] = True
    except TypeError as exc:
        raise ValueError("Grouping/replicate values must be hashable and sortable.") from exc

    train_paths = set(image_paths.iloc[np.flatnonzero(~test_mask)])
    test_paths = set(image_paths.iloc[np.flatnonzero(test_mask)])
    if train_paths & test_paths:
        raise ValueError("Train/test leakage: images_path (same FOV) is shared across assignments.")

    train = positions[~test_mask].tolist()
    test = positions[test_mask].tolist()
    train_plot = positions[~test_mask & plot_mask].tolist()
    test_plot = positions[test_mask & plot_mask].tolist()
    for name, pool in (("train", train_plot), ("test", test_plot)):
        if not pool:
            warnings.warn(f"Empty {name} plot pool after metadata filtering; returning [].", UserWarning, stacklevel=2)
    return NpyDatasetSplit(
        _sample(train, train_size, train_rng),
        _sample(test, test_size, test_rng),
        _sample(train_plot, plot_size, train_plot_rng),
        _sample(test_plot, plot_size, test_plot_rng),
    )
