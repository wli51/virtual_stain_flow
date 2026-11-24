"""Self-contained NumPy image datasets indexed by tiles.parquet."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseImageDataset
from .ds_engine.npy_manifest import NpyFileState, NpyManifest
from ..transforms.base_transform import LoggableTransform


class NpyImageDataset(BaseImageDataset):
    """Read ``pedcp-numpy-tiles-v1`` exports using local per-FOV tile indices.

    Requires ``export.json`` (NCHW) and ``tiles.parquet``. Select image or mask
    names with the base dataset's input/target channel keys; both default to
    empty selections until explicitly assigned. Raw values are never scaled,
    including by the descriptor's normalization_divisor. Inherited transforms
    and tensor conversion are unchanged, including separate shared-transform
    application to inputs and targets.

    ``tiles`` preserves relative paths and uses tile_id as its index; metadata
    contains every non-path column, including tile_index. Read-only mappings
    are cached by file path (default two, -1 unlimited); each returned CHW tile
    is a fresh writable stack. No ``_SUCCESS`` marker is required.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        input_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        target_channel_keys: Optional[Union[str, Sequence[str]]] = None,
        transforms: Optional[Sequence[LoggableTransform]] = None,
        input_transforms: Optional[Sequence[LoggableTransform]] = None,
        target_transforms: Optional[Sequence[LoggableTransform]] = None,
        cache_capacity: Optional[int] = None,
        check_exists: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        with (self.root_dir / "export.json").open(encoding="utf-8") as descriptor:
            self.export_info = json.load(descriptor)
        if not isinstance(self.export_info, dict):
            raise ValueError("export.json must contain an object.")
        if self.export_info.get("format") != "pedcp-numpy-tiles-v1":
            raise ValueError("Unsupported export format; expected pedcp-numpy-tiles-v1.")
        if self.export_info.get("axes") != "NCHW":
            raise ValueError("Only NCHW exports are supported.")
        self.channel_names = self._read_names("channels")
        self.mask_names = self._read_names("masks")
        names = self.channel_names + self.mask_names
        if not self.channel_names or len(set(names)) != len(names):
            raise ValueError("Export channel names must be nonempty and disjoint from mask names.")

        tiles = pd.read_parquet(self.root_dir / "tiles.parquet")
        if not tiles.columns.is_unique:
            raise ValueError("tiles.parquet must have unique column names.")
        if "tile_id" in tiles.columns:
            tiles = tiles.set_index("tile_id")
        if tiles.index.name != "tile_id":
            raise ValueError("tiles.parquet must contain a tile_id column or named index.")
        if (tiles.empty or tiles.index.hasnans or not tiles.index.is_unique
                or any(isinstance(i, str) and not i.strip() for i in tiles.index)):
            raise ValueError("tile_id must be nonempty, non-null and unique in a non-empty table.")
        path_columns = ["images_path", "masks_path"]
        missing = set(["tile_index", *path_columns]) - set(tiles.columns)
        if missing:
            raise ValueError(f"tiles.parquet is missing required columns: {sorted(missing)}")
        indices = tiles["tile_index"]
        if (not pd.api.types.is_integer_dtype(indices.dtype)
                or pd.api.types.is_bool_dtype(indices.dtype)
                or indices.isna().any() or (indices < 0).any()):
            raise ValueError("tile_index must contain nonnegative integers (not bool).")

        self.tiles = tiles.copy()
        self.metadata = tiles.drop(columns=path_columns).copy()
        resolved = {}
        for column in path_columns:
            for value in tiles[column].unique():
                if value not in resolved:
                    resolved[value] = self._resolve_path(value)
        image_paths = tiles["images_path"].map(resolved)
        mask_paths = tiles["masks_path"].map(resolved)
        file_index = pd.DataFrame(
            {name: image_paths for name in self.channel_names}
            | {name: mask_paths for name in self.mask_names}, index=tiles.index
        )
        super().__init__(
            file_state=NpyFileState(
                NpyManifest(
                    file_index=file_index,
                    check_exists=check_exists,
                    tile_indices=tuple(int(i) for i in indices),
                    channel_names=tuple(self.channel_names),
                    mask_names=tuple(self.mask_names),
                ),
                cache_capacity=cache_capacity,
            ),
            input_channel_keys=input_channel_keys,
            target_channel_keys=target_channel_keys,
            transforms=transforms,
            input_transforms=input_transforms,
            target_transforms=target_transforms,
        )

    def _read_names(self, key: str) -> list[str]:
        names = self.export_info.get(key)
        if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError(f"export.json {key} must be a list of nonempty names.")
        return list(names)

    def _resolve_path(self, value: Any) -> Path:
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Array paths must be non-empty relative paths.")
        path = Path(value)
        resolved = (self.root_dir / path).resolve()
        if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(self.root_dir):
            raise ValueError(f"Array path must stay relative to root_dir: {value}")
        if path.suffix != ".npy":
            raise ValueError(f"Expected a .npy array path, got: {value}")
        if resolved.exists() and not resolved.is_file():
            raise ValueError(f"Array path is not a file: {value}")
        return resolved

    def get_metadata(self, idx: Optional[int] = None) -> pd.Series:
        """Return metadata for a positional index (or last accessed sample).

        The Series name is the tile_id. This does not load image arrays.
        """
        if idx is None:
            idx = self.index_state.last
        if idx is None:
            raise ValueError("No sample accessed yet; provide idx explicitly.")
        return self.metadata.iloc[idx].copy()

    def get_masks(
        self, idx: Optional[int] = None,
        mask_keys: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        """Return a raw CHW mask copy without changing the active sample.

        Masks retain their original labels/dtype and receive no transforms.
        To transform masks as targets, select their names as target_channel_keys.
        """
        if idx is None:
            idx = self.index_state.last
        if idx is None:
            raise ValueError("No sample accessed yet; provide idx explicitly.")
        keys = self.mask_names if mask_keys is None else self._validate_channel_keys(mask_keys)
        if any(key not in self.mask_names for key in keys):
            raise ValueError("mask_keys must contain only export mask names.")
        return self.file_state.read_planes(idx, keys)

    def to_config(self) -> Dict[str, Any]:
        """Serialize constructor options, not absolute per-file paths.

        As with BaseImageDataset, transforms are omitted. Change root_dir in
        the config to relocate a dataset; the parquet index is read again.
        """
        return {
            "root_dir": str(self.root_dir),
            "input_channel_keys": self.input_channel_keys,
            "target_channel_keys": self.target_channel_keys,
            "cache_capacity": self.file_state.cache_capacity,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NpyImageDataset":
        """Reconstruct from a configuration, optionally with a relocated root."""
        if "root_dir" not in config:
            raise ValueError("Configuration must include 'root_dir' for NpyImageDataset.")
        return cls(**config)
