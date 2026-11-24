"""NumPy array loading for directory-backed datasets."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .manifest import DatasetManifest, FileState


@dataclass(frozen=True)
class NpyManifest(DatasetManifest):
    """Named planes in per-FOV NCHW arrays; PIL mode is unused."""

    tile_indices: Tuple[int, ...] = ()
    channel_names: Tuple[str, ...] = ()
    mask_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Validate unique paths, not millions of repeated channel/path cells.
        if not isinstance(self.file_index, pd.DataFrame) or self.file_index.empty:
            raise ValueError("file_index must be a non-empty DataFrame.")
        names = self.channel_names + self.mask_names
        if not names or len(set(names)) != len(names):
            raise ValueError("Channel and mask names must be non-empty and distinct.")
        if tuple(self.file_index.columns) != names:
            raise ValueError("file_index columns must match channel and mask names.")
        if len(self.tile_indices) != len(self.file_index) or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0
            for i in self.tile_indices
        ):
            raise ValueError("tile_indices must contain one nonnegative integer per row.")
        paths = pd.unique(self.file_index.to_numpy().ravel())
        for path in paths:
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("file_index must contain absolute Path objects.")
            if self.check_exists and not path.is_file():
                raise FileNotFoundError(f"Array file not found: {path}")

    def read_image(self, path: Path) -> np.ndarray:
        """Map an NCHW numeric array read-only without scaling or unpickling."""
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            arr.close()
            raise ValueError(f"Expected a single .npy array: {path}")
        if arr.dtype.kind not in "uif" or arr.ndim != 4 or 0 in arr.shape:
            raise ValueError(f"Expected a nonempty real numeric NCHW array: {path}, {arr.shape}")
        return arr

    def get_image_dimensions(
        self, idx: int, channels: Optional[Sequence[str]] = None
    ) -> Tuple[Tuple[int, ...], ...]:
        """Return (height, width) for each selected named plane."""
        shapes = {}
        paths = self.get_paths_for_keys(idx, channels or self.channel_keys)
        for path in dict.fromkeys(paths):
            shapes[path] = self.read_image(path).shape[-2:]
        return tuple(
            shapes[path] for path in paths
        )


class NpyFileState(FileState):
    """Reuse FileState's path-keyed LRU; return independent CHW tile stacks."""

    manifest: NpyManifest

    def __post_init__(self) -> None:
        if self.cache_capacity is None:
            self.cache_capacity = 2
        if isinstance(self.cache_capacity, bool) or not isinstance(self.cache_capacity, Integral):
            raise ValueError("cache_capacity must be -1 or a positive integer.")
        super().__post_init__()

    def read_planes(self, idx: int, keys: Sequence[str]) -> np.ndarray:
        """Select planes without changing the active sample state."""
        if not keys:
            raise ValueError("Select at least one named channel or mask.")
        planes = []
        tile_index = self.manifest.tile_indices[idx]
        paths = self.manifest.get_paths_for_keys(idx, keys)
        for key, path in zip(keys, paths):
            names = (self.manifest.channel_names if key in self.manifest.channel_names
                     else self.manifest.mask_names)
            arr = self._get_or_load(path)
            if arr.shape[1] != len(names):
                raise ValueError(f"Channel count mismatch for {path}: expected {len(names)}, got {arr.shape[1]}")
            if tile_index >= arr.shape[0]:
                raise IndexError(f"tile_index {tile_index} out of bounds for {path} with {arr.shape[0]} tiles")
            planes.append(arr[tile_index, names.index(key)])
        self._ensure_same_spatial_shape(planes, paths)
        return np.stack(planes, axis=0)

    def update(self, idx: int, input_keys: Sequence[str], target_keys: Sequence[str]) -> None:
        if not input_keys or not target_keys:
            raise ValueError("Set input_channel_keys and target_channel_keys before reading samples.")
        inputs = self.read_planes(idx, input_keys)
        targets = self.read_planes(idx, target_keys)
        if inputs.shape[-2:] != targets.shape[-2:]:
            raise ValueError("Spatial shape mismatch between input and target arrays.")
        self.input_image_raw, self.target_image_raw = inputs, targets
        self.input_paths = self.manifest.get_paths_for_keys(idx, input_keys)
        self.target_paths = self.manifest.get_paths_for_keys(idx, target_keys)
