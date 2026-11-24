"""Directory exports use local tile indices, not parquet row positions."""

import json
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from virtual_stain_flow.datasets import BaseImageDataset, NpyImageDataset
from virtual_stain_flow.transforms.base_transform import LoggableTransform


@pytest.fixture
def export_root(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    (root / "export.json").write_text(json.dumps({
        "format": "pedcp-numpy-tiles-v1", "axes": "NCHW",
        "channels": ["OrigDNA", "OrigBrightfield"],
        "masks": ["cells", "nuclei"], "normalization_divisor": 65535.0,
    }))
    rows = []
    for fov in range(2):
        folder = root / "fovs" / str(fov)
        folder.mkdir(parents=True)
        images = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5) + fov * 1000
        masks = np.zeros((3, 2, 4, 5), dtype=np.uint8)
        masks[:, 0] = 1
        np.save(folder / "images.npy", images)
        np.save(folder / "masks.npy", masks)
        # Deliberately shuffled and filtered local indices.
        for tile in (2, 0):
            rows.append({
                "tile_id": f"{fov}:{tile}", "tile_index": tile,
                "images_path": f"fovs/{fov}/images.npy",
                "masks_path": f"fovs/{fov}/masks.npy",
                "cell_line": "U2OS", "x": tile * 119,
            })
    pd.DataFrame(rows).to_parquet(root / "tiles.parquet", index=False)
    return root


def make_dataset(root, **kwargs):
    return NpyImageDataset(
        root, input_channel_keys="OrigBrightfield",
        target_channel_keys=["OrigDNA"], **kwargs,
    )


class AffineTransform(LoggableTransform):
    """Small deterministic, in-place transform to exercise cache isolation."""

    def __init__(self, scale=1, offset=0):
        super().__init__(name="affine")
        self.scale, self.offset = scale, offset

    def apply(self, img, **params):
        img *= self.scale
        img += self.offset
        return img

    def to_config(self):
        return {"scale": self.scale, "offset": self.offset}


def test_named_channels_local_indices_and_metadata(export_root):
    ds = make_dataset(export_root)
    assert isinstance(ds, BaseImageDataset)
    assert len(ds) == 4
    assert ds.channel_names == ["OrigDNA", "OrigBrightfield"]
    assert ds.mask_names == ["cells", "nuclei"]
    assert ds.export_info["normalization_divisor"] == 65535.0
    with pytest.raises(ValueError, match="No sample accessed"):
        ds.get_metadata()
    for idx, (fov, tile) in enumerate(((0, 2), (0, 0), (1, 2), (1, 0))):
        expected = np.load(export_root / f"fovs/{fov}/images.npy")[tile]
        inputs, targets = ds.get_raw_item(idx)
        np.testing.assert_array_equal(inputs, expected[[1]])
        np.testing.assert_array_equal(targets, expected[[0]])
        assert inputs.dtype == np.float32
        assert ds.index_state.last == idx
        assert ds.get_metadata().name == f"{fov}:{tile}"
        assert ds.get_metadata()["tile_index"] == tile
        assert ds.get_metadata()["cell_line"] == "U2OS"
    assert set(ds.metadata.columns) == {"tile_index", "cell_line", "x"}
    ds.input_channel_keys = ["OrigDNA", "OrigBrightfield", "OrigDNA"]
    assert ds[0][0].shape == (3, 4, 5)
    with pytest.raises(ValueError, match="not found"):
        ds.target_channel_keys = "unknown"


def test_transforms_order_cache_isolation_and_dataloader(export_root):
    ds = make_dataset(
        export_root, transforms=[AffineTransform(offset=2)],
        input_transforms=[AffineTransform(scale=3)],
        target_transforms=[AffineTransform(offset=7)],
    )
    raw_x, raw_y = ds.get_raw_item(0)
    for _ in range(2):
        x, y = ds[0]
        torch.testing.assert_close(x, torch.from_numpy((raw_x + 2) * 3))
        torch.testing.assert_close(y, torch.from_numpy(raw_y + 9))
    x, y = next(iter(DataLoader(ds, batch_size=2)))
    assert x.shape == y.shape == (2, 1, 4, 5)
    assert x.dtype == y.dtype == torch.float32
    for name in ("transforms", "input_transforms", "target_transforms"):
        with pytest.raises(ValueError, match="LoggableTransform"):
            make_dataset(export_root, **{name: [object()]})


def test_masks_selection_and_copy(export_root):
    ds = make_dataset(export_root)
    ds[0]
    masks = ds.get_masks(2, ["nuclei", "cells"])
    assert masks.dtype == np.uint8
    np.testing.assert_array_equal(masks[:, 0, 0], [0, 1])
    assert ds.index_state.last == 0
    masks[:] = 99
    assert ds.get_masks().max() == 1
    ds.target_channel_keys = ["nuclei", "cells"]
    assert ds.get_raw_item(0)[1].dtype == np.uint8
    assert ds[0][1].dtype == torch.float32
    with pytest.raises(ValueError, match="only export mask"):
        ds.get_masks(mask_keys="OrigDNA")


def test_lazy_mappings_and_lru(export_root):
    with patch("numpy.load", wraps=np.load) as load:
        ds = make_dataset(export_root, cache_capacity=1)
        ds.get_metadata(0)
        assert load.call_count == 0
        ds[0]
        ds[1]
        assert load.call_count == 1
        assert load.call_args.kwargs == {"mmap_mode": "r", "allow_pickle": False}
        assert isinstance(next(iter(ds.file_state._cache.values())), np.memmap)
        ds[2]
        ds[0]
        assert load.call_count == 3
        assert len(ds.file_state._cache) == 1


def test_config_relocation_and_named_parquet_index(export_root, tmp_path):
    table_path = export_root / "tiles.parquet"
    pd.read_parquet(table_path).set_index("tile_id").to_parquet(table_path)
    ds = make_dataset(export_root, cache_capacity=-1)
    config_path = tmp_path / "config.json"
    ds.to_json_config(config_path)
    config = json.loads(config_path.read_text())
    relocated = tmp_path / "relocated"
    shutil.copytree(export_root, relocated)
    config["root_dir"] = str(relocated)
    restored = NpyImageDataset.from_config(config)
    torch.testing.assert_close(restored[0][0], ds[0][0])
    assert all(p.is_relative_to(relocated) for p in restored.file_index.iloc[0])
    assert restored.file_state.cache_capacity == -1


@pytest.mark.parametrize("column,value,message", [
    ("tile_id", None, "tile_id"),
    ("tile_index", -1, "tile_index"),
    ("tile_index", 0.5, "tile_index"),
    ("images_path", "../outside.npy", "relative"),
    ("images_path", "/outside.npy", "relative"),
    ("masks_path", None, "relative paths"),
])
def test_invalid_index(export_root, column, value, message):
    path = export_root / "tiles.parquet"
    table = pd.read_parquet(path)
    table[column] = value
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        make_dataset(export_root)


def test_missing_files_and_tile_bounds(export_root):
    ds = make_dataset(export_root)
    path = export_root / "fovs/0/images.npy"
    np.save(path, np.zeros((1, 2, 4, 5), dtype=np.float32))
    with pytest.raises(IndexError, match="tile_index 2"):
        ds[0]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        make_dataset(export_root, check_exists=True)


@pytest.mark.parametrize("shape,message", [
    ((3, 4, 5), "NCHW"),
    ((3, 1, 4, 5), "Channel count mismatch"),
])
def test_invalid_arrays(export_root, shape, message):
    np.save(export_root / "fovs/0/images.npy", np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match=message):
        make_dataset(export_root)[0]
