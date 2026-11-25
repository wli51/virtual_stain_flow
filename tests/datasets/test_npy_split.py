"""Focused metadata-only replicate splits in root parquet coordinates."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from virtual_stain_flow.datasets import NpyDatasetSplit, split_npy_dataset


@pytest.fixture
def split_root(tmp_path):
    rows = []
    for density in (100, 1000):
        for cell_line in ("U2OS", "A549"):
            for plate in ("P2", "P1"):
                for well in ("B02", "A01", "C03"):
                    fov = f"{density}/{cell_line}/{plate}/{well}"
                    for tile in (2, 0):
                        rows.append({
                            "tile_id": f"{fov}:{tile}", "tile_index": tile,
                            "images_path": f"{fov}/images.npy", "masks_path": f"{fov}/masks.npy",
                            "seeding_density": density, "cell_line": cell_line,
                            "plate": plate, "well": well,
                            "all_fluorescence_nonconstant": tile == 0,
                        })
    pd.DataFrame(rows).sample(frac=1, random_state=9).to_parquet(
        tmp_path / "tiles.parquet", index=False,
    )
    # No descriptor or array files: splitting must only need parquet metadata.
    return tmp_path


@pytest.mark.parametrize("replicates,n_holdout", [("well", 1), (["plate", "well"], 2)])
def test_group_holdout_composite_replicates_root_positions_and_lazy(split_root, replicates, n_holdout):
    table = pd.read_parquet(split_root / "tiles.parquet")
    table.set_index("tile_id").to_parquet(split_root / "tiles.parquet")
    with patch("numpy.load", side_effect=AssertionError("no array loads")):
        result = split_npy_dataset(split_root, replicate_columns=replicates, n_holdout=n_holdout)
    assert isinstance(result, NpyDatasetSplit)
    train, test, train_plot, test_plot = result
    assert all(isinstance(i, int) for pool in result for i in pool)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(len(table)))
    assert set(table.iloc[train].images_path).isdisjoint(table.iloc[test].images_path)
    columns = [replicates] if isinstance(replicates, str) else replicates
    for _, group in table.groupby(["seeding_density", "cell_line"]):
        group_train = group.loc[group.index.isin(train)]
        group_test = group.loc[group.index.isin(test)]
        train_reps = set(group_train[columns].itertuples(index=False, name=None))
        test_reps = set(group_test[columns].itertuples(index=False, name=None))
        assert len(test_reps) == n_holdout
        assert train_reps.isdisjoint(test_reps)
        if replicates == "well":
            assert group_test.groupby("well").plate.nunique().eq(2).all()
    for pool, assigned in ((train_plot, train), (test_plot, test)):
        assert len(pool) == 5
        assert set(pool) <= set(assigned)
        assert table.iloc[pool].all_fluorescence_nonconstant.all()


def test_reproducibility_independent_caps_plot_filters_and_eligible_indices(split_root):
    full = split_npy_dataset(split_root, plot_filters={"seeding_density": 100}, plot_size=None)
    capped = split_npy_dataset(
        split_root, train_size=1, test_size=2, plot_size=None,
        plot_filters={"seeding_density": 100},
    )
    assert len(capped.train_indices) == 1
    assert len(capped.test_indices) == 2
    assert capped.train_plot_indices == full.train_plot_indices
    assert capped.test_plot_indices == full.test_plot_indices
    assert len(capped.train_plot_indices) > len(capped.train_indices)
    assert len(capped.test_plot_indices) > len(capped.test_indices)
    small_plots = split_npy_dataset(split_root, plot_size=2)
    train_cap = split_npy_dataset(split_root, train_size=1, plot_size=2)
    assert train_cap.test_indices == small_plots.test_indices
    assert train_cap.train_plot_indices == small_plots.train_plot_indices
    assert train_cap.test_plot_indices == small_plots.test_plot_indices
    assert small_plots == split_npy_dataset(split_root, plot_size=2)
    assert small_plots != split_npy_dataset(split_root, seed=43, plot_size=2)
    assert split_npy_dataset(split_root, train_size=1000, test_size=1000, plot_size=2) == small_plots
    table = pd.read_parquet(split_root / "tiles.parquet")
    assert set(table.iloc[full.train_indices + full.test_indices].seeding_density) == {100, 1000}
    for pool in (full.train_plot_indices, full.test_plot_indices):
        assert table.iloc[pool].seeding_density.eq(100).all()
    eligible = table.index[table.seeding_density.eq(1000)]
    filtered = split_npy_dataset(split_root, indices=np.array(eligible[::-1]), plot_size=None)
    assert sorted(filtered.train_indices + filtered.test_indices) == eligible.tolist()
    assert filtered == split_npy_dataset(split_root, indices=eligible, plot_size=None)


def test_empty_plot_pool_warns_and_explicit_flag_opt_out(split_root):
    path = split_root / "tiles.parquet"
    table = pd.read_parquet(path)
    table["all_fluorescence_nonconstant"] = False
    table.to_parquet(path, index=False)
    with pytest.warns(UserWarning, match="Empty .* plot pool") as record:
        result = split_npy_dataset(split_root)
    assert len(record) == 2
    assert result.train_plot_indices == result.test_plot_indices == []
    table = table.drop(columns="all_fluorescence_nonconstant")
    table.to_parquet(path, index=False)
    result = split_npy_dataset(split_root, plot_column=None, plot_size=None)
    assert result.train_plot_indices == result.train_indices
    assert result.test_plot_indices == result.test_indices
    with pytest.warns(UserWarning, match="Empty .* plot pool"):
        empty = split_npy_dataset(split_root, plot_column=None, plot_filters={"seeding_density": -1})
    assert empty.train_plot_indices == empty.test_plot_indices == []
    assert empty.train_indices == result.train_indices


@pytest.mark.parametrize("column,change,message", [
    ("cell_line", "drop", "missing required columns"),
    ("well", "null", "missing/NA"),
    ("seeding_density", "null", "missing/NA"),
    ("all_fluorescence_nonconstant", "drop", "missing required columns"),
    ("all_fluorescence_nonconstant", "null", "non-null boolean"),
    ("all_fluorescence_nonconstant", "integer", "non-null boolean"),
    ("tile_id", "duplicate", "tile_id"),
])
def test_missing_or_invalid_schema(split_root, column, change, message):
    path = split_root / "tiles.parquet"
    table = pd.read_parquet(path)
    if change == "drop":
        table = table.drop(columns=column)
    elif change == "null":
        if column == "all_fluorescence_nonconstant":
            table[column] = table[column].astype("boolean")
        table.loc[0, column] = None
    elif change == "integer":
        table[column] = table[column].astype(int)
    else:
        table.loc[0, column] = table.loc[1, column]
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        split_npy_dataset(split_root)


@pytest.mark.parametrize("kwargs,message", [
    ({"n_holdout": 3}, "requires at least 4"),
    ({"n_holdout": True}, "positive integer"),
    ({"group_columns": ["well"]}, "overlap"),
    ({"replicate_columns": ["well", "well"]}, "duplicate column"),
    ({"group_columns": []}, "nonempty column"),
    ({"train_size": 0}, "positive integer"),
    ({"test_size": 1.5}, "positive integer"),
    ({"plot_size": -1}, "positive integer"),
    ({"seed": True}, "nonnegative integer"),
    ({"indices": [True]}, "not bool"),
    ({"indices": []}, "non-empty"),
    ({"indices": [0, 0]}, "duplicates"),
    ({"plot_filters": {"seeding_density": [100]}}, "scalar"),
    ({"plot_filters": {"missing_density": 100}}, "missing required columns"),
])
def test_invalid_split_parameters(split_root, kwargs, message):
    with pytest.raises(ValueError, match=message):
        split_npy_dataset(split_root, **kwargs)


def test_fov_leakage_checked_before_caps(split_root):
    path = split_root / "tiles.parquet"
    table = pd.read_parquet(path)
    # Every replicate points to one FOV; alias spelling must not hide leakage.
    table["images_path"] = np.where(table.well.eq("A01"), "same/images.npy", "same/./images.npy")
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="leakage.*same FOV"):
        split_npy_dataset(split_root, train_size=1, test_size=1)
