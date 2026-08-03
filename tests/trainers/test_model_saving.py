import pathlib

import pytest
import torch

from virtual_stain_flow.trainers.model_saving import (
    build_weight_filename,
    clone_model,
    save_model_weights,
    select_model_for_saving,
)


class SaveableLinear(torch.nn.Linear):
    def save_weights(self, filename, dir):
        path = pathlib.Path(dir) / filename
        torch.save(self.state_dict(), path)
        return path


class NonCloneableModel(torch.nn.Module):
    def __deepcopy__(self, memo):
        raise TypeError("cannot clone")


def test_clone_model_is_independent():
    model = SaveableLinear(2, 1)
    snapshot = clone_model(model)

    with torch.no_grad():
        model.weight.add_(1)

    assert snapshot is not model
    assert not torch.equal(snapshot.weight, model.weight)


def test_clone_model_reports_non_cloneable_model():
    model = NonCloneableModel()

    with pytest.raises(RuntimeError, match="independent snapshot"):
        clone_model(model)


def test_select_model_for_saving_prefers_requested_snapshot():
    model = SaveableLinear(2, 1)
    snapshot = clone_model(model)

    assert select_model_for_saving(model, snapshot, True) is snapshot
    assert select_model_for_saving(model, snapshot, False) is model


def test_build_weight_filename_preserves_components():
    assert build_weight_filename("generator", "weights_best", ".pth") == (
        "generator_weights_best.pth"
    )


def test_save_model_weights_delegates_to_model(tmp_path):
    model = SaveableLinear(2, 1)

    path = save_model_weights(
        model=model,
        save_path=tmp_path,
        prefix="generator",
        suffix="weights_best",
        file_ext=".pth",
    )

    assert path == tmp_path / "generator_weights_best.pth"
    assert path.exists()
