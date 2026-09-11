"""Tests for AbstractTrainer save-related class methods."""

import torch

from virtual_stain_flow.trainers.AbstractTrainer import AbstractTrainer


class MinimalArtifactTrainer(AbstractTrainer):
    """Concrete trainer realization to exercise base save methods."""

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        return {"loss": torch.tensor(0.0)}

    def evaluate_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        return {"loss": torch.tensor(0.0)}


def test_save_model_writes_current_model_file(
    mock_model_with_save,
    mock_optimizer,
    train_dataloader,
    val_dataloader,
    tmp_path,
):
    trainer = MinimalArtifactTrainer(
        model=mock_model_with_save,
        optimizer=mock_optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=torch.device("cpu"),
    )

    saved_paths = trainer.save_model(save_path=tmp_path, best_model=False)

    assert isinstance(saved_paths, list)
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].name == "generator_weights_0.pth"


def test_save_optimizer_state_writes_recent_file(
    mock_model_with_save,
    mock_optimizer,
    train_dataloader,
    val_dataloader,
    tmp_path,
):
    trainer = MinimalArtifactTrainer(
        model=mock_model_with_save,
        optimizer=mock_optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=torch.device("cpu"),
    )

    saved_paths = trainer.save_optimizer_state(save_path=tmp_path, recent=True)

    assert isinstance(saved_paths, list)
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].name == "optimizer_recent.pth"


def test_save_optimizer_state_non_recent_not_supported(
    mock_model_with_save,
    mock_optimizer,
    train_dataloader,
    val_dataloader,
    tmp_path,
):
    trainer = MinimalArtifactTrainer(
        model=mock_model_with_save,
        optimizer=mock_optimizer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=torch.device("cpu"),
    )

    import pytest

    with pytest.raises(NotImplementedError, match="non-recent optimizer states"):
        trainer.save_optimizer_state(save_path=tmp_path, recent=False)
