"""Integration test for trainer model saving with a real model."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from virtual_stain_flow.losses.wgan_losses import (
    AdversarialLoss,
    GradientPenaltyLoss,
    WassersteinLoss,
)
from virtual_stain_flow.models.discriminator import PatchBasedDiscriminator
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.trainers.logging_gan_trainer import LoggingWGANTrainer
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer


def test_best_unet_round_trip_through_trainer(tmp_path):
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=2,
        depth=2,
        _num_units=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    samples = torch.randn(1, 1, 8, 8)
    loader = DataLoader(TensorDataset(samples, samples), batch_size=1)
    trainer = SingleGeneratorTrainer(
        model=model,
        optimizer=optimizer,
        losses=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
    )

    trainer.update_early_stop_counter()
    assert isinstance(trainer.best_model, UNet)

    with torch.no_grad():
        next(trainer.model.parameters()).add_(1)

    saved_paths = trainer.save_model(tmp_path, best_model=True)
    assert saved_paths is not None

    restored_model = UNet.from_config(model.to_config())
    checkpoint = torch.load(
        saved_paths[0], map_location="cpu", weights_only=True
    )
    restored_model.load_state_dict(checkpoint)

    trainer.best_model.eval()
    restored_model.eval()
    test_input = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        expected = trainer.best_model(test_input)
        actual = restored_model(test_input)

    torch.testing.assert_close(actual, expected)


def test_best_wgan_generator_round_trip_through_trainer(tmp_path):
    generator = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=2,
        depth=2,
        _num_units=1,
    )
    discriminator = PatchBasedDiscriminator(
        in_channels=2,
        base_filters=2,
        n_down_sample_layer=1,
        n_additional_layer=0,
    )
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=2e-4
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=2e-4
    )
    samples = torch.randn(1, 1, 8, 8)
    loader = DataLoader(TensorDataset(samples, samples), batch_size=1)
    trainer = LoggingWGANTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_losses=torch.nn.L1Loss(),
        generator_adverserial_loss=AdversarialLoss(),
        discriminator_loss=WassersteinLoss(),
        discriminator_gradient_penalty_loss=GradientPenaltyLoss(),
        discriminator_gradient_penalty_loss_weight=10.0,
        n_discriminator_steps=5,
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
    )

    trainer.update_early_stop_counter()
    assert isinstance(trainer.best_model, UNet)

    with torch.no_grad():
        next(trainer.model.parameters()).add_(1)

    saved_paths = trainer.save_model(tmp_path, best_model=True)
    assert saved_paths is not None

    restored_generator = UNet.from_config(generator.to_config())
    checkpoint = torch.load(
        saved_paths[0], map_location="cpu", weights_only=True
    )
    restored_generator.load_state_dict(checkpoint)

    trainer.best_model.eval()
    restored_generator.eval()
    test_input = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        expected = trainer.best_model(test_input)
        actual = restored_generator(test_input)

    torch.testing.assert_close(actual, expected)
