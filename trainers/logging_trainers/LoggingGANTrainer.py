import pathlib
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict, Any

import numpy as np
import torch

from .AbstractLoggingTrainer import AbstractLoggingTrainer, MlflowLogger
from ...losses.AbstractLoss import AbstractLoss
from ...losses.wgan_losses import (
    GradientPenaltyLoss, 
    AdveserialGeneratorLoss,
    WassersteinDiscriminatorLoss 
)
from ...losses.loss_item_group import LossItem, LossGroup

path_type = Union[pathlib.Path, str]
"""
Trainer class for logging capabilities, for 
training a Generator + Adversarial Discriminator pair.
Intended to be used with the WGAN infranstructure of the
package.
"""
class LoggingGANTrainer(AbstractLoggingTrainer):
    
    def __init__(
        self,
        generator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        generator_reconstruction_loss_fn: Sequence[torch.nn.Module],
        discriminator: torch.nn.Module,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_loss_fn: torch.nn.Module = WassersteinDiscriminatorLoss(),
        generator_update_freq: int = 5,        
        generator_reconstruction_loss_weights: Sequence[float] = None,
        generator_adversarial_loss_fn: Optional[torch.nn.Module] = AdveserialGeneratorLoss(),
        discriminator_update_freq: int = 1,
        gradient_penalty_loss_fn: Optional[torch.nn.Module] = GradientPenaltyLoss(),
        **kwargs
    ):

        super().__init__(**kwargs)

        self._generator = generator
        self._generator_optimizer = generator_optimizer

        if isinstance(generator_reconstruction_loss_fn, Sequence):
            if not all(isinstance(fn, torch.nn.Module) for fn in generator_reconstruction_loss_fn):
                raise TypeError(
                    "All elements in generator_reconstruction_loss_fn must be torch.nn.Modules."
                )
        elif isinstance(generator_reconstruction_loss_fn, torch.nn.Module):
            generator_reconstruction_loss_fn = [generator_reconstruction_loss_fn]
        else:
            raise TypeError(
                "generator_reconstruction_loss_fn must be a torch.nn.Module or a list of torch.nn.Modules."
            )
        
        if generator_adversarial_loss_fn is None:
            generator_adversarial_loss_fn = NoAdversarialGeneratorLoss()
        elif not isinstance(generator_adversarial_loss_fn, torch.nn.Module):
            raise TypeError(
                "generator_adversarial_loss_fn must be a torch.nn.Module or None."
            )
        
        if generator_reconstruction_loss_weights is None:
            generator_reconstruction_loss_weights = [1.0] * len(generator_reconstruction_loss_fn)
        elif len(generator_reconstruction_loss_weights) != len(generator_reconstruction_loss_fn):
            raise ValueError(
                "generator_reconstruction_loss_weights must have the same length as generator_reconstruction_loss_fn."
            )

        self.gen_loss_group = LossGroup(
            "generator_losses",
            [
                *[LossItem(module=loss_fn, args=("target", "pred"), weight=weight)
                 for loss_fn, weight in zip(generator_reconstruction_loss_fn, generator_reconstruction_loss_weights)],
                LossItem(
                    module=generator_adversarial_loss_fn,
                    args=("discriminator_fake_as_real_prob",),
                    weight=1.0)
            ]
        )

        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer

        if not isinstance(discriminator_loss_fn, torch.nn.Module):
            raise TypeError(
                "discriminator_loss_fn must be a torch.nn.Module."
            )

        if gradient_penalty_loss_fn is None:
            gradient_penalty_loss_fn = NoGradientPenaltyLoss()
        elif isinstance(gradient_penalty_loss_fn, torch.nn.Module):
            gradient_penalty_loss_fn.trainer = self # allow access to device
        else:
            raise TypeError(
                "gradient_penalty_loss_fn must be a torch.nn.Module or None."
            )
        
        self.disc_loss_group = LossGroup(
            "discriminator_losses",
            [
                LossItem(
                    module=discriminator_loss_fn,
                    args=("discriminator_real_as_real_prob", 
                          "discriminator_fake_as_real_prob"),
                    weight=1.0
                ),
                LossItem(
                    module=gradient_penalty_loss_fn,
                    args=("real_target_input_stack", 
                          "fake_target_input_stack",
                          "discriminator"),
                    weight=1.0,
                    compute_at=('train',) # only compute during training
                )
            ]
        )

        if not(generator_update_freq == 1) and \
            not (discriminator_update_freq == 1):
            raise ValueError(
                "One of the generator or discriminator update frequencies must "
                "be set to 1. "
            )
        self ._generator_update_freq = generator_update_freq
        self._discriminator_update_freq = discriminator_update_freq

        # Due to the asynchronous nature of GAN generator updates, 
        # we keep track of last values of the generator and discriminator losses
        # so that we log the last updated loss for generator and discriminator
        # during epochs they are not updated. Alternative strategy would
        # be to modify the logging system to only log loss values during
        # updates but that would introduce additional complexity. 
        zero = 0.0
        self._last_discriminator_losses = {
            k: zero for k in self.disc_loss_group.keys}
        self._last_generator_losses = {
            k: zero for k in self.gen_loss_group.keys}
        
        # track generator update counts
        self._global_step = 0
        
    def _train_discriminator_step(
        self,
        input: torch.tensor, 
        target: torch.tensor,
    ) -> Dict[str, float]:
        """
        Internal helper method to perform a single training step
        for the discriminator. Expects the input and target
        to be already moved to the device. If the predicted images
        are not provided, they are generated using the generator.
        """

        # batch-wise update control
        if self._global_step % self._discriminator_update_freq == 0:          

            self._discriminator.train()
            self._discriminator_optimizer.zero_grad()

            with torch.no_grad():
                pred = self._generator(input)

            # the discriminator is trained on real and fake images stacked
            # with the real input, this results in (B, IN_C + TARGET_C, H, W) tensors
            real_target_input_stack = torch.cat([target, input], dim=1)
            fake_target_input_stack = torch.cat([pred, input], dim=1)
            
            # discriminator predicts the prob of given stack being real
            discriminator_real_as_real_prob = self._discriminator(
                real_target_input_stack)
            discriminator_fake_as_real_prob = self._discriminator(
                fake_target_input_stack)
            
            ctx = {
                'real_target_input_stack': real_target_input_stack,
                'fake_target_input_stack': fake_target_input_stack,
                'discriminator_real_as_real_prob': discriminator_real_as_real_prob,
                'discriminator_fake_as_real_prob': discriminator_fake_as_real_prob,
                'discriminator': self._discriminator
            }

            total_disc_loss, logs = self.disc_loss_group(ctx, phase='train')
            total_disc_loss.backward()
            self._discriminator_optimizer.step()

            self._last_discriminator_losses = logs

            return logs
        
        else:
            return self._last_discriminator_losses

    def _train_generator_step(
        self,
        input: torch.tensor,
        target: torch.tensor,
    ) -> Dict[str, float]:
        """
        Internal helper method to perform a single training step
        for the generator. Expects the input and target
        to be already moved to the device.
        """

        if self._global_step % self._discriminator_update_freq == 0:

            self._generator.train()
            self._generator_optimizer.zero_grad()
            
            pred = self._generator(input)

            fake_target_input_stack = torch.cat(
                (pred, input), dim=1)    
            discriminator_fake_as_real_prob = self._discriminator(
                fake_target_input_stack
            )

            ctx = {
                'target': target,
                'pred': pred,
                'discriminator_fake_as_real_prob': discriminator_fake_as_real_prob
            }

            total_gen_loss, logs = self.gen_loss_group(ctx, phase='train')
            total_gen_loss.backward()
            self._generator_optimizer.step()
            
            self._last_generator_losses = logs

            return logs
        
        else:
            return self._last_generator_losses
        
    def train_step(
        self,
        input: torch.tensor,
        target: torch.tensor,
    ):
        input, target = input.to(self.device), target.to(self.device)

        disc_loss_dict = self._train_discriminator_step(
            input=input, target=target
        )

        gen_loss_dict = self._train_generator_step(
            input=input, target=target
        )

        self._global_step += 1

        self._generator.eval()
        self._discriminator.eval()
        with torch.no_grad():
            pred = self._generator(input)
            for _, metric in self.metrics.items():
                metric.update(pred, target, validation=False)

        return {
            **disc_loss_dict,
            **gen_loss_dict,
        }
    
    def evaluate_step(
        self,
        input: torch.tensor,
        target: torch.tensor,
    ):
        input, target = input.to(self.device), target.to(self.device)

        self._generator.eval()
        self._discriminator.eval()

        with torch.no_grad():
            
            pred = self._generator(input)

            real_target_input_stack = torch.cat(
                (target, input), dim=1)
            fake_target_input_stack = torch.cat(
                (pred, input), dim=1)
            discriminator_real_as_real_prob = self._discriminator(
                real_target_input_stack)
            discriminator_fake_as_real_prob = self._discriminator(
                fake_target_input_stack)

            _, disc_logs = self.disc_loss_group({
                'real_target_input_stack': real_target_input_stack,
                'fake_target_input_stack': fake_target_input_stack,
                'discriminator_real_as_real_prob': discriminator_real_as_real_prob,
                'discriminator_fake_as_real_prob': discriminator_fake_as_real_prob,
                'discriminator': self._discriminator
                }, phase='eval')

            _, gen_logs = self.gen_loss_group({
                'target': target,
                'pred': pred,
                'discriminator_fake_as_real_prob': discriminator_fake_as_real_prob
            }, phase='eval')
                        
            for _, metric in self.metrics.items():
                metric.update(pred, target, validation=True)

            return {**disc_logs, **gen_logs}
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        
        self._generator.train()
        self._discriminator.train()

        all_unagg_loss_dict = defaultdict(list)
        for input, target in self._train_loader:
            batch_loss = self.train_step(input, target)
            for key, value in batch_loss.items():
                all_unagg_loss_dict[key].append(value)

        all_agg_loss_dict = {}
        for key, loss_list in all_unagg_loss_dict.items():
            all_agg_loss_dict[key] = sum(loss_list) / len(loss_list)

        return all_agg_loss_dict
    
    def evaluate_epoch(self):
        """
        Evaluate the model for one epoch.
        """
        
        self._generator.eval()
        self._discriminator.eval()

        all_unagg_loss_dict = defaultdict(list)
        for input, target in self._val_loader:
            batch_loss = self.evaluate_step(input, target)
            for key, value in batch_loss.items():
                all_unagg_loss_dict[key].append(value)

        all_agg_loss_dict = {}
        for key, loss_list in all_unagg_loss_dict.items():
            all_agg_loss_dict[key] = sum(loss_list) / len(loss_list)
        
        return all_agg_loss_dict
    
    def train(
        self,
        logger: MlflowLogger
    ):
        self._generator.to(self.device)
        self._discriminator.to(self.device)

        # superclass implements generic logging training framework and
        # takes over from here
        super().train(logger)
    
    @property
    def generator(self):
        return self._generator
    
    @property
    def generator_optimizer(self):
        return self._generator_optimizer
    
    @property
    def model(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator
    
    @property
    def generator_update_freq(self):
        return self._generator_update_freq
    
    @property
    def discriminator_update_freq(self):
        return self._discriminator_update_freq
    
class NoGradientPenaltyLoss(AbstractLoss):

    def __init__(
        self
    ):
        super().__init__("NoGradientPenaltyLoss")

    def forward(
        self,
        **kwargs
    ):
        """
        No-op for gradient penalty loss when not used.
        """
        return torch.tensor(0.0, device=self.trainer.device)
    
class NoAdversarialGeneratorLoss(AbstractLoss):

    def __init__(
        self
    ):
        super().__init__("NoAdversarialGeneratorLoss")

    def forward(
        self,
        **kwargs
    ):
        """
        No-op for adversarial generator loss when not used.
        """
        return torch.tensor(0.0, device=self.trainer.device)