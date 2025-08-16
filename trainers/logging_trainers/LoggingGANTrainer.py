import pathlib
from collections import defaultdict
from typing import Optional, Sequence, Union, Dict

import numpy as np
import torch

from .AbstractLoggingTrainer import AbstractLoggingTrainer
from ...losses.AbstractLoss import AbstractLoss
from ...losses.wgan_losses import (
    GradientPenaltyLoss, 
    AdveserialGeneratorLoss,
    WassersteinDiscriminatorLoss 
)

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
            self._generator_reconstruction_loss_fn = generator_reconstruction_loss_fn
        elif isinstance(generator_reconstruction_loss_fn, torch.nn.Module):
            self._generator_reconstruction_loss_fn = [generator_reconstruction_loss_fn]
        else:
            raise TypeError(
                "generator_reconstruction_loss_fn must be a torch.nn.Module or a list of torch.nn.Modules."
            )
        
        if generator_adversarial_loss_fn is None:
            self._generator_adversarial_loss_fn = NoAdversarialGeneratorLoss()
        elif isinstance(generator_adversarial_loss_fn, torch.nn.Module):
            self._generator_adversarial_loss_fn = generator_adversarial_loss_fn
        else:
            raise TypeError(
                "generator_adversarial_loss_fn must be a torch.nn.Module or None."
            )

        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer
        self._discriminator_loss_fn = discriminator_loss_fn

        if gradient_penalty_loss_fn is None:
            self._gradient_penalty_loss_fn = NoGradientPenaltyLoss()
        elif isinstance(gradient_penalty_loss_fn, torch.nn.Module):
            self._gradient_penalty_loss_fn = gradient_penalty_loss_fn
        else:
            raise TypeError(
                "gradient_penalty_loss_fn must be a torch.nn.Module or None."
            )

        if not(generator_update_freq == 1) or \
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
        
        self._last_discriminator_losses = {
            self._get_loss_name(self._discriminator_loss_fn): 
            torch.tensor(0.0, device=self.device).detach(),
            self._get_loss_name(self._gradient_penalty_loss_fn):
            torch.tensor(0.0, device=self.device).detach(),
        }

        self._last_generator_losses = {
            **{self._get_loss_name(loss_fn): 
               torch.tensor(0.0, device=self.device).detach()
            for loss_fn in self._generator_reconstruction_loss_fn},
            self._get_loss_name(self._generator_adversarial_loss_fn): 
                torch.tensor(0.0, device=self.device).detach(),
        }
        
    def _train_discriminator_step(
        self,
        inputs: torch.tensor, 
        targets: torch.tensor,
        pred: Optional[torch.tensor] = None,
    ) -> Dict[str, np.float]:
        """
        Internal helper method to perform a single training step
        for the discriminator. Expects the inputs and targets
        to be already moved to the device. If the predicted images
        are not provided, they are generated using the generator.
        """
        if self.epoch % self.discriminator_update_freq == 0:            

            if pred is None:
                pred = self._generator(inputs)

            self._discriminator_optimizer.zero_grad()

            # the discriminator is trained on real and fake images stacked
            # with the real inputs, this results in (2* B, C, H, W) tensors
            real_target_input_stack = torch.cat(
                (targets, inputs), dim=1)            
            # here the predicted images are the GAN "fake" images
            fake_target_input_stack = torch.cat(
                (pred, inputs), dim=1)
            
            # discriminator predicts the prob of given stack being real
            discriminator_real_as_real_prob = self._discriminator(
                real_target_input_stack)
            discriminator_fake_as_real_prob = self._discriminator(
                fake_target_input_stack)

            discriminator_loss = self._discriminator_loss_fn(
                discriminator_real_as_real_prob,
                discriminator_fake_as_real_prob, 
            )

            if self._gradient_penalty_loss_fn is not None:
                gp_loss = self._gradient_penalty_loss_fn(
                    real_target_input_stack,
                    fake_target_input_stack,
                )
            else:
                gp_loss = torch.tensor(0.0, device=self.device)

            total_discriminator_loss =  discriminator_loss + gp_loss
            total_discriminator_loss.backward()
            self._discriminator_optimizer.step()

            # update the last discriminator loss and gradient penalty loss
            self._last_discriminator_losses[
                self._get_loss_name(self._discriminator_loss_fn)] = \
                discriminator_loss.detach().item()
            self._last_discriminator_losses[
                self._get_loss_name(self._gradient_penalty_loss_fn)] = \
                gp_loss.detach().item()

        return self._last_discriminator_losses

    def _train_generator_step(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
        pred: Optional[torch.tensor] = None,
    ) -> Dict[str, np.float]:
        """
        Internal helper method to perform a single training step
        for the generator. Expects the inputs and targets
        to be already moved to the device.
        """

        if self.epoch % self.generator_update_freq == 0:

            if pred is None:
                pred = self._generator(inputs)

            self._generator_optimizer.zero_grad()
            fake_as_real_prob = self._discriminator(
                torch.cat((pred, inputs), dim=1)
            )
            
            total_loss = torch.tensor(0.0, device=self.device)
            
            for loss_fn in self._generator_reconstruction_loss_fn:
                loss_fn_name = self._get_loss_name(loss_fn)
                _loss = loss_fn(targets, pred)
                total_loss += _loss
                self._last_generator_losses[loss_fn_name] = \
                    _loss.detach().item()

            _adv_loss = self._generator_adversarial_loss_fn(fake_as_real_prob)
            total_loss += _adv_loss
            self._last_generator_losses[
                self._get_loss_name(self._generator_adversarial_loss_fn)] = \
                _adv_loss.detach().item()

            total_loss.backward()
            self._generator_optimizer.step()        

        return self._last_generator_losses
        
    def train_step(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
    ):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # generation forward pass is always needed regardless of
        # whether the generator is updated. 
        predicted_images = self._generator(inputs)

        all_loss_dict = {}

        disc_loss_dict = self._train_discriminator_step(
            inputs=inputs, targets=targets, pred=predicted_images
        )

        gen_loss_dict = self._train_generator_step(
            inputs=inputs, targets=targets, pred=predicted_images
        )

        return {
            **disc_loss_dict,
            **gen_loss_dict,
        }
    
    def evaluate_step(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
    ):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self._generator.eval()
        self._discriminator.eval()

        all_loss_dict = {}

        with torch.no_grad():
            
            pred = self._generator(inputs)

            real_target_input_stack = torch.cat(
                (targets, inputs), dim=1)
            fake_target_input_stack = torch.cat(
                (pred, inputs), dim=1)
            discriminator_real_as_real_prob = self._discriminator(
                real_target_input_stack)
            discriminator_fake_as_real_prob = self._discriminator(
                fake_target_input_stack)
            
            discriminator_loss = self._discriminator_loss_fn(
                discriminator_real_as_real_prob,
                discriminator_fake_as_real_prob, 
            )
            
            all_loss_dict[self._get_loss_name(
                self._discriminator_loss_fn
            )] = \
                discriminator_loss.item()

            if self._gradient_penalty_loss_fn is not None:
                gp_loss = self._gradient_penalty_loss_fn(
                    real_target_input_stack,
                    fake_target_input_stack,
                )
            else:
                gp_loss = torch.tensor(0.0, device=self.device)
                
            all_loss_dict[self._get_loss_name(
                self._gradient_penalty_loss_fn
            )] = \
                gp_loss.item()
            
            for loss_fn in self._generator_reconstruction_loss_fn:
                loss_fn_name = self._get_loss_name(loss_fn)
                _loss = loss_fn(targets, pred)
                all_loss_dict[loss_fn_name] = _loss.item()
            
            discriminator_fake_as_real_prob = self._discriminator(
                torch.cat((pred, inputs), dim=1)
            )
            _gen_adv_loss = self._generator_adversarial_loss_fn(
                discriminator_fake_as_real_prob
            )
            all_loss_dict[self._get_loss_name(
                self._generator_adversarial_loss_fn
            )] = _gen_adv_loss.item()
            
            for _, metric in self.metrics.items():
                metric.update(pred, targets, validation=True)

            return all_loss_dict
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        
        self._generator.train()
        self._discriminator.train()

        all_unagg_loss_dict = defaultdict(list)
        for inputs, targets in self._train_loader:
            batch_loss = self.train_step(inputs, targets)
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
        for inputs, targets in self._val_loader:
            batch_loss = self.evaluate_step(inputs, targets)
            for key, value in batch_loss.items():
                all_unagg_loss_dict[key].append(value)

        all_agg_loss_dict = {}
        for key, loss_list in all_unagg_loss_dict.items():
            all_agg_loss_dict[key] = sum(loss_list) / len(loss_list)
        
        return all_agg_loss_dict
    
    @property
    def generator(self):
        return self._generator
    
    @property
    def generator_optimizer(self):
        return self._generator_optimizer
    
    @property
    def model(self):
        return {
            "generator": self._generator,
            "discriminator": self._discriminator
        }
    
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