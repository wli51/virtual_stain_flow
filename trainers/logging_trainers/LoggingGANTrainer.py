import pathlib
from collections import defaultdict
from typing import Optional, List, Union

import torch
from torch.utils.data import DataLoader, random_split

from .AbstractLoggingTrainer import AbstractLoggingTrainer 

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
        generator_loss_fn: torch.nn.Module,
        discriminator: torch.nn.Module,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_loss_fn: torch.nn.Module,
        gradient_penalty_loss_fn: Optional[torch.nn.Module] = None,
        generator_update_freq: int = 5,
        discriminator_update_freq: int = 1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self._generator = generator
        self._generator_optimizer = generator_optimizer
        self._generator_loss_fn = generator_loss_fn

        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer
        self._discriminator_loss_fn = discriminator_loss_fn

        self._gradient_penalty_loss_fn = gradient_penalty_loss_fn

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
        self._last_discriminator_loss = torch.tensor(
            0.0, device=self.device).detach()
        self._last_gradient_penalty_loss = torch.tensor(
            0.0, device=self.device).detach()
        self._last_generator_loss = torch.tensor(
            0.0, device=self.device).detach()
        
    def _train_discriminator_step(
        self,
        inputs: torch.tensor, 
        targets: torch.tensor,
        pred: Optional[torch.tensor] = None,
    ):
        """
        Internal helper method to perform a single training step
        for the discriminator. Expects the inputs and targets
        to be already moved to the device. If the predicted images
        are not provided, they are generated using the generator.
        """
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
        self._last_discriminator_loss = discriminator_loss.detach()
        self._last_gradient_penalty_loss = gp_loss.detach()

        return {
            self.discriminator_loss_name:
                self._last_discriminator_loss.item(),
            self.gradient_penalty_loss_name:
                self._last_gradient_penalty_loss.item(),
        }

    def _train_generator_step(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
        pred: Optional[torch.tensor] = None,
    ):
        """
        Internal helper method to perform a single training step
        for the generator. Expects the inputs and targets
        to be already moved to the device.
        """
        if pred is None:
            pred = self._generator(inputs)

        self._generator_optimizer.zero_grad()
        fake_as_real_prob = self._discriminator(
            torch.cat((pred, inputs), dim=1)
        )
        generator_loss = self._generator_loss_fn(
            fake_as_real_prob, targets, pred
        )
        generator_loss.backward()
        self._generator_optimizer.step()

        # update the last generator loss
        self._last_generator_loss = generator_loss.detach()
        return {
            self.generator_loss_name:
                self._last_generator_loss.item(),
        }
        
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

        if self.epoch % self.discriminator_update_freq:
            loss_dict = self._train_discriminator_step(
                inputs=inputs, targets=targets, pred=predicted_images
            )
            all_loss_dict.update(loss_dict)
        else:
            all_loss_dict.update({
                type(self._discriminator_loss_fn).__name__:
                    self._last_discriminator_loss.item(),
                type(self._gradient_penalty_loss_fn).__name__:
                    self._last_gradient_penalty_loss.item(),
            })

        if self.epoch % self.generator_update_freq:
            loss_dict = self._train_generator_step(
                inputs=inputs, targets=targets, pred=predicted_images
            )
            all_loss_dict.update(loss_dict)
        else:
            all_loss_dict.update({
                type(self._generator_loss_fn).__name__:
                    self._last_generator_loss.item(),
            })

        return all_loss_dict
    
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
            
            all_loss_dict[self.discriminator_loss_name] = \
                discriminator_loss.item()

            if self._gradient_penalty_loss_fn is not None:
                gp_loss = self._gradient_penalty_loss_fn(
                    real_target_input_stack,
                    fake_target_input_stack,
                )
            else:
                gp_loss = torch.tensor(0.0, device=self.device)
                
            all_loss_dict[self.gradient_penalty_loss_name] = \
                gp_loss.item()

            generator_loss = self._generator_loss_fn(
                discriminator_fake_as_real_prob, targets, pred
            )

            all_loss_dict[self.generator_loss_name] = \
                generator_loss.item()

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
    
    @property
    def generator_loss_name(self):
        return type(self._generator_loss_fn).__name__
    
    @property
    def discriminator_loss_name(self):
        return type(self._discriminator_loss_fn).__name__
    
    @property
    def gradient_penalty_loss_name(self):
        return type(self._gradient_penalty_loss_fn).__name__ \
            if self._gradient_penalty_loss_fn is not None else \
                "NoGradientPenaltyLoss"