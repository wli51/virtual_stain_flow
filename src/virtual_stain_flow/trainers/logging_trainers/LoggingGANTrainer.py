import pathlib
from collections import defaultdict
from typing import Optional, List, Union, Sequence, Dict

import torch

from .AbstractLoggingTrainer import AbstractLoggingTrainer 
from ...losses.loss_group import LossItem, LossGroup


path_type = Union[pathlib.Path, str]

class LoggingGANTrainer(AbstractLoggingTrainer):
    """
    """
    def __init__(
        self,
        generator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        generator_loss_group: LossGroup,
        discriminator: torch.nn.Module,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_loss_group: LossGroup,
        generator_update_freq: int = 5,
        discriminator_update_freq: int = 1,        
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self._generator = generator
        self._generator_optimizer = generator_optimizer
        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer

        if not isinstance(generator_loss_group, LossGroup):
            raise TypeError(
                "generator_loss_group must be an instance of LossGroup"
            )
        self._generator_loss_group = generator_loss_group

        if not isinstance(discriminator_loss_group, LossGroup):
            raise TypeError(
                "discriminator_loss_group must be an instance of LossGroup"
            )
        self._discriminator_loss_group = discriminator_loss_group

        if not (
            generator_update_freq == 1 or discriminator_update_freq == 1
        ):
            raise ValueError(
                "Either generator_update_freq or discriminator_update_freq "
                "must be 1."
            )
        self._discriminator_update_freq = discriminator_update_freq
        self._generator_update_freq = generator_update_freq

        self._global_step = 0
        # Due to the asynchronous nature of GAN generator updates, 
        # we keep track of last values of the generator and discriminator losses
        # so that we log the last updated loss for generator and discriminator
        # during epochs they are not updated. Alternative strategy would
        # be to modify the logging system to only log loss values during
        # updates but that would introduce additional complexity. 
        zero = 0.0
        self._last_discriminator_losses: Dict[str, float] = {
            k: zero for k in self._discriminator_loss_group.item_names()}
        self._last_generator_losses: Dict[str, float] = {
            k: zero for k in self._generator_loss_group.item_names()}

    def _train_discriminator_step(
        self,
        inputs: torch.tensor, 
        targets: torch.tensor
    ) -> Dict[str, float]:
        """
        """

        if self._global_step % self._discriminator_update_freq != 0:
            return self._last_discriminator_losses
        
        self._discriminator.train()
        self._discriminator_optimizer.zero_grad()

        with torch.no_grad():
            preds = self._generator(inputs)

        ctx = {}

        for tensor, type in zip(
            [targets, preds],
            ['real', 'fake']
        ):
            ctx[f'{type}_input_stack'] = torch.cat([tensor, inputs], dim=1)
            ctx[f'p_{type}_as_real'] = self._discriminator(
                ctx[f'{type}_input_stack'])
            
        ctx['discriminator'] = self._discriminator

        total_disc_loss, logs = self._discriminator_loss_group(
            train=True,
            inputs=ctx
        )
        total_disc_loss.backward()
        self._discriminator_optimizer.step()
        
        # can't use warlus here becaus we are stuck with python 3.9
        self._last_discriminator_losses = logs
        return logs
    
    def _train_generator_step(
        self,
        inputs: torch.tensor,
        targets: torch.tensor
    ) -> Dict[str, float]:
        """
        """

        if self._global_step % self._generator_update_freq != 0:
            return self._last_generator_losses
        
        self._generator.train()
        self._generator_optimizer.zero_grad()

        preds = self._generator(inputs)
        
        ctx = {
            'targets': targets,
            'preds': preds,
        }

        with torch.no_grad():
            ctx['p_fake_as_real'] = self._discriminator(
                torch.cat([preds, inputs], dim=1)
            )

        total_gen_loss, logs = self._generator_loss_group(
            train=True,
            inputs=ctx
        )
        total_gen_loss.backward()
        self._generator_optimizer.step()

        # can't use warlus here becaus we are stuck with python 3.9
        self._last_generator_losses = logs
        return logs
    

