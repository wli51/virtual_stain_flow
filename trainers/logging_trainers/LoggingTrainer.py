import pathlib
from collections import defaultdict
from typing import Optional, List, Union, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split

from .AbstractLoggingTrainer import AbstractLoggingTrainer
from ...losses.loss_item_group import LossItem, LossGroup

path_type = Union[pathlib.Path, str]

class LoggingTrainer(AbstractLoggingTrainer):
    """
    Trainer class for single network CNN image-image models 
    Prototyping for the new mlflow logger
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        backprop_loss: Union[torch.nn.Module, List[torch.nn.Module]],
        backprop_loss_weights: Optional[List[float]] = None,
        **kwargs                    
    ):
        """
        Initialize the trainer with the model, optimizer and loss function.

        :param model: The model to be trained.
        :param optimizer: The optimizer to be used for training.
        :param backprop_loss: The loss function to be used for training or a list of loss functions.
        :param kwargs: Additional arguments passed to the parent AbstractTrainer class.
        """

        super().__init__(**kwargs)

        self._model = model
        self._optimizer = optimizer
        backprop_loss = backprop_loss if isinstance(backprop_loss, list) else \
            [backprop_loss]
        if backprop_loss_weights is None:
            # If no weights are provided, use equal weights for all losses
            backprop_loss_weights = [1.0] * len(backprop_loss)
        elif isinstance(backprop_loss_weights, (int, float)):
            # If a single weight is provided, use it for all losses
            backprop_loss_weights = [backprop_loss_weights] * len(backprop_loss)
        elif isinstance(backprop_loss_weights, list):
            if len(backprop_loss_weights) == len(backprop_loss):
                pass
            else:
                raise ValueError(
                    "Length of backprop_loss_weights must match the number of loss functions."
                )
        else:
            raise TypeError(
                "backprop_loss_weights must be a float, int, or list of floats."
            )
        
        self._gen_loss_group = LossGroup(
            "generator_losses",
            [
                LossItem(
                    module=loss_fn,
                    args=("target", "pred"),
                    weight=weight
                ) for loss_fn, weight in zip(
                    backprop_loss,
                    backprop_loss_weights
                )
            ]
        )
        
    """
    Overidden methods from the parent abstract class
    """
    def train_step(
            self, 
            input: torch.tensor, 
            target: torch.tensor
        ):
        """
        Perform a single training step on batch.

        :param input: The input image data batch
        :param target: The target image data batch
        """

        # move the data to the device
        input, target = input.to(self.device), target.to(self.device)

        # set the model to train
        self.model.train()
        # set the optimizer gradients to zero        
        self.optimizer.zero_grad()

        # Forward pass
        pred = self.model(input)

        ctx = {
            'target': target,
            'pred': pred
        }

        total_gen_loss, logs = self._gen_loss_group(ctx, phase='train')

        total_gen_loss.backward()
        self.optimizer.step()

        # Calculate the metrics outputs and update the metrics
        for _, metric in self.metrics.items():
            metric.update(pred, target, validation=False)
        
        return logs
    
    def evaluate_step(
            self, 
            input: torch.tensor, 
            target: torch.tensor
        ):
        """
        Perform a single evaluation step on batch.

        :param input: The input image data batch
        :param target: The target image data batch
        """

        # move the data to the device
        input, target = input.to(self.device), target.to(self.device)

        # set the model to evaluation
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            pred = self.model(input)

            ctx = {
                'target': target,
                'pred': pred
            }
            _, logs = self._gen_loss_group(ctx, phase='eval')

            # Calculate the metrics pred and update the metrics
            for _, metric in self.metrics.items():
                metric.update(pred, target, validation=True)
        
        return logs
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        losses = defaultdict(list)
        # Iterate over the train_loader
        for inputs, targets in self._train_loader:
            batch_loss = self.train_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

        # reduce loss
        return {
            key: sum(value) / len(value) for key, value in losses.items()
        }
    
    def evaluate_epoch(self):
        """
        Evaluate the model for one epoch.
        """

        self.model.eval()
        losses = defaultdict(list)
        # Iterate over the val_loader
        for inputs, targets in self._val_loader:
            batch_loss = self.evaluate_step(inputs, targets)
            for key, value in batch_loss.items():
                losses[key].append(value)

        # reduce loss
        return {
            key: sum(value) / len(value) for key, value in losses.items()
        }