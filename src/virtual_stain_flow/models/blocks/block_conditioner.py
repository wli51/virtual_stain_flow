"""
block_conditioner.py

Conditioner class for compute units, providing a no-op implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Conditioner(ABC, torch.nn.Module):
    """
    Abstract base class for conditioners used in compute units.
    """
    def __init__(self, unit: Optional[torch.nn.Module] = None):
        """
        Bind the conditioner to a specific compute unit.
        
        :param unit: The compute unit to which this conditioner is bound.
        """
        super().__init__()
        self.unit = unit

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        **kwargs: dict
    ) -> Tensor:
        raise NotImplementedError(
            "Conditioner subclasses must implement the forward method."
        )


class IdentityConditioner(Conditioner):
    """
    No-op conditioner for compute units.
    """

    def forward(
        self,
        x: Tensor,
        **kwargs: dict
    ) -> Tensor:
        """
        No-op conditioner, simply returns the input tensor unchanged.
        Disregards all keyword conditioning arguments.
        """
        return x
