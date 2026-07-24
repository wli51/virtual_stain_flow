"""
Abstraction one level below blocks. 
Allows finer grained operations such as conditioning that happens to
    only parts of a compute or scaling blocks. 
"""

from typing import Optional
from abc import ABC, abstractmethod

import timm
import torch
from torch import Tensor

from .utils import (
    get_norm,
    NormType,
    get_activation,
    ActivationType,
)
from .block_conditioner import Conditioner, IdentityConditioner


class ConditionableUnit(ABC, torch.nn.Module):
    """
    Base class for units supporting conditioning, 
        defaults to no-op condition.  
    """
    def __init__(self, conditioner: Optional[Conditioner] = None) -> None:
        super().__init__()
        self.conditioner = conditioner if conditioner is not None else IdentityConditioner()

    def condition(self, x: Tensor, **kwargs: dict) -> Tensor:
        return self.conditioner(x, **kwargs)

    @abstractmethod
    def forward(self, x: Tensor, **kwargs: dict) -> Tensor:
        pass

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    def out_h(self, in_h: int) -> int:
            return in_h
        
    def out_w(self, in_w: int) -> int:
        return in_w


class ChannelAdaptUnit(ConditionableUnit):
    """
    Channel adaptation unit that adjusts the number of channels in the input tensor
        to match the desired output channels, with optional normalization and activation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: NormType = 'layer',
        activation_type: ActivationType = 'none',
    ) -> None:
        super().__init__(conditioner=None)
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.adapt = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same'
        )
        self.norm = get_norm(out_channels, norm_type)
        self.activation = get_activation(activation_type)

    def forward(self, x: Tensor, **kwargs: dict) -> Tensor:
        if self._in_channels != self._out_channels:
            x = self.adapt(x)
        x = self.condition(x, **kwargs)
        x = self.norm(x)
        x = self.activation(x)
        return x

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class ConvUnit(ConditionableUnit):
    """
    Standard convolutional2d unit with optional normalization and activation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: NormType = 'batch',
        activation_type: ActivationType = 'relu'
    ) -> None:
        super().__init__(conditioner=None)
        self._in_channels = in_channels
        self._out_channels = out_channels if out_channels is not None else in_channels
        self.conv = torch.nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = get_norm(self._out_channels, norm_type)
        self.activation = get_activation(activation_type)

    def forward(self, x: Tensor, **kwargs: dict) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.condition(x, **kwargs)
        x = self.activation(x)
        return x

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class ConvNeXtUnit(ConditionableUnit):
    """
    ConvNeXt unit wrapping timm.models.convnext.ConvNeXtBlock, 
        which bundles normalization, activation and residual connections. 
    For the sake of faithful reproducition of convnext v2 behavior
        hard-coded to use the Conv1x1 in place of MLP, gelu for activation,
        and layer normalization.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        ls_init_value: Optional[float] = None
    ):
        """
        Initializes the ConvNeXt compute unit with the specified parameters.

        :param in_channels: Number of input channels. 
        :param kernel_size: Size of the convolutional kernel.
        :param stride: Stride for the convolution.
        :param ls_init_value: Initial value for the layer scale parameter.
        :param unit_index: Optional index of the compute unit.
        """
        super().__init__(conditioner=None)

        self.module = timm.models.convnext.ConvNeXtBlock(
            in_chs=in_channels,
            out_chs=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            ls_init_value=ls_init_value,
            # this is a switch between 2 equivalent implementations
            # here by setting conv_mlp=True 1x1 convolutional layers are used
            # to simulate MLP, as is in original convnext implementation
            conv_mlp=True,
            use_grn=False,
            act_layer='gelu', # same as the original implementation
            norm_layer=timm.layers.LayerNorm2d # same as original implementation
        )
        self._in_channels = in_channels
        self._out_channels = in_channels

    def forward(
        self, 
        x: Tensor,
        **kwargs: dict,
    ) -> Tensor:
        """
        Hijacks the original timm.models.convnext.ConvNeXtBlock forward pass
            and incorporate conditioning before the residual connection.

        :param x: The input tensor to the compute module.
        :param kwargs: Conditioning arguments, relayed to conditioner.
        :return: The output tensor after applying the module and conditioning.
        """
        # original forward pass from timm.models.convnext.ConvNeXtBlock
        shortcut = x
        x = self.module.conv_dw(x)
        x = self.module.norm(x)
        x = self.module.mlp(x)
        if self.module.gamma is not None:
            x = x.mul(self.module.gamma.reshape(1, -1, 1, 1))
        x = self.module.drop_path(x)
        # implementation differs from the original here to support conditioning
        # before residual connection
        x = self.condition(x, **kwargs)
        x = x + self.module.shortcut(shortcut)
        return x

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels
