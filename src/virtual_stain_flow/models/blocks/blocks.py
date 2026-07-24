"""
blocks.py
 
A block is a feature extraction/learning group, bundling learnable nn layers, 
    normalizations, and activation functions, and abstracts away the specific
    internal arrangement details (order, type, number of repetition of same sequences).
The behavior of a block on an input feature map tensor of (B, C, H, W) can be
    one of the two: 
    1) returns a (B, C'', H', W') output, where the spatial dimensions (H', W') 
        changes in a way determined by the block's implementation and 
        number of channels may or may not change.
        (Blocks that behave this way are defined in the `up_down_blocks.py` module)
    2) returns a (B, C'', H, W) output, preserving the spatial dimensions
        while the number of channels (C'') may or may not change.
        (Blocks that behave this way are defined in this module)
"""


from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from .utils import (
    NormType,
    ActivationType,
    validate_network,
)
from .units import (
    ConditionableUnit,
    ChannelAdaptUnit, 
    ConvNeXtUnit, 
    ConvUnit,
)


class AbstractBlock(ABC, torch.nn.Module):
    """
    Abstract base class for all neural network blocks.
    Provides a common interface and basic validation for input and output channels,
    as well as the number of units within the block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_units: int = 1,
        **kwargs: dict
    ):
        
        super().__init__()

        # Centralizing input type checking
        if not isinstance(in_channels, int):
            raise TypeError("Expected in_channels to be int, "
                            f"got {type(in_channels).__name__}")
        if in_channels <= 0:
            raise ValueError("Expected in_channels to be positive, "
                             f"got {in_channels}")
        if out_channels is None:
            out_channels = in_channels
        if not isinstance(out_channels, int):
            raise TypeError("Expected out_channels to be int, "
                            f"got {type(out_channels).__name__}")
        if out_channels <= 0:
            raise ValueError("Expected out_channels to be positive, "
                             f"got {out_channels}")
        if not isinstance(num_units, int):
            raise TypeError("Expected num_units to be int, "
                            f"got {type(num_units).__name__}")
        if num_units <= 0:
            raise ValueError("Expected num_units to be positive, "
                             f"got {num_units}")

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_units = num_units

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.
        """
        raise NotImplementedError
    
    # Centralizing property definitions for blocks
    @property
    def in_channels(self) -> int:
        return self._in_channels
    @property
    def out_channels(self) -> int:
        return self._out_channels
    @property
    def num_units(self) -> int:
        return self._num_units
    
    # These 2 below should be overriden to reflect the actual spatial dimension
    # changes the block applies. By default they indicate spatial preserving
    # blocks, i.e. the height and width of the input tensor remain unchanged.
    def out_h(self, in_h: int) -> int:
        return in_h
    
    def out_w(self, in_w: int) -> int:
        return in_w    


class ComputeBlock(AbstractBlock):
    """
    Reusable compute block, incorporating Unit abstraction to support conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        compute_units: list[ConditionableUnit],
        preprocess_unit: Optional[torch.nn.Module] = None,
    ):
        """
        Initializes the compute block with the specified parameters.

        :param in_channels: Number of input channels.
        :param compute_units: A list of compute units for this compute block.
        :param preprocess_unit: An optional preprocessing module applied before the compute units.
        """    
        
        preprocess_unit = preprocess_unit or torch.nn.Identity()
        n_channels = validate_network(preprocess_unit, expected_in_channels=in_channels)

        if compute_units is None:
            raise ValueError("compute_units must be provided and cannot be None.")
        else:
            for unit in compute_units:
                n_channels = validate_network(unit, expected_in_channels=n_channels)

        super().__init__(
            in_channels=in_channels,
            out_channels=n_channels,
            num_units = len(compute_units)
        )

        self.preprocess_unit = preprocess_unit
        self.units = torch.nn.ModuleList([
            unit
            for i, unit in enumerate(compute_units)
        ])

    def forward(
        self, 
        x: Tensor,
        **kwargs: dict,
    ) -> Tensor:
        x = self.preprocess_unit(x)
        for unit in self.units:
            x = unit(x, **kwargs)
        return x


class Conv2DConvNeXtBlock(ComputeBlock):
    """
    A ConvNeXt block that applies a sequence of ConvNeXt units 
    with inital 2D convolution to adjust the number of channels if needed.
    Mimics the design of timm.models.convnext.ConvNeXtStage but less sophisticated
    in implementation. 
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_units: int = 1,
        norm_type: NormType = 'layer',
        conv_kernel_size: int = 1,
        convnext_kernel_size: int = 7
    ):
        """
        Initializes a ConvNeXt block with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If
            None, defaults to in_channels.
        :param num_units: Number of ConvNeXt units in the block.
        :param norm_type: Type of normalization to apply.
            Default is 'layer' (GroupNorm with 1 group).
        :param conv_kernel_size: Kernel size for the initial 2d convolution
            that adjusts the number of channels if in_channels != out_channels.
            Default is 1 (identity convolution, fastest). 
            Though one may want to use larger kernel sizes (3) to allow for 
            some spatial feature extraction prior to the ConvNeXtBlocks. 
        :param convnext_kernel_size: Kernel size for the depth-wise convolution
            layers inside the ConvNeXtBlock. Default is 7, 
            as recommended by Liu et al. (2022), for large receptive fields.
        """
        
        out_channels = out_channels or in_channels

        if in_channels != out_channels:
            # insert a spatial dimension preserving 1x1 convolution
            # operation to adjust the number of channels because
            # ConvNeXtBlock expects same input and output channels
            # if the in/out channels are matched, this won't be added. 
            preprocess_unit = ChannelAdaptUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_kernel_size=conv_kernel_size,
                norm_type=norm_type,
                activation_type='none'
            )
        else:
            preprocess_unit = None

        units = [
            ConvNeXtUnit(
                in_channels=out_channels, 
                kernel_size=convnext_kernel_size,
                stride=1, # fixed
                ls_init_value=None,
            ) for _ in range(num_units)
        ]

        super().__init__(
            in_channels = in_channels,
            preprocess_unit = preprocess_unit,
            compute_units = units,
        )


class Conv2DNormActBlock(ComputeBlock):
    """
    A Conv2D block that applies a sequence of Conv2D -> Norm -> Activation
        layers, commonly used in UNet architectures.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_units: int = 1,
        norm_type: NormType = 'batch',
        act_type: ActivationType = 'relu'
    ):
        """
        Initializes a Conv2D block with the specified parameters.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels. If None, defaults to
            in_channels.
        :param num_units: Number of Conv2D>Norm>Activation units in the block.
        :param norm_type: Type of normalization to apply. Default is 'batch'
            (GroupNorm with 1 group).
        :param act_type: Type of activation function to apply. Default is 'relu'.
        """

        out_channels = out_channels or in_channels
        mid_channels = [out_channels] * (num_units - 1)

        compute_units = torch.nn.ModuleList([
            ConvUnit(
                in_channels=_in,
                out_channels=_out,
                norm_type=norm_type,
                activation_type=act_type,
            )
            for _in, _out in
            zip([in_channels] + mid_channels, [out_channels] + mid_channels)
        ])

        super().__init__(
            in_channels = in_channels,
            compute_units = compute_units,
            preprocess_unit = None
        )
