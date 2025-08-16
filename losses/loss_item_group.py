from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn

from .AbstractLoss import AbstractLoss

def _get_loss_name(
        loss_fn: Union[nn.Module, AbstractLoss]
):
    """
    Helper method to get the name of the loss function.
    """
    if isinstance(loss_fn, AbstractLoss) and hasattr(loss_fn, "metric_name"):
        return loss_fn.metric_name
    elif isinstance(loss_fn, torch.nn.Module):
        return type(loss_fn).__name__
    else:
        raise TypeError(
            "Expected loss_fn to be either a torch.nn.Module or an AbstractLoss instance."
        )    

@dataclass
class LossItem:
    """
    Wrapper around torch.nn.Module loss function to additionally contain
    a loss key (name) and a weight.
    """
    module: nn.Module
    args: Tuple[str, ...]
    key: Optional[str] = None
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.key is None:
            self.key = _get_loss_name(self.module)

        if isinstance(self.args, str):
            self.args = (self.args,)
        elif not isinstance(self.args, tuple):
            raise TypeError(
                f"Expected args to be a str or tuple, got {type(self.args).__name__}"
            )
    
    def compute(
            self, 
            ctx: Mapping[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            zero = next(iter(ctx.values())).new_tensor(0.0)  # device-safe zero
            return zero, zero
        
        inputs = [ctx[name] for name in self.args]

        raw = self.module(*inputs)  # raw loss
        if raw.dim() > 0:
            raw = raw.mean()     # standardize to scalar if needed
        return raw, self.weight * raw
    
class LossGroup:
    """
    A collection of loss items that can/should be computed together, 
    surrounding the update of the same network for convenience during training.
    """
    def __init__(
        self,
        name: str,
        items: Sequence[LossItem]
    ):
        self._name = name
        self._items: List[LossItem] = list(items)

    @property
    def keys(self) -> List[str]:
        return [item.key for item in self._items]
    
    def __call__(
        self, 
        ctx: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:        
        total = None
        logs: Dict[str, float] = {}
        
        for it in self._items:
            raw, wraw = it.compute(ctx)
            logs[it.key] = float(raw.detach().item())
            total = wraw if total is None else (total + wraw)
            
        return total, logs
    
    def set_weights(
        self, 
        weights: Mapping[str, float]
    ):
        """
        Set the weights of the loss items in the group.
        
        :param weights: A mapping of loss item keys to their weights.
        """
        for item in self._items:
            if item.key in weights:
                item.weight = float(weights[item.key])

    def enable(
        self, 
        keys: Iterable[str]
    ):
        """
        Enable loss items in the group by their keys.
        
        :param keys: An iterable of loss item keys to enable.
        """
        keep = set(keys)
        for it in self._items:
            it.enabled = it.key in keep