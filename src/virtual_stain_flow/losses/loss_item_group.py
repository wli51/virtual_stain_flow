from dataclasses import dataclass
from typing import (
    Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional, Literal
)

import torch
import torch.nn as nn

from .AbstractLoss import AbstractLoss
from .dynamic_weight import WeightLike, WeightSchedule, FixedWeight

Phase = Literal['train', 'eval']

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
    
def _device_tensor_from_ctx(
        ctx: Mapping[str, torch.Tensor], val=0.0
    ) -> torch.Tensor:
    """
    Safely create a tensor on the same device and dtype as tensors in ctx.
    """
    try:
        t = next(iter(ctx.values()))
    except StopIteration:
        raise ValueError("ctx is empty; cannot infer device/dtype for zero tensor.")
    return t.new_tensor(val)

def _to_schedule(w: WeightLike) -> WeightSchedule:
    """
    Convert a weight-like object to a WeightSchedule.
    Handles the fixed float to callable conversion mostly and checks for
    valid callable types.
    """
    if isinstance(w, (int, float)):
        return FixedWeight(float(w))
    if callable(w):
        # Any callable() returning float counts; if it lacks tick handlers, that’s fine.
        if not hasattr(w, "__call__"):
            raise TypeError("Weight callable must be invocable with zero args.")
        return w  # type: ignore[return-value]
    raise TypeError(f"Unsupported weight type: {type(w).__name__}")

@dataclass
class LossItem:
    """
    Wrapper around torch.nn.Module loss function to additionally contain
    a loss key (name) and a weight.
    """
    module: nn.Module
    args: Tuple[str, ...]
    key: Optional[str] = None
    weight: WeightLike = 1.0
    enabled: bool = True
    # default to compute in both phases
    compute_at: Tuple[Phase, ...] = ('train', 'eval')
    reduction: Literal['mean', 'sum', 'none'] = 'mean'

    def __post_init__(self):
        if self.key is None:
            self.key = _get_loss_name(self.module)
        if isinstance(self.args, str):
            self.args = (self.args,)
        elif not isinstance(self.args, tuple):
            raise TypeError(
                f"Expected args to be a str or tuple, got {type(self.args).__name__}"
            )
        
        self._weight_schedule: WeightSchedule = _to_schedule(self.weight)

    def _current_weight(self) -> float:
        w = self._weight_schedule()
        return float(w)
    
    def compute(
            self, 
            ctx: Mapping[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            zero = _device_tensor_from_ctx(ctx, 0.0)
            return zero, zero
        
        try:
            inputs = [ctx[name] for name in self.args]
        except KeyError as e:
            missing = [k for k in self.args if k not in ctx]
            raise KeyError(
                f"Missing ctx keys for loss '{self.key}': {missing}") from e

        raw = self.module(*inputs)  # raw loss

        if raw.dim() > 0 and self.reduction != 'none':
            raw = raw.mean() if self.reduction == 'mean' else raw.sum()
        return raw, self._current_weight() * raw
    
    def active(self, phase: Phase) -> bool:
        """
        Check if the loss item is active for the given phase.
        
        :param phase: The phase to check ('train' or 'eval').
        :return: True if the loss item is active for the phase, False otherwise.
        """
        return self.enabled and (phase in self.compute_at)
    
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
        ctx: Mapping[str, torch.Tensor],
        phase: Phase = 'train'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:        
        total = None
        logs: Dict[str, float] = {}
        
        for it in self._items:            
            if not it.active(phase):
                logs[it.key] = 0.0 # defaults to zero
                continue
            raw, wraw = it.compute(ctx)
            logs[it.key] = float(raw.detach().item()) if raw.dim() == 0 else \
                float(raw.mean().detach().item())
            total = wraw if total is None else (total + wraw)
        
        # Incase at some point all loss items are disabled,
        # we still want to return a zero tensor for total
        # as opposed to erroring out
        if total is None:
            total = _device_tensor_from_ctx(ctx, 0.0)
            
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

    def collect_weight_callbacks(self) -> List[WeightSchedule]:
        """
        Centralized registration method for the trainer to associate all
        dynamic weight schedules with the trainer's tick handlers.
        """
        out: List[WeightSchedule] = []
        for it in self._items:
            sched = it._weight_schedule
            # Only return if it *has* at least one tick method
            if hasattr(sched, "on_epoch_end") or hasattr(sched, "on_batch_end"):
                out.append(sched)
        return out