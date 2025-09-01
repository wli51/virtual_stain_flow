from math import cos, pi
from abc import ABC, abstractmethod
from typing import (
    Callable, Protocol, Union, Optional
)

from ..callbacks.callbacks import Callback, CallbackBase

class WeightSchedule(Callback, Protocol):
    """
    Protocol for Dynamic weight that can be called to get the current weight, 
    and optionally ticked by trainer. Inherits the Callback protocol for the
    hooks. 
    """
    def __call__(self) -> float: ...
    def on_epoch_end(self, **kwargs) -> None: ...  
    def on_batch_end(self, **kwargs) -> None: ...

WeightLike = Union[float, WeightSchedule, Callable[[], float]]  

class AbstractWeightSchedule(CallbackBase, ABC):
    """
    Nominal base for WeightSchedule. 
    - Guarantees no-op hooks inerhited from CallbackBase
    - enforces __call__ in subclasses
    - subclasses type checkable as WeightSchedule Protocol
    """
    @abstractmethod
    def __call__(self) -> float: ...

    def clone(self, with_state: bool = False) -> "AbstractWeightSchedule":
        # with_state not used in abstract base, but could be in subclasses
        import copy
        return copy.deepcopy(self)

class FixedWeight(AbstractWeightSchedule):
    """Wrap a constant float as a schedule to unify handling."""
    def __init__(self, value: float):
        self.value = float(value)
    def __call__(self) -> float:
        return self.value

class WarmupCosine(AbstractWeightSchedule):
    """
    Linear warm-up (start -> base) for `warmup_epochs`, then cosine decay 
    (base -> target) over Epochs.
    Defaults to a warm-up with weight 1.0 for 10 epochs, then cosine decay to 0.0
    over 100 epochs.
    """
    def __init__(
        self,
        base: float = 1.0,
        target: float = 0.0,
        warmup_epochs: int = 10,
        anneal_epochs: int = 100,
        start: Optional[float] = None,
    ):
        """
        Initialize the WarmupCosine weight schedule.

        :param base: The base weight value after warm-up.
        :param target: The target weight value after decay.
        :param warmup_epochs: Number of epochs for linear warm-up.
        :param decay_epochs: Number of epochs for cosine decay.
        :param start: The starting weight value for warm-up.
        """
        self.base = float(base)
        self.target = float(target)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.anneal_epochs = max(1, int(anneal_epochs))
        self.start = float(start) if start is not None else self.base
        self._epoch: int = 0

    def __call__(self) -> float:
        t = self._epoch
        we = self.warmup_epochs
        ae = self.anneal_epochs

        # Linear warm-up: start -> base
        if we > 0 and t <= we:
            return self.start + (self.base - self.start) * (t / we)

        # Cosine interpolate: base -> target
        td = max(0, t - we)
        if td <= ae:
            return self.target + (self.base - self.target) * (1 + cos(pi * td / ae)) * 0.5

        # After schedule: clamp at target
        return self.target

    def on_epoch_end(self, epoch: Optional[int]=None, **kwargs) -> None:
        """tick function"""
        self._epoch += 1

    def clone(self, with_state: bool = False) -> "WarmupCosine":
        import copy
        new = copy.deepcopy(self)
        if not with_state:
            new._epoch = 0
        return new