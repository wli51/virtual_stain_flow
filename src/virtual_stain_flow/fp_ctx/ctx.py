"""
ctx.py

Context class for bundling tensors in a forward pass so they can be 
    passed to LossGroup orderly. 
"""

from typing import Dict, Iterable, Tuple

import torch


class Context:
    """
    A simple context class to bundle tensors in a forward pass.
    Behaves like a dictionary plus a `require` method to interface with
        ForwardGroup and LossGroup.
    """

    __slots__ = {"_store", }

    def __init__(self, **kwargs: torch.Tensor):

        self._store: Dict[str, torch.Tensor] = {}
        for k, v in kwargs.items():
            self._store[k] = v

    def add(self, **tensors: torch.Tensor):
        """
        Adds new tensors to the context.

        :param tensors: Keyword arguments,
            where keys are the names of the tensors.
        """
        self._store.update(tensors)
        return self
    
    def require(self, keys: Iterable[str]) -> None:
        """
        Checks that all required keys are present in the context.
            Raises a ValueError if any key is missing.

        :param keys: An iterable of keys that are required to be present.
        """
        missing = [k for k in keys if k not in self._store]
        if missing:
            raise ValueError(
                f"Missing required inputs {missing} for forward group."
            )
        return None
    
    def __getitem__(self, key: str) -> torch.Tensor:
        return self._store[key]
    
    def as_kwargs(self) -> Dict[str, torch.Tensor]:
        """
        Returns the context as a dictionary of keyword arguments.
        Intended use: loss_group(train, **ctx.as_kwargs())
        """
        return self._store
    
    def as_metric_args(self) -> Tuple[torch.Tensor]:
        """
        Returns the predictions and targets tensors for 
            Image quality assessment metric computation.
        Intended use: metric.update(*ctx.as_metric_args())
        """
        return (self._store['preds'], self._store['targets'])
    
    def __repr__(self) -> str:
        if not self._store:
            return "Context()"
        
        items = []
        for key, tensor in self._store.items():
            if isinstance(tensor, torch.Tensor):
                shape_str = "(" + ",".join(map(str, tensor.shape)) + ")"
                items.append(f"  {key}: {shape_str} "
                             f"({tensor.device}) ({tensor.dtype})")
            else:
                items = [f"  {key}: {tensor.__class__.__name__}"] + items
        
        items_str = "\n".join(items)
        return f"Context(\n{items_str}\n)"
