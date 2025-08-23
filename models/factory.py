# factory.py
import importlib
import inspect
import pathlib
from typing import Dict, Type, TypeVar, Optional, Callable, Union, Any

import torch

from .base_model import BaseGeneratorModel

T = TypeVar("T")

def _qualname(obj: Any) -> str:
    """Fully qualified import path for a class or function."""
    return f"{obj.__module__}.{obj.__name__}"

def _locate(path: str) -> Any:
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)

def from_config(
    config: Dict[str, Any],
    *,
    base_type: Optional[Type[T]] = BaseGeneratorModel,
    hook: Optional[Callable[[T, Dict[str, Any]], None]] = None,  # e.g. load weights
    strict_class_check: bool = True,
) -> T:
    """
    Centralized creator for any model with a `from_config` classmethod.
    Expects config['class_path'] and config['init'] OR a single init dict.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    # Accept raw init-only dicts by allowing a fallback
    class_path = config.get("class_path")
    init_cfg = config.get("init", config)

    if not class_path:
        raise ValueError("Config missing 'class_path'. "
                         "Use model.to_config() to generate a proper config.")

    cls = _locate(class_path)

    if strict_class_check and base_type is not None:
        if not inspect.isclass(cls) or not issubclass(cls, base_type):
            raise TypeError(f"{class_path} is not a subclass of {base_type.__name__}")

    if not hasattr(cls, "from_config"):
        raise AttributeError(f"{class_path} has no classmethod 'from_config'")

    obj = cls.from_config({"class_path": class_path, "init": init_cfg, **config})

    # optional post-create hook (e.g., load checkpoints or frozen weights)
    if hook is not None:
        hook(obj, config)

    return obj

def load(
    config: Dict[str, Any],
    weights: Union[pathlib.Path, str],
    **kwargs: Any
):
    model_obj = from_config(config, **kwargs)
    if isinstance(weights, (str, pathlib.Path)):
        weights_path = pathlib.Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weight file {weights_path} does not exist.")
    else:
        raise TypeError(f"Expected weights to be a string or pathlib.Path, got {type(weights)}")
    
    try:
        model_obj.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True),
        )
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load weights from {weights_path}: {e}") from e

    return model_obj   