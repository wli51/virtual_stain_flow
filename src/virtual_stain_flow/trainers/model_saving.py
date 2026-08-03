"""
Functional helpers for model snapshots and weight persistence.
"""

import copy
import pathlib
from typing import Optional

import torch


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    try:
        snapshot = copy.deepcopy(model)
    except Exception as error:
        raise RuntimeError(
            f"Unable to create an independent snapshot of {type(model).__name__}."
        ) from error

    if not isinstance(snapshot, torch.nn.Module) or snapshot is model:
        raise RuntimeError(
            f"Unable to create an independent snapshot of {type(model).__name__}."
        )
    return snapshot


def select_model_for_saving(
    model: torch.nn.Module,
    best_model: Optional[torch.nn.Module],
    use_best_model: bool,
) -> torch.nn.Module:
    if use_best_model and best_model is not None:
        return best_model
    return model


def build_weight_filename(
    prefix: str,
    suffix: str,
    file_ext: str,
) -> str:
    return f"{prefix}_{suffix}{file_ext}"


def save_model_weights(
    model: torch.nn.Module,
    save_path: pathlib.Path,
    prefix: str,
    suffix: str,
    file_ext: str,
) -> pathlib.Path:
    filename = build_weight_filename(prefix, suffix, file_ext)
    return model.save_weights(filename=filename, dir=save_path)
