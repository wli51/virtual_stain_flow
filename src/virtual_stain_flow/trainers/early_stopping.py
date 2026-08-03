"""
Functional helpers for selecting metrics and updating early-stopping state.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence


@dataclass(frozen=True)
class EarlyStopUpdate:
    improved: bool
    best_value: Any
    counter: int
    should_stop: bool


def collect_early_stop_metric(
    metric_name: Optional[str],
    val_losses: Mapping[str, Sequence[Any]],
    val_metrics: Mapping[str, Sequence[Any]],
) -> Optional[Any]:
    if metric_name is None:
        return None
    if metric_name in val_losses:
        return val_losses[metric_name][-1]
    if metric_name in val_metrics:
        return val_metrics[metric_name][-1]
    raise ValueError("Invalid early termination metric")


def collect_default_validation_loss(
    val_losses: Mapping[str, Sequence[Any]],
) -> Optional[Any]:
    if not val_losses:
        return None

    first_loss_values = next(iter(val_losses.values()))
    if not first_loss_values:
        return None
    return first_loss_values[-1]


def update_early_stop_state(
    current_value: Any,
    best_value: Any,
    counter: int,
    patience: int,
    mode: Literal["min", "max"],
) -> EarlyStopUpdate:
    if mode not in ("min", "max"):
        raise ValueError(f"Unsupported early termination mode: {mode}")

    improved = (
        current_value < best_value
        if mode == "min"
        else current_value > best_value
    )

    if improved:
        return EarlyStopUpdate(
            improved=True,
            best_value=current_value,
            counter=0,
            should_stop=False,
        )

    updated_counter = counter + 1
    return EarlyStopUpdate(
        improved=False,
        best_value=best_value,
        counter=updated_counter,
        should_stop=updated_counter >= patience,
    )
