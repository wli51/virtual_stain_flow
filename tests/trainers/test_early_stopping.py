from collections import OrderedDict

import pytest

from virtual_stain_flow.trainers.early_stopping import (
    collect_default_validation_loss,
    collect_early_stop_metric,
    update_early_stop_state,
)


def test_collect_explicit_metric_prefers_validation_loss():
    value = collect_early_stop_metric(
        "score",
        {"score": [0.4]},
        {"score": [0.8]},
    )

    assert value == 0.4


def test_collect_explicit_metric_rejects_unknown_name():
    with pytest.raises(ValueError, match="Invalid early termination metric"):
        collect_early_stop_metric("missing", {}, {})


def test_default_validation_loss_uses_first_inserted_key():
    val_losses = OrderedDict([
        ("reconstruction", [0.7, 0.5]),
        ("adversarial", [0.2, 0.1]),
    ])

    assert collect_default_validation_loss(val_losses) == 0.5


def test_default_validation_loss_is_none_when_empty():
    assert collect_default_validation_loss({}) is None


@pytest.mark.parametrize(
    ("mode", "current", "best"),
    [("min", 0.2, float("inf")), ("max", 0.8, float("-inf"))],
)
def test_update_improves_from_mode_sentinel(mode, current, best):
    update = update_early_stop_state(current, best, 2, 3, mode)

    assert update.improved is True
    assert update.best_value == current
    assert update.counter == 0
    assert update.should_stop is False


def test_update_stops_at_patience_boundary():
    update = update_early_stop_state(0.5, 0.3, 2, 3, "min")

    assert update.improved is False
    assert update.best_value == 0.3
    assert update.counter == 3
    assert update.should_stop is True


def test_update_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported early termination mode"):
        update_early_stop_state(0.5, 0.3, 0, 1, "median")
