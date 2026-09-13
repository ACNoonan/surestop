"""surestop — stop bad runs early, with a bound on how often you stop one that would have ended well."""

from .core import KillRule, NotEnoughCalibration, last_value

__version__ = "0.2.0"
__all__ = ["KillRule", "NotEnoughCalibration", "last_value", "ConformalPruner"]


from typing import Any


def __getattr__(name: str) -> Any:
    # Optuna is optional: import the pruner only when someone asks for it.
    if name == "ConformalPruner":
        from .optuna_pruner import ConformalPruner
        return ConformalPruner
    raise AttributeError(name)
