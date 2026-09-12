"""certified_pruner — early-kill training runs with a bound on killing runs that would have ended well."""

from .core import KillRule, NotEnoughCalibration, last_value

__version__ = "0.1.0"
__all__ = ["KillRule", "NotEnoughCalibration", "last_value", "CertifiedPruner"]


from typing import Any


def __getattr__(name: str) -> Any:
    # Optuna is optional: import the pruner only when someone asks for it.
    if name == "CertifiedPruner":
        from .optuna_pruner import CertifiedPruner
        return CertifiedPruner
    raise AttributeError(name)
