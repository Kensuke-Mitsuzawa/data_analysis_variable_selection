"""1D-Wasserstein variable selection module.
"""
from .config import WassersteinSelectionConfig
from .selector import WassersteinVariableSelector

__all__ = [
    "WassersteinSelectionConfig",
    "WassersteinVariableSelector",
]
