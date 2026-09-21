"""Backwards-compatibility shim for Wasserstein variable selection.
"""
from ..variable_selection.wasserstein.config import WassersteinSelectionConfig
from ..variable_selection.wasserstein.selector import WassersteinVariableSelector

__all__ = [
    "WassersteinSelectionConfig",
    "WassersteinVariableSelector",
]
