"""Backwards-compatibility shim for MMD variable selection.
"""
from ..variable_selection.mmd.config import MMDSelectionConfig
from ..variable_selection.mmd.selector import MMDVariableSelector

__all__ = [
    "MMDSelectionConfig",
    "MMDVariableSelector",
]
