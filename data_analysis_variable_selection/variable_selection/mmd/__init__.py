"""MMD-based variable selection module.
"""
from .config import MMDSelectionConfig
from .selector import MMDVariableSelector

__all__ = [
    "MMDSelectionConfig",
    "MMDVariableSelector",
]
