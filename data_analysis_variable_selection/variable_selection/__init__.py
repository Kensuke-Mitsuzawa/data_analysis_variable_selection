"""Two-sample variable selection module grouping MMD, 1D-Wasserstein, and base selector abstractions.
"""
from ..common.base_selector import BaseVariableSelector
from .mmd.config import MMDSelectionConfig
from .mmd.selector import MMDVariableSelector
from .wasserstein.config import WassersteinSelectionConfig
from .wasserstein.selector import WassersteinVariableSelector

__all__ = [
    "BaseVariableSelector",
    "MMDSelectionConfig",
    "MMDVariableSelector",
    "WassersteinSelectionConfig",
    "WassersteinVariableSelector",
]
