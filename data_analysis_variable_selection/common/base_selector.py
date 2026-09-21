import abc
import typing as ty
import numpy as np

from .models import VariableSelectionResult


class BaseVariableSelector(abc.ABC):
    """Abstract base class for two-sample variable selection algorithms.

    Subclasses implement coordinate or subspace selection strategies (e.g. MMD, 1D-Wasserstein)
    to identify anchor variables (hat_S) distinguishing distribution X from distribution Y.
    """

    @abc.abstractmethod
    def select_variables(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        **kwargs: ty.Any
    ) -> VariableSelectionResult:
        """Executes variable selection across two-sample distributions.

        Args:
            sample_x: Feature matrix for distribution X of shape (n_X, d).
            sample_y: Feature matrix for distribution Y of shape (n_Y, d).
            names_variables: List of variable/feature names corresponding to columns.
            **kwargs: Dynamic algorithm-specific keyword parameters or overrides.

        Returns:
            VariableSelectionResult containing selected indices, names, weights, and metadata.
        """
        raise NotImplementedError()
        # end def select_variables
# end class BaseVariableSelector
