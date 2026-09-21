import logging
import typing as ty
import numpy as np
import torch

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import RamBackendStaticDataset
from mmd_tst_variable_detector.weights_initialization import weights_initialization
from mmd_tst_variable_detector.utils.variable_detection import detect_variables

from ...common.base_selector import BaseVariableSelector
from ...common.models import VariableSelectionResult
from .config import WassersteinSelectionConfig

logger = logging.getLogger(__name__)


class WassersteinVariableSelector(BaseVariableSelector):
    """Executes two-sample variable selection using 1D-Wasserstein distance across feature coordinates.

    Utilizes the coordinate-wise 1D-Wasserstein distance algorithm implemented in `mmd_tst_variable_detector`
    to rank features distinguishing distribution X from distribution Y.
    """

    def __init__(
        self,
        config: ty.Optional[WassersteinSelectionConfig] = None,
        **kwargs: ty.Any
    ):
        """Initializes the 1D-Wasserstein variable selector.

        Args:
            config: Optional WassersteinSelectionConfig instance.
            **kwargs: Dynamic configuration parameter overrides (e.g. distributed_backend='single').
        """
        if config is not None:
            data_dict = config.model_dump()
            data_dict.update(kwargs)
            self.config = WassersteinSelectionConfig(**data_dict)
        else:
            self.config = WassersteinSelectionConfig(**kwargs)
        # end if
        # end def __init__

    def select_variables(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        **kwargs: ty.Any
    ) -> VariableSelectionResult:
        """Executes 1D-Wasserstein variable selection between distributions X and Y.

        Args:
            sample_x: Feature matrix for distribution X of shape (n_X, d).
            sample_y: Feature matrix for distribution Y of shape (n_Y, d).
            names_variables: Feature names corresponding to columns.
            **kwargs: Per-run parameter overrides (e.g. top_k_fallback, threshold_weights).

        Returns:
            VariableSelectionResult containing selected variable indices, names, weights, and metadata.
        """
        effective_config = self.config
        if kwargs:
            curr_dict = self.config.model_dump()
            curr_dict.update(kwargs)
            effective_config = WassersteinSelectionConfig(**curr_dict)
        # end if

        num_features = sample_x.shape[1]
        assert num_features == sample_y.shape[1], (
            f"Feature count mismatch: X has {num_features}, Y has {sample_y.shape[1]}"
        )
        assert len(names_variables) == num_features, (
            f"names_variables length {len(names_variables)} does not match feature count {num_features}"
        )

        tensor_x = torch.from_numpy(sample_x).float()
        tensor_y = torch.from_numpy(sample_y).float()
        dataset_static = RamBackendStaticDataset(x=tensor_x, y=tensor_y)

        # Compute coordinate-wise 1D-Wasserstein distances via package implementation
        weights = weights_initialization(
            dataset_input=dataset_static,
            approach_name="wasserstein",
            distributed_backend=effective_config.distributed_backend,
            dask_client=effective_config.dask_client,
        )

        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights, dtype=np.float64)
        # end if

        if effective_config.is_normalize_weights and np.max(weights) > 0.0:
            weights = weights / np.max(weights)
        # end if

        detected_indices = self._detect_variables_from_weights(
            weights=weights,
            num_features=num_features,
            config=effective_config,
        )

        selected_names = [names_variables[idx] for idx in detected_indices]
        selected_weights = [float(weights[idx]) for idx in detected_indices]

        return VariableSelectionResult(
            indices_selected=detected_indices,
            names_selected=selected_names,
            weights_selected=selected_weights,
            p_value=None,
            metadata_selection={
                "algorithm": "1d_wasserstein",
                "distributed_backend": effective_config.distributed_backend,
                "all_weights": weights.tolist(),
                "num_features": num_features,
                "detection_approach": effective_config.variable_detection_approach,
            }
        )
        # end def select_variables

    def select_variables_wasserstein(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        **kwargs: ty.Any
    ) -> VariableSelectionResult:
        """Alias for select_variables.
        """
        return self.select_variables(
            sample_x=sample_x,
            sample_y=sample_y,
            names_variables=names_variables,
            **kwargs
        )
        # end def select_variables_wasserstein

    def _detect_variables_from_weights(
        self,
        weights: np.ndarray,
        num_features: int,
        config: WassersteinSelectionConfig
    ) -> ty.List[int]:
        """Detects active anchor variable indices from normalized weights with top-k fallback.

        Args:
            weights: Array of 1D-Wasserstein weights of shape (d,).
            num_features: Number of features.
            config: WassersteinSelectionConfig.

        Returns:
            Sorted list of integer column indices.
        """
        detected_indices: ty.List[int] = []

        try:
            detected_indices = detect_variables(
                variable_weights=weights,
                variable_detection_approach=config.variable_detection_approach,
                threshold_weights=config.threshold_weights,
            )
        except Exception as exc:
            logger.warning(f"detect_variables encountered issue ({exc}). Using threshold fallback.")
            detected_indices = np.where(weights >= config.threshold_weights)[0].tolist()
        # end try

        if not detected_indices:
            k = min(config.top_k_fallback, num_features)
            detected_indices = np.argsort(-weights)[:k].tolist()
        # end if

        return sorted(detected_indices)
        # end def _detect_variables_from_weights
# end class WassersteinVariableSelector
