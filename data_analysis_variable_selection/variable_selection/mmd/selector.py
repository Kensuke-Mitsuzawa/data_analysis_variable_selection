import copy
import logging
import typing as ty
import numpy as np
import torch

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import RamBackendStaticDataset
from mmd_tst_variable_detector.distance_module import L2Distance
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.commons import (
    InterpretableMmdTrainParameters,
    RegularizationParameter,
)
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdDetector
from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer
from mmd_tst_variable_detector.utils.variable_detection import detect_variables

from ...common.base_selector import BaseVariableSelector
from ...common.models import VariableSelectionResult
from .config import MMDSelectionConfig

logger = logging.getLogger(__name__)


class MMDVariableSelector(BaseVariableSelector):
    """Executes Maximum Mean Discrepancy (MMD) variable selection to discover intrinsic anchor variables (hat_S).

    Supports configurable algorithm strategies:
    - 'algorithm_one': Full dataset optimization with ARD weights and objective ratio regularization.
    - 'mmd_cv': Cross-validation / stability selection across random subsamples with frequency aggregation.
    """

    def __init__(self, config: ty.Optional[MMDSelectionConfig] = None, **kwargs: ty.Any):
        """Initializes the selector with configuration options or keyword parameter overrides.

        Args:
            config: Optional MMDSelectionConfig instance.
            **kwargs: Dynamic keyword overrides (e.g. algorithm='mmd-cv', device='cuda', max_epochs=100).
        """
        if config is not None:
            # Apply any additional kwargs onto provided config
            data_dict = config.model_dump()
            data_dict.update(kwargs)
            self.config = MMDSelectionConfig(**data_dict)
        else:
            self.config = MMDSelectionConfig(**kwargs)
        # end if
        # end def __init__

    def select_variables(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        **kwargs: ty.Any
    ) -> VariableSelectionResult:
        """Executes MMD-based variable selection using the configured algorithm strategy.

        Args:
            sample_x: Standardized feature matrix for distribution X (n_X, d).
            sample_y: Standardized feature matrix for distribution Y (n_Y, d).
            names_variables: Feature names corresponding to columns.
            **kwargs: Per-run parameter overrides (e.g. algorithm, device, max_epochs, top_k_fallback).

        Returns:
            VariableSelectionResult containing selected anchor indices, names, weights, and algorithm metadata.
        """
        effective_config = self.config
        if kwargs:
            curr_dict = self.config.model_dump()
            curr_dict.update(kwargs)
            effective_config = MMDSelectionConfig(**curr_dict)
        # end if

        algo_name = effective_config.normalize_algorithm_name()

        if algo_name == "algorithm_one":
            return self._run_algorithm_one(
                sample_x=sample_x,
                sample_y=sample_y,
                names_variables=names_variables,
                config=effective_config,
            )
        elif algo_name == "mmd_cv":
            return self._run_mmd_cv(
                sample_x=sample_x,
                sample_y=sample_y,
                names_variables=names_variables,
                config=effective_config,
            )
        else:
            raise ValueError(f"Unknown algorithm '{algo_name}'")
        # end if
        # end def select_variables

    def select_variables_mmd(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        **kwargs: ty.Any
    ) -> VariableSelectionResult:
        """Alias for select_variables for backwards compatibility.
        """
        return self.select_variables(
            sample_x=sample_x,
            sample_y=sample_y,
            names_variables=names_variables,
            **kwargs
        )
        # end def select_variables_mmd

    def resolve_device_accelerator(self, device_spec: str) -> str:
        """Resolves device string to target hardware accelerator ('cuda' or 'cpu').

        Args:
            device_spec: 'auto', 'cpu', 'cuda', 'gpu'.

        Returns:
            Resolved hardware device string: 'cuda' or 'cpu'.
        """
        spec = device_spec.lower().strip()
        if spec == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif spec in ("cuda", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return "cpu"
        # end if
        # end def resolve_device_accelerator

    def _run_algorithm_one(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        config: MMDSelectionConfig
    ) -> VariableSelectionResult:
        """Executes Algorithm 1: ARD weight optimization on full dataset.
        """
        num_features = sample_x.shape[1]
        device_str = self.resolve_device_accelerator(config.device)

        weights = self._optimize_weights_ard(
            sample_x=sample_x,
            sample_y=sample_y,
            config=config,
            device_str=device_str,
        )

        detected_indices = self._detect_variables_from_weights(
            weights=weights,
            num_features=num_features,
            config=config,
        )

        selected_names = [names_variables[idx] for idx in detected_indices]
        selected_weights = [float(weights[idx]) for idx in detected_indices]

        return VariableSelectionResult(
            indices_selected=detected_indices,
            names_selected=selected_names,
            weights_selected=selected_weights,
            p_value=None,
            metadata_selection={
                "algorithm": "algorithm_one",
                "device_requested": config.device,
                "device_resolved": device_str,
                "all_weights": weights.tolist(),
                "num_features": num_features,
                "epochs_trained": config.max_epochs,
                "regularizer_l1": config.regularizer_l1,
                "regularizer_l2": config.regularizer_l2,
            }
        )
        # end def _run_algorithm_one

    def _run_mmd_cv(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        config: MMDSelectionConfig
    ) -> VariableSelectionResult:
        """Executes MMD-CV: Stability selection across random cross-validation subsamples.
        """
        num_features = sample_x.shape[1]
        num_x = sample_x.shape[0]
        num_y = sample_y.shape[0]
        device_str = self.resolve_device_accelerator(config.device)

        n_subsamples = max(2, config.n_cv_subsampling)
        sub_ratio = min(1.0, max(0.2, config.subsampling_ratio))

        size_sub_x = max(10, int(num_x * sub_ratio))
        size_sub_y = max(10, int(num_y * sub_ratio))

        fold_weights_list: ty.List[np.ndarray] = []
        selection_counts = np.zeros(num_features, dtype=int)

        np.random.seed(42)

        for fold_idx in range(n_subsamples):
            # Subsample X and Y
            idx_x = np.random.choice(num_x, size=size_sub_x, replace=False)
            idx_y = np.random.choice(num_y, size=size_sub_y, replace=False)

            sub_x = sample_x[idx_x]
            sub_y = sample_y[idx_y]

            weights_fold = self._optimize_weights_ard(
                sample_x=sub_x,
                sample_y=sub_y,
                config=config,
                device_str=device_str,
            )
            fold_weights_list.append(weights_fold)

            detected_fold = self._detect_variables_from_weights(
                weights=weights_fold,
                num_features=num_features,
                config=config,
            )
            for det_idx in detected_fold:
                selection_counts[det_idx] += 1
            # end for
        # end for fold_idx

        # Compute empirical selection frequencies and average weights
        selection_frequencies = selection_counts / float(n_subsamples)
        average_weights = np.mean(np.array(fold_weights_list), axis=0)

        # Select variables meeting the stability threshold
        stable_indices = np.where(selection_frequencies >= config.cv_stability_threshold)[0].tolist()

        # If none met threshold, take top_k by frequency and mean weight
        if not stable_indices:
            score_composite = selection_frequencies * 10.0 + average_weights
            k_fallback = min(config.top_k_fallback, num_features)
            stable_indices = np.argsort(-score_composite)[:k_fallback].tolist()
        # end if

        stable_indices = sorted(stable_indices)
        selected_names = [names_variables[idx] for idx in stable_indices]
        selected_weights = [float(average_weights[idx]) for idx in stable_indices]

        return VariableSelectionResult(
            indices_selected=stable_indices,
            names_selected=selected_names,
            weights_selected=selected_weights,
            p_value=None,
            metadata_selection={
                "algorithm": "mmd_cv",
                "device_requested": config.device,
                "device_resolved": device_str,
                "n_cv_subsampling": n_subsamples,
                "subsampling_ratio": sub_ratio,
                "cv_stability_threshold": config.cv_stability_threshold,
                "selection_frequencies": selection_frequencies.tolist(),
                "average_weights": average_weights.tolist(),
                "num_features": num_features,
            }
        )
        # end def _run_mmd_cv

    def _optimize_weights_ard(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        config: MMDSelectionConfig,
        device_str: str
    ) -> np.ndarray:
        """Runs the PyTorch optimization loop to estimate ARD weights.
        """
        num_features = sample_x.shape[1]

        tensor_x = torch.from_numpy(sample_x).float()
        tensor_y = torch.from_numpy(sample_y).float()

        dataset_train = RamBackendStaticDataset(x=tensor_x, y=tensor_y)
        dataset_val = RamBackendStaticDataset(x=tensor_x, y=tensor_y)

        bandwidth_init = torch.ones(num_features).float()

        kernel = QuadraticKernelGaussianKernel(
            distance_module=L2Distance(coordinate_size=1),
            bandwidth=bandwidth_init,
            ard_weights=torch.ones(num_features).float(),
            is_dimension_median_heuristic=True,
            use_fused_kernel=config.use_fused_kernel if device_str == "cuda" else False,
        )

        mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel)

        if config.regularizer_l1 is not None or config.regularizer_l2 is not None:
            lambda_1_val = float(config.regularizer_l1 if config.regularizer_l1 is not None else 0.0)
            lambda_2_val = float(config.regularizer_l2 if config.regularizer_l2 is not None else 0.0)
            train_params = InterpretableMmdTrainParameters(
                batch_size=config.batch_size,
                regularization_parameter=RegularizationParameter(
                    lambda_1=lambda_1_val,
                    lambda_2=lambda_2_val,
                )
            )
        else:
            # Follow the default regularization behavior of the package's implemented algorithm
            train_params = InterpretableMmdTrainParameters(
                batch_size=config.batch_size
            )
        # end if

        try:
            detector = InterpretableMmdDetector(
                mmd_estimator=mmd_estimator,
                training_parameter=train_params,
                dataset_train=dataset_train,
                dataset_validation=dataset_val
            )

            trainer = PurePytorchTrainer(
                max_epochs=config.max_epochs,
                accelerator=device_str,
                enable_progress_bar=False,
                logger=False,
                use_fused_kernel=config.use_fused_kernel if device_str == "cuda" else False,
            )

            trainer.fit(detector)
            weights_tensor = detector.mmd_estimator.kernel_obj.ard_weights
            weights = weights_tensor.detach().cpu().numpy()
        except Exception as exc:
            logger.warning(f"MMD optimization encountered issue ({exc}). Using difference of means heuristic.")
            mean_diff = np.abs(np.mean(sample_x, axis=0) - np.mean(sample_y, axis=0))
            weights = mean_diff / (np.max(mean_diff) + 1e-8)
        # end try

        return weights
        # end def _optimize_weights_ard

    def _detect_variables_from_weights(
        self,
        weights: np.ndarray,
        num_features: int,
        config: MMDSelectionConfig
    ) -> ty.List[int]:
        """Applies variable detection strategy (hist_based or threshold) to ARD weights.
        """
        detected_indices: ty.List[int] = []
        try:
            detected_indices = detect_variables(
                variable_weights=weights,
                variable_detection_approach=config.variable_detection_approach,
                threshold_weights=config.threshold_weights,
            )
        except Exception:
            detected_indices = []
        # end try

        # If detection selected 0 or all variables, select top_k by weight
        k_target = min(config.top_k_fallback, num_features)
        if len(detected_indices) == 0 or len(detected_indices) == num_features:
            top_ranked = np.argsort(-weights)[:k_target].tolist()
            detected_indices = top_ranked
        # end if

        if len(detected_indices) > k_target:
            detected_indices = sorted(detected_indices, key=lambda idx: weights[idx], reverse=True)[:k_target]
        # end if

        return sorted(detected_indices)
        # end def _detect_variables_from_weights
# end class MMDVariableSelector
