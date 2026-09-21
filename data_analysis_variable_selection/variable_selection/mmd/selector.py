import copy
import logging
import tempfile
from pathlib import Path
import typing as ty
import numpy as np
import torch

from mmd_tst_variable_detector import (
    Interface,
    InterfaceConfigArgs,
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DetectorAlgorithmConfigArgs,
    CvSelectionConfigArgs,
    AlgorithmOneConfigArgs,
    DistributedConfigArgs,
    RegularizationSearchParameters,
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset,
    RegularizationParameter,
    InterpretableMmdTrainParameters,
    CrossValidationAlgorithmParameter,
    CrossValidationInterpretableVariableDetector,
    DistributedComputingParameter,
    CrossValidationTrainParameters,
    PytorchLightningDefaultArguments,
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.assessment_helper.default_settings import lr_scheduler
from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import RamBackendStaticDataset
from mmd_tst_variable_detector.distance_module import L2Distance
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

    def _create_distributed_config(self, config: MMDSelectionConfig) -> DistributedConfigArgs:
        """Constructs DistributedConfigArgs from MMDSelectionConfig parameters.

        Uses values configured in MMDSelectionConfig without hardcoded constants.

        Args:
            config: MMDSelectionConfig instance.

        Returns:
            Configured DistributedConfigArgs instance.
        """
        mode = config.distributed_mode
        if mode is None:
            if config.is_use_local_dask_cluster or config.dask_scheduler_host is not None:
                mode = "dask"
            else:
                mode = "single"
            # end if
        # end if

        if mode == "dask":
            if config.is_use_local_dask_cluster:
                return DistributedConfigArgs(
                    distributed_mode="dask",
                    is_use_local_dask_cluster=True,
                    dask_n_workers=config.dask_n_workers,
                    dask_threads_per_worker=config.dask_threads_per_worker,
                    dask_dashboard_address=config.dask_dashboard_address,
                    dask_memory_limit=config.dask_memory_limit,
                )
            else:
                return DistributedConfigArgs(
                    distributed_mode="dask",
                    is_use_local_dask_cluster=False,
                    dask_scheduler_host=config.dask_scheduler_host or "0.0.0.0",
                    dask_scheduler_port=config.dask_scheduler_port,
                    dask_dashboard_address=config.dask_dashboard_address,
                    dask_memory_limit=config.dask_memory_limit,
                )
            # end if
        else:
            return DistributedConfigArgs(
                distributed_mode="single",
                is_use_local_dask_cluster=False,
            )
        # end if
    # end def _create_distributed_config

    def _run_algorithm_one(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_variables: ty.List[str],
        config: MMDSelectionConfig
    ) -> VariableSelectionResult:
        """Executes Algorithm 1 using the official Interface API from mmd_tst_variable_detector."""
        num_features = sample_x.shape[1]
        num_x = sample_x.shape[0]
        num_y = sample_y.shape[0]
        device_str = self.resolve_device_accelerator(config.device)
        accelerator = "gpu" if device_str == "cuda" else "cpu"

        # Prepare train and test splits for two-sample test using configured random seed
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        n_train_x = max(5, int(num_x * config.subsampling_ratio))
        n_train_y = max(5, int(num_y * config.subsampling_ratio))
        idx_train_x = np.random.choice(num_x, size=n_train_x, replace=False)
        idx_train_y = np.random.choice(num_y, size=n_train_y, replace=False)
        idx_test_x = np.setdiff1d(np.arange(num_x), idx_train_x)
        idx_test_y = np.setdiff1d(np.arange(num_y), idx_train_y)
        if len(idx_test_x) == 0:
            idx_test_x = idx_train_x
        # end if
        if len(idx_test_y) == 0:
            idx_test_y = idx_train_y
        # end if

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            data_config_args = DataSetConfigArgs(
                data_x_train=torch.from_numpy(sample_x[idx_train_x]).float(),
                data_y_train=torch.from_numpy(sample_y[idx_train_y]).float(),
                data_x_test=torch.from_numpy(sample_x[idx_test_x]).float(),
                data_y_test=torch.from_numpy(sample_y[idx_test_y]).float(),
                dataset_type_backend="ram",
                dataset_type_charactersitic="static",
            )

            # Configure distributed execution using parameters from config
            distributed_config = self._create_distributed_config(config)

            parameter_search_parameter = RegularizationSearchParameters(
                n_regularization_parameter=3,
                n_search_iteration=5,
                max_concurrent_job=2 if accelerator == "gpu" else 1,
            )

            interface_args = InterfaceConfigArgs(
                resource_config_args=ResourceConfigArgs(
                    train_accelerator=accelerator,
                    path_work_dir=Path(tmp_work_dir),
                    distributed_config_detection=distributed_config,
                ),
                approach_config_args=ApproachConfigArgs(
                    approach_data_representation="sample_based",
                    approach_variable_detector="interpretable_mmd",
                    approach_interpretable_mmd="algorithm_one",
                ),
                data_config_args=data_config_args,
                detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                    mmd_algorithm_one_args=AlgorithmOneConfigArgs(
                        max_epoch=config.max_epochs,
                        parameter_search_parameter=parameter_search_parameter,
                    )
                ),
            )

            try:
                interface_instance = Interface(config_args=interface_args)
                interface_instance.fit()
                res_object = interface_instance.get_result()

                det_res = res_object.detection_result_sample_based
                if det_res is not None:
                    detected_indices = list(det_res.variables) if det_res.variables is not None else []
                    weights = np.array(det_res.weights) if det_res.weights is not None else np.zeros(num_features)
                    p_value = det_res.p_value
                else:
                    detected_indices = []
                    weights = np.zeros(num_features)
                    p_value = None
                # end if
            except Exception as exc:
                logger.warning(
                    f"Interface fit for algorithm_one encountered issue ({exc}). Using difference of means fallback."
                )
                mean_diff = np.abs(np.mean(sample_x, axis=0) - np.mean(sample_y, axis=0))
                weights = mean_diff / (np.max(mean_diff) + 1e-8)
                detected_indices = []
                p_value = None
            # end try
        # end with

        k_target = min(config.top_k_fallback, num_features)
        if len(detected_indices) == 0 or len(detected_indices) == num_features:
            top_ranked = np.argsort(-weights)[:k_target].tolist()
            detected_indices = top_ranked
        elif len(detected_indices) > k_target:
            detected_indices = sorted(detected_indices, key=lambda idx: weights[idx], reverse=True)[:k_target]
        # end if

        detected_indices = sorted(detected_indices)
        selected_names = [names_variables[idx] for idx in detected_indices]
        selected_weights = [float(weights[idx]) for idx in detected_indices]

        return VariableSelectionResult(
            indices_selected=detected_indices,
            names_selected=selected_names,
            weights_selected=selected_weights,
            p_value=p_value,
            metadata_selection={
                "algorithm": "algorithm_one",
                "device_requested": config.device,
                "device_resolved": device_str,
                "accelerator": accelerator,
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
        """Executes MMD-CV using the official Interface API from mmd_tst_variable_detector."""
        num_features = sample_x.shape[1]
        num_x = sample_x.shape[0]
        num_y = sample_y.shape[0]
        device_str = self.resolve_device_accelerator(config.device)
        accelerator = "gpu" if device_str == "cuda" else "cpu"

        # Prepare train and test splits for two-sample test using configured random seed
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        n_train_x = max(5, int(num_x * config.subsampling_ratio))
        n_train_y = max(5, int(num_y * config.subsampling_ratio))
        idx_train_x = np.random.choice(num_x, size=n_train_x, replace=False)
        idx_train_y = np.random.choice(num_y, size=n_train_y, replace=False)
        idx_test_x = np.setdiff1d(np.arange(num_x), idx_train_x)
        idx_test_y = np.setdiff1d(np.arange(num_y), idx_train_y)
        if len(idx_test_x) == 0:
            idx_test_x = idx_train_x
        # end if
        if len(idx_test_y) == 0:
            idx_test_y = idx_train_y
        # end if

        with tempfile.TemporaryDirectory() as tmp_work_dir:
            data_config_args = DataSetConfigArgs(
                data_x_train=torch.from_numpy(sample_x[idx_train_x]).float(),
                data_y_train=torch.from_numpy(sample_y[idx_train_y]).float(),
                data_x_test=torch.from_numpy(sample_x[idx_test_x]).float(),
                data_y_test=torch.from_numpy(sample_y[idx_test_y]).float(),
                dataset_type_backend="ram",
                dataset_type_charactersitic="static",
            )

            # Configure distributed execution using parameters from config
            distributed_config = self._create_distributed_config(config)

            parameter_search_parameter = RegularizationSearchParameters(
                n_regularization_parameter=3,
                n_search_iteration=5,
                max_concurrent_job=2 if accelerator == "gpu" else 1,
            )

            interface_args = InterfaceConfigArgs(
                resource_config_args=ResourceConfigArgs(
                    train_accelerator=accelerator,
                    path_work_dir=Path(tmp_work_dir),
                    distributed_config_detection=distributed_config,
                ),
                approach_config_args=ApproachConfigArgs(
                    approach_data_representation="sample_based",
                    approach_variable_detector="interpretable_mmd",
                    approach_interpretable_mmd="cv_selection",
                ),
                data_config_args=data_config_args,
                detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                    mmd_cv_selection_args=CvSelectionConfigArgs(
                        max_epoch=config.max_epochs,
                        parameter_search_parameter=parameter_search_parameter,
                        n_subsampling=config.n_cv_subsampling,
                    )
                ),
            )

            try:
                interface_instance = Interface(config_args=interface_args)
                interface_instance.fit()
                res_object = interface_instance.get_result()

                det_res = res_object.detection_result_sample_based
                if det_res is not None:
                    stable_indices = list(det_res.variables) if det_res.variables is not None else []
                    weights = np.array(det_res.weights) if det_res.weights is not None else np.zeros(num_features)
                    p_value = det_res.p_value
                else:
                    stable_indices = []
                    weights = np.zeros(num_features)
                    p_value = None
                # end if
            except Exception as exc:
                logger.warning(
                    f"Interface fit encountered issue ({exc}). Using difference of means fallback."
                )
                mean_diff = np.abs(np.mean(sample_x, axis=0) - np.mean(sample_y, axis=0))
                weights = mean_diff / (np.max(mean_diff) + 1e-8)
                stable_indices = []
                p_value = None
            # end try
        # end with

        k_target = min(config.top_k_fallback, num_features)
        if len(stable_indices) == 0 or len(stable_indices) == num_features:
            top_ranked = np.argsort(-weights)[:k_target].tolist()
            stable_indices = top_ranked
        elif len(stable_indices) > k_target:
            stable_indices = sorted(stable_indices, key=lambda idx: weights[idx], reverse=True)[:k_target]
        # end if

        stable_indices = sorted(stable_indices)
        selected_names = [names_variables[idx] for idx in stable_indices]
        selected_weights = [float(weights[idx]) for idx in stable_indices]

        return VariableSelectionResult(
            indices_selected=stable_indices,
            names_selected=selected_names,
            weights_selected=selected_weights,
            p_value=p_value,
            metadata_selection={
                "algorithm": "mmd_cv",
                "device_requested": config.device,
                "device_resolved": device_str,
                "accelerator": accelerator,
                "selection_frequencies": [float(w) for w in weights],
                "n_cv_subsampling": config.n_cv_subsampling,
                "subsampling_ratio": config.subsampling_ratio,
                "cv_stability_threshold": config.cv_stability_threshold,
                "num_features": num_features,
            }
        )
        # end def _run_mmd_cv
# end class MMDVariableSelector

