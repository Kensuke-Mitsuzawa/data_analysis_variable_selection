import typing as ty
from pydantic import BaseModel, Field, ConfigDict


class MMDSelectionConfig(BaseModel):
    """User-facing configuration options for MMD variable selection algorithms.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core algorithm option: 'algorithm_one' or 'mmd_cv'
    algorithm: str = Field(
        default="algorithm_one",
        description="Variable selection algorithm: 'algorithm_one' (Algorithm 1) or 'mmd_cv' (Cross-Validation / Stability Selection)."
    )

    # Hardware / Device
    device: str = Field(
        default="auto",
        description="Hardware accelerator device: 'auto' (resolves to cuda if available, else cpu), 'cpu', 'cuda', 'gpu'."
    )

    # Optimization parameters
    max_epochs: int = Field(
        default=9999,
        description="Maximum training epochs for ARD weight optimization."
    )
    batch_size: int = Field(
        default=-1,
        description="Batch size for training. -1 for full dataset batch."
    )
    learning_rate: float = Field(
        default=0.005,
        description="Learning rate for Adam optimizer."
    )

    # Regularization parameters (sparsity vs smooth fit)
    regularizer_l1: ty.Optional[float] = Field(
        default=None,
        description="L1 penalty on ARD weights promoting sparsity. If None, uses default from implemented algorithm in package."
    )
    regularizer_l2: ty.Optional[float] = Field(
        default=None,
        description="L2 penalty on ARD weights. If None, uses default from implemented algorithm in package."
    )

    # Variable detection / thresholding
    variable_detection_approach: str = Field(
        default="hist_based",
        description="Detection thresholding strategy: 'hist_based' (valley in weight histogram) or 'threshold' (fixed cut)."
    )
    threshold_weights: float = Field(
        default=0.1,
        description="Threshold cut value when variable_detection_approach is 'threshold'."
    )
    top_k_fallback: int = Field(
        default=5,
        description="Fallback / default number of anchor variables to retain."
    )

    # Cross-validation / Subsampling parameters (used when algorithm is 'mmd_cv')
    n_cv_subsampling: int = Field(
        default=5,
        description="Number of subsampling repetitions / folds for mmd_cv stability selection."
    )
    subsampling_ratio: float = Field(
        default=0.8,
        description="Fraction of samples used in each cross-validation fold."
    )
    cv_stability_threshold: float = Field(
        default=0.4,
        description="Minimum selection frequency across CV folds to deem an anchor variable stable."
    )

    # Engine and kernel acceleration
    trainer_backend: str = Field(
        default="pure_pytorch",
        description="Training runner engine: 'pure_pytorch' or 'lightning'."
    )
    use_fused_kernel: bool = Field(
        default=False,
        description="Whether to use fused Triton CUDA kernel when running on GPU."
    )

    # Random seed
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducible train/test splitting and subsampling in MMD selection."
    )

    # Distributed / Dask options
    distributed_mode: ty.Optional[str] = Field(
        default=None,
        description="Distributed mode: 'single' or 'dask'. If None, inferred automatically based on is_use_local_dask_cluster / dask_scheduler_host."
    )
    dask_client: ty.Optional[ty.Any] = Field(
        default=None,
        description="Optional pre-existing Dask distributed Client object."
    )
    dask_scheduler_host: ty.Optional[str] = Field(
        default=None,
        description="Host address of Dask scheduler if distributed cluster is configured."
    )
    dask_scheduler_port: int = Field(
        default=8786,
        description="Port of Dask scheduler."
    )
    dask_dashboard_address: ty.Optional[str] = Field(
        default=":8787",
        description="Dask dashboard address."
    )
    dask_n_workers: int = Field(
        default=4,
        description="Number of Dask workers when using local Dask cluster."
    )
    dask_threads_per_worker: int = Field(
        default=2,
        description="Number of threads per Dask worker."
    )
    dask_memory_limit: ty.Optional[ty.Union[str, int]] = Field(
        default=0,
        description="Memory limit per Dask worker (0 or string like '4GB')."
    )
    is_use_local_dask_cluster: bool = Field(
        default=False,
        description="Whether to create a local Dask cluster."
    )

    def normalize_algorithm_name(self) -> str:
        """Normalizes algorithm alias strings (e.g. 'algorithm-one' -> 'algorithm_one').

        Returns:
            Normalized string: 'algorithm_one' or 'mmd_cv'.
        """
        algo_cleaned = self.algorithm.strip().lower().replace("-", "_")
        if algo_cleaned in ("algorithm_one", "algorithm1", "one"):
            return "algorithm_one"
        elif algo_cleaned in ("mmd_cv", "cv_selection", "cv", "stability_selection"):
            return "mmd_cv"
        else:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. Choose from 'algorithm_one' or 'mmd_cv'."
            )
        # end if
        # end def normalize_algorithm_name
# end class MMDSelectionConfig
