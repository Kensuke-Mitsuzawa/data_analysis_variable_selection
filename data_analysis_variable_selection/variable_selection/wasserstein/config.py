import typing as ty
from pydantic import BaseModel, Field, ConfigDict


class WassersteinSelectionConfig(BaseModel):
    """User-facing configuration options for 1D-Wasserstein variable selection.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Computation execution backend: 'single', 'joblib', or 'dask'
    distributed_backend: str = Field(
        default="single",
        description="Parallelization backend for 1D-Wasserstein distance computation: 'single', 'joblib', or 'dask'."
    )
    dask_client: ty.Optional[ty.Any] = Field(
        default=None,
        description="Optional Dask client instance when distributed_backend is 'dask'."
    )

    # Detection / Thresholding parameters
    variable_detection_approach: str = Field(
        default="hist_based",
        description="Detection thresholding strategy: 'hist_based' (valley detection in histogram) or 'threshold' (fixed cut)."
    )
    threshold_weights: float = Field(
        default=0.1,
        description="Threshold cut value when variable_detection_approach is 'threshold'."
    )
    top_k_fallback: int = Field(
        default=5,
        description="Fallback number of top variables to select if no variables meet detection threshold."
    )
    is_normalize_weights: bool = Field(
        default=True,
        description="Whether to normalize Wasserstein weights to maximum 1.0."
    )
# end class WassersteinSelectionConfig
