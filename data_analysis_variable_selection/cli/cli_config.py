import tomllib
import typing as ty
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ProjectConfig(BaseModel):
    """General project and storage configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_directory: str = Field(
        default_factory=lambda: str(Path("outputs/run").resolve()),
        description="Path to directory where DuckDB, artifacts, and reports are saved (must be absolute)."
    )
    database_file: str = Field(
        default="analysis_warehouse.duckdb",
        description="Filename of the DuckDB database."
    )
    dataset_name: str = Field(
        default="ames_housing",
        description="Target dataset identifier: 'ames_housing' or 'speed_dating'."
    )
    directory_name_log: str = Field(
        default="log",
        description="Directory name for logs."
    )
    file_name_log: str = Field(
        default="pipeline_run.log",
        description="Filename for logs."
    )

    @field_validator("output_directory")
    @classmethod
    def validate_absolute_output_directory(cls, v: str) -> str:
        """Validates that output_directory is defined as an absolute path.

        Args:
            v: Output directory path string.

        Returns:
            Validated absolute path string.

        Raises:
            ValueError: If output_directory is not an absolute path.
        """
        path_obj = Path(v)
        if not path_obj.is_absolute():
            raise ValueError(
                f"The parameter 'output_directory' must be an absolute path definition, got: '{v}'. "
                f"Relative paths are likely to cause path resolution errors during variable selection and pipeline runs. "
                f"Please specify an absolute path (e.g., '{path_obj.resolve()}')."
            )
        # end if
        return str(path_obj)
    # end def validate_absolute_output_directory
# end class ProjectConfig


class AmesDatasetConfig(BaseModel):
    """Ames Housing dataset specific configuration."""
    raw_data_path: ty.Optional[str] = Field(
        default=None,
        description="Path to local raw CSV file. If None, loaded via OpenML or synthetic fallback."
    )
    url_download: ty.Optional[str] = Field(
        default="https://www.openml.org/data/get_csv/21785545/ames_housing.csv",
        description="URL to download raw dataset if file does not exist."
    )
    max_records_per_distribution: ty.Optional[int] = Field(
        default=500,
        description="Subsample limit per distribution to control O(N^2) complexity."
    )
    random_seed_sampling: int = Field(
        default=42,
        description="Seed for deterministic record subsampling."
    )
# end class AmesDatasetConfig


class SpeedDatingDatasetConfig(BaseModel):
    """Speed Dating dataset specific configuration."""
    raw_data_path: ty.Optional[str] = Field(
        default=None,
        description="Path to local raw CSV file."
    )
    url_download: ty.Optional[str] = Field(
        default="https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment",
        description="Source URL for the dataset on Kaggle: 'https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment'."
    )
    max_records_per_distribution: ty.Optional[int] = Field(
        default=500,
        description="Subsample limit per distribution to control O(N^2) complexity."
    )
    random_seed_sampling: int = Field(
        default=42,
        description="Seed for deterministic record subsampling."
    )
# end class SpeedDatingDatasetConfig


class DatasetConfig(BaseModel):
    """Container for dataset configurations."""
    ames_housing: AmesDatasetConfig = Field(default_factory=AmesDatasetConfig)
    speed_dating: SpeedDatingDatasetConfig = Field(default_factory=SpeedDatingDatasetConfig)
# end class DatasetConfig


class MMDAlgorithmConfig(BaseModel):
    """Options for MMD Variable Selector."""
    algorithm: str = Field(
        default="algorithm_one",
        description="MMD selection algorithm: 'algorithm_one' or 'mmd_cv'."
    )
    device: str = Field(
        default="auto",
        description="Target compute device: 'auto', 'cpu', 'cuda', 'gpu'."
    )
    max_epochs: int = Field(
        default=9999,
        description="Maximum optimization epochs for ARD weights."
    )
    top_k_fallback: int = Field(
        default=5,
        description="Fallback top-K variables if optimization selects none."
    )
    regularizer_l1: ty.Optional[float] = Field(
        default=None,
        description="L1 sparsity regularization coefficient (None uses package default)."
    )
    regularizer_l2: ty.Optional[float] = Field(
        default=None,
        description="L2 smooth regularization coefficient (None uses package default)."
    )
    learning_rate: float = Field(
        default=0.01,
        description="Learning rate for Adam optimizer in ARD weight fitting."
    )
    variable_detection_approach: str = Field(
        default="hist_based",
        description="Thresholding approach: 'hist_based' or 'threshold'."
    )
    threshold_weights: float = Field(
        default=0.1,
        description="Weight cutoff when using threshold detection approach."
    )
    n_cv_subsampling: int = Field(
        default=5,
        description="Number of subsampling folds for mmd_cv stability selection."
    )
    subsampling_ratio: float = Field(
        default=0.8,
        description="Subsampling ratio per fold for mmd_cv."
    )
    cv_stability_threshold: float = Field(
        default=0.5,
        description="Selection frequency threshold to retain stable features in mmd_cv."
    )
    use_fused_kernel: bool = Field(
        default=False,
        description="Whether to use fused Triton CUDA kernel when running on GPU."
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducible train/test splitting and subsampling in MMD selection."
    )
    distributed_mode: ty.Optional[str] = Field(
        default=None,
        description="Distributed mode: 'single' or 'dask'. If None, inferred automatically."
    )
    is_use_local_dask_cluster: bool = Field(
        default=False,
        description="Whether to spin up a local concurrent Dask cluster for distributed MMD computation."
    )
    dask_scheduler_host: ty.Optional[str] = Field(
        default=None,
        description="Host address of external Dask scheduler if using distributed cluster."
    )
    dask_scheduler_port: int = Field(
        default=8786,
        description="Port of external Dask scheduler."
    )
    dask_dashboard_address: ty.Optional[str] = Field(
        default=":8787",
        description="Dashboard address for Dask."
    )
    dask_n_workers: int = Field(
        default=4,
        description="Number of Dask workers when using local Dask cluster."
    )
    dask_threads_per_worker: int = Field(
        default=2,
        description="Number of threads per worker when using local Dask cluster."
    )
    dask_memory_limit: ty.Optional[ty.Union[str, int]] = Field(
        default=0,
        description="Memory limit per worker for Dask cluster (e.g. 0 or '4GB')."
    )
# end class MMDAlgorithmConfig


class WassersteinAlgorithmConfig(BaseModel):
    """Options for 1D-Wasserstein Variable Selector."""
    distributed_backend: str = Field(
        default="single",
        description="Computation backend: 'single', 'joblib', 'dask'."
    )
    variable_detection_approach: str = Field(
        default="hist_based",
        description="Thresholding approach: 'hist_based' or 'threshold'."
    )
    threshold_weights: float = Field(
        default=0.1,
        description="Weight cutoff when using threshold detection approach."
    )
    top_k_fallback: int = Field(
        default=5,
        description="Fallback top-K variables if detector selects none."
    )
# end class WassersteinAlgorithmConfig


class VariableDetectionConfig(BaseModel):
    """Configuration for variable detection stage."""
    method: str = Field(
        default="mmd",
        description="Selection approach: 'mmd' or 'wasserstein'."
    )
    mmd: MMDAlgorithmConfig = Field(default_factory=MMDAlgorithmConfig)
    wasserstein: WassersteinAlgorithmConfig = Field(default_factory=WassersteinAlgorithmConfig)
# end class VariableDetectionConfig


class VariableAnalysisConfig(BaseModel):
    """Configuration for correlation, clustering, and prototyping."""
    correlation_method: str = Field(
        default="graphical_lasso",
        description="Relationship matrix algorithm: 'graphical_lasso' or 'pearson'."
    )
    correlation_threshold: float = Field(
        default=0.2,
        description="Edge filtering threshold for graph and cluster connection."
    )
    num_clusters: int = Field(
        default=5,
        description="Number of thematic clusters for agglomerative clustering."
    )
    sample_scope: str = Field(
        default="subset",
        description="Sample scope for variable correlation and clustering: 'subset' (default) or 'whole'. Graphical Lasso requires huge RAM as sample size increases, so 'subset' is recommended."
    )
# end class VariableAnalysisConfig


class ReportConfig(BaseModel):
    """Configuration for report and deliverable export."""
    report_title: str = Field(
        default="Two-Sample Variable Selection Analysis Report",
        description="Title of the executive report."
    )
    export_excel: bool = Field(default=True, description="Whether to export multi-sheet Excel workbook.")
    export_plots: bool = Field(default=True, description="Whether to generate visual charts.")
    export_markdown: bool = Field(default=True, description="Whether to generate Markdown report summary.")
    export_dataset_report: bool = Field(
        default=True,
        description="Whether to export dataset-specific exploratory report with shallow-level statistics."
    )
    dataset_report_title: ty.Optional[str] = Field(
        default=None,
        description="Optional title for the dataset-specific report."
    )
    sample_scope: str = Field(
        default="whole",
        description="Sample scope for marginal distribution plots, prototype exemplars, and persona radar charts: 'whole' (default) or 'subset'."
    )
# end class ReportConfig


class PipelineCliConfig(BaseModel):
    """Top-level pipeline configuration parsed from TOML."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    variable_detection: VariableDetectionConfig = Field(default_factory=VariableDetectionConfig)
    variable_analysis: VariableAnalysisConfig = Field(default_factory=VariableAnalysisConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    def get_database_path(self) -> str:
        """Returns resolved absolute or relative path to DuckDB warehouse."""
        return str(Path(self.project.output_directory) / self.project.database_file)
        # end def get_database_path

    def get_features_container_path(self) -> str:
        """Returns path to cached preprocessed features container (.npz)."""
        return str(Path(self.project.output_directory) / "preprocessed_features.npz")
        # end def get_features_container_path

    def get_log_file_path(self) -> str:
        """Returns resolved path to the log file in project.output_directory / directory_name_log / file_name_log."""
        dir_log = Path(self.project.output_directory) / self.project.directory_name_log
        dir_log.mkdir(parents=True, exist_ok=True)
        return str(dir_log / self.project.file_name_log)
        # end def get_log_file_path
# end class PipelineCliConfig


def load_toml_config(path_toml: str) -> PipelineCliConfig:
    """Parses and validates a TOML configuration file.

    Args:
        path_toml: Filepath to the TOML configuration.

    Returns:
        Validated PipelineCliConfig instance.
    """
    config_path = Path(path_toml)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path_toml}")
    # end if

    with config_path.open("rb") as f_in:
        raw_dict = tomllib.load(f_in)
    # end with

    return PipelineCliConfig(**raw_dict)
    # end def load_toml_config
