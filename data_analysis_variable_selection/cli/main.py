import logging
import os
import sys
import typing as ty
import typer
import numpy as np
from pathlib import Path
import shutil

from .cli_config import PipelineCliConfig, load_toml_config
from ..datasets.setup_handler import DatasetSetupHandler
from ..datasets.feature_tracker import FeatureOperationTracker
from ..datasets.base_report_generator import BaseDatasetReportGenerator
from ..datasets.ames_housing.config import AmesPreprocessingConfig
from ..datasets.ames_housing.preprocessor import AmesHousingPreprocessor
from ..datasets.ames_housing.report_generator import AmesHousingReportGenerator
from ..datasets.speed_dating.config import SpeedDatingPreprocessingConfig
from ..datasets.speed_dating.preprocessor import SpeedDatingPreprocessor
from ..datasets.speed_dating.report_generator import SpeedDatingReportGenerator
from ..common.models import TwoSampleDataContainer, VariableSelectionResult, CorrelationResult, VariableClusteringResult, PrototypeSampleResult
from ..common.scaler import ZScoreFeatureScaler
from ..variable_selection.mmd.config import MMDSelectionConfig
from ..variable_selection.mmd.selector import MMDVariableSelector
from ..variable_selection.wasserstein.config import WassersteinSelectionConfig
from ..variable_selection.wasserstein.selector import WassersteinVariableSelector
from ..correlation.analyzer import CorrelationAnalyzer
from ..clustering.clusterer import VariableClusterer
from ..prototype.extractor import PrototypeExtractor
from ..database.manager import DuckDBStorageManager
from ..export.artifact_exporter import ArtifactExporter
from ..export.report_synthesizer import ReportSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cui_pipeline")


def configure_pipeline_logging(cfg: PipelineCliConfig) -> None:
    """Configures root logging to emit messages to both sys.stderr and the configured log file."""
    log_file_path = cfg.get_log_file_path()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    # end for

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
# end def configure_pipeline_logging

app = typer.Typer(
    name="cui_pipeline",
    help="Command-line user interface for Two-Sample Variable Selection & Analysis Pipeline.",
    add_completion=False,
)


def copy_config_file(path_toml_config: Path, cfg: PipelineCliConfig) -> None:
    """Copy the TOML configuration file to the output directory.
    Saving the original TOML configuration file for reproducibility.
    """
    path_destination_config = Path(cfg.project.output_directory) / "config.toml"
    
    shutil.copy2(path_toml_config, path_destination_config)
    typer.echo(f"✓ Saved configuration to: {path_destination_config}")
# end def copy_config_file


@app.command("setup")
def cmd_setup(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Setup datasets: download and uncompress raw files as specified in TOML config."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)
    os.makedirs(cfg.project.output_directory, exist_ok=True)
    handler = DatasetSetupHandler()
    raw_path = handler.setup_dataset(cfg)
    copy_config_file(config, cfg)
    typer.echo(f"✓ Setup complete. Verified raw dataset at: {raw_path}")
    # end def cmd_setup


@app.command("preprocess")
def cmd_preprocess(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Preprocess data: transform features, save human-readable table in DuckDB, and cache array container."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)
    os.makedirs(cfg.project.output_directory, exist_ok=True)

    dataset_name = cfg.project.dataset_name.lower().strip()
    if dataset_name == "ames_housing":
        ames_cfg = AmesPreprocessingConfig(
            path_data_file=cfg.dataset.ames_housing.raw_data_path,
            max_records_per_distribution=cfg.dataset.ames_housing.max_records_per_distribution,
            random_seed_sampling=cfg.dataset.ames_housing.random_seed_sampling,
        )
        preprocessor = AmesHousingPreprocessor(config=ames_cfg)
    elif dataset_name == "speed_dating":
        sd_cfg = SpeedDatingPreprocessingConfig(
            path_data_file=cfg.dataset.speed_dating.raw_data_path,
            url_download=cfg.dataset.speed_dating.url_download,
            max_records_per_distribution=cfg.dataset.speed_dating.max_records_per_distribution,
            random_seed_sampling=cfg.dataset.speed_dating.random_seed_sampling,
        )
        preprocessor = SpeedDatingPreprocessor(config=sd_cfg)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not yet supported for preprocessing.")
    # end if

    container = preprocessor.prepare_two_sample_data(apply_subsampling=False)

    # 1. Automatically track feature operations and export feature list markdown document
    tracker = FeatureOperationTracker()
    list_features = tracker.track_features_dataset(
        preprocessor=preprocessor,
        config=cfg,
        container=container,
    )
    path_features_md = tracker.export_feature_list_markdown(
        list_features=list_features,
        config=cfg,
    )

    # 2. Save human-readable table into DuckDB
    db = DuckDBStorageManager(path_database=cfg.get_database_path())
    db.initialize_database_schema()
    db.insert_preprocessed_features(container)
    db.close_connection_database()

    # 3. Save array container to working directory
    npz_path = cfg.get_features_container_path()
    container.save_to_npz(npz_path)

    typer.echo(
        f"✓ Preprocessing complete. X shape: {container.sample_matrix_x.shape}, "
        f"Y shape: {container.sample_matrix_y.shape}, Features: {len(container.name_features)}"
    )
    typer.echo(f"✓ Tracked {len(list_features)} feature operations and exported to: {path_features_md}")
    typer.echo(f"✓ Saved preprocessed human-readable table to DuckDB: {cfg.get_database_path()}")
    typer.echo(f"✓ Cached preprocessed arrays to: {npz_path}")

    copy_config_file(config, cfg)
    # end def cmd_preprocess


@app.command("variable-detection")
def cmd_variable_detection(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Execute variable selection (MMD or 1D-Wasserstein) and persist anchor variables."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)

    # Load preprocessed container
    npz_path = cfg.get_features_container_path()
    logger.info("loading dataset.")
    if os.path.exists(npz_path):
        container = TwoSampleDataContainer.load_from_npz(npz_path)
    else:
        db = DuckDBStorageManager(path_database=cfg.get_database_path())
        container = db.fetch_preprocessed_features()
        db.close_connection_database()
    # end if
    logger.info(f"Loaded dataset: X shape {container.sample_matrix_x.shape}, Y shape {container.sample_matrix_y.shape}, Features {len(container.name_features)}")

    # Subsample if max_records_per_distribution is configured
    limit_records = None
    seed = 42
    if cfg.project.dataset_name.lower().strip() == "ames_housing":
        limit_records = cfg.dataset.ames_housing.max_records_per_distribution
        seed = cfg.dataset.ames_housing.random_seed_sampling
    elif cfg.project.dataset_name.lower().strip() == "speed_dating":
        limit_records = cfg.dataset.speed_dating.max_records_per_distribution
        seed = cfg.dataset.speed_dating.random_seed_sampling
    # end if

    if limit_records is not None and limit_records > 0:
        container_detection = container.create_subsample_container(
            max_records_per_distribution=limit_records,
            random_seed=seed,
        )
    else:
        container_detection = container
    # end if
    logger.info(f"Variable detection sample: X shape {container_detection.sample_matrix_x.shape}, Y shape {container_detection.sample_matrix_y.shape}")

    # Standardize
    logger.info("Standardizing dataset.")
    scaler = ZScoreFeatureScaler()
    scaled = scaler.scale_features_zscore(container_detection)
    logger.info("Standardizing dataset. Done.")

    logger.info("Executing variable detection.")
    method = cfg.variable_detection.method.lower().strip()
    if method == "mmd":
        logger.info("Using MMD method.")
        mmd_cfg = MMDSelectionConfig(
            algorithm=cfg.variable_detection.mmd.algorithm,
            device=cfg.variable_detection.mmd.device,
            max_epochs=cfg.variable_detection.mmd.max_epochs,
            top_k_fallback=cfg.variable_detection.mmd.top_k_fallback,
            regularizer_l1=cfg.variable_detection.mmd.regularizer_l1,
            regularizer_l2=cfg.variable_detection.mmd.regularizer_l2,
            learning_rate=cfg.variable_detection.mmd.learning_rate,
            variable_detection_approach=cfg.variable_detection.mmd.variable_detection_approach,
            threshold_weights=cfg.variable_detection.mmd.threshold_weights,
            n_cv_subsampling=cfg.variable_detection.mmd.n_cv_subsampling,
            subsampling_ratio=cfg.variable_detection.mmd.subsampling_ratio,
            cv_stability_threshold=cfg.variable_detection.mmd.cv_stability_threshold,
            use_fused_kernel=cfg.variable_detection.mmd.use_fused_kernel,
            random_seed=cfg.variable_detection.mmd.random_seed,
            distributed_mode=cfg.variable_detection.mmd.distributed_mode,
            is_use_local_dask_cluster=cfg.variable_detection.mmd.is_use_local_dask_cluster,
            dask_scheduler_host=cfg.variable_detection.mmd.dask_scheduler_host,
            dask_scheduler_port=cfg.variable_detection.mmd.dask_scheduler_port,
            dask_dashboard_address=cfg.variable_detection.mmd.dask_dashboard_address,
            dask_n_workers=cfg.variable_detection.mmd.dask_n_workers,
            dask_threads_per_worker=cfg.variable_detection.mmd.dask_threads_per_worker,
            dask_memory_limit=cfg.variable_detection.mmd.dask_memory_limit,
        )
        selector = MMDVariableSelector(config=mmd_cfg)
    elif method == "wasserstein":
        logger.info("Using Wasserstein method.")
        wass_cfg = WassersteinSelectionConfig(
            distributed_backend=cfg.variable_detection.wasserstein.distributed_backend,
            variable_detection_approach=cfg.variable_detection.wasserstein.variable_detection_approach,
            threshold_weights=cfg.variable_detection.wasserstein.threshold_weights,
            top_k_fallback=cfg.variable_detection.wasserstein.top_k_fallback,
        )
        selector = WassersteinVariableSelector(config=wass_cfg)
    else:
        raise ValueError(f"Unknown variable detection method '{method}'. Use 'mmd' or 'wasserstein'.")
    # end if

    logger.info("Executing variable detection.")
    result = selector.select_variables(
        sample_x=scaled.sample_matrix_x_scaled,
        sample_y=scaled.sample_matrix_y_scaled,
        names_variables=scaled.name_features,
    )
    logger.info("Executing variable detection. Done.")

    logger.info("Persisting variable detection results to DuckDB.")
    # Persist to DuckDB
    db = DuckDBStorageManager(path_database=cfg.get_database_path())
    db.initialize_database_schema()
    db.insert_records_selection(result)
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_selection_x', ?)",
        [str(container_detection.sample_matrix_x.shape[0])]
    )
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_selection_y', ?)",
        [str(container_detection.sample_matrix_y.shape[0])]
    )
    is_sub = (limit_records is not None and limit_records > 0 and (container_detection.sample_matrix_x.shape[0] < container.sample_matrix_x.shape[0]))
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('scope_variable_selection', ?)",
        ["subset" if is_sub else "whole"]
    )
    db.close_connection_database()

    typer.echo(f"✓ Variable detection complete ({method}). Discovered {len(result.indices_selected)} anchor variables:")
    for name_var, weight in zip(result.names_selected, result.weights_selected):
        typer.echo(f"  - {name_var}: weight = {weight:.4f}")
    # end for
    logger.info("Persisting variable detection results to DuckDB. Done.")

    copy_config_file(config, cfg)
    # end def cmd_variable_detection


@app.command("variable-analysis")
def cmd_variable_analysis(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Analyze relationships: calculate correlation/precision matrix and cluster variables."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)

    # Load preprocessed container and anchors from DB
    npz_path = cfg.get_features_container_path()
    if os.path.exists(npz_path):
        container = TwoSampleDataContainer.load_from_npz(npz_path)
    else:
        db = DuckDBStorageManager(path_database=cfg.get_database_path())
        container = db.fetch_preprocessed_features()
        db.close_connection_database()
    # end if

    # Determine sample scope for variable analysis (subset vs whole)
    analysis_scope = cfg.variable_analysis.sample_scope.lower().strip()
    limit_records = None
    seed = 42
    if cfg.project.dataset_name.lower().strip() == "ames_housing":
        limit_records = cfg.dataset.ames_housing.max_records_per_distribution
        seed = cfg.dataset.ames_housing.random_seed_sampling
    elif cfg.project.dataset_name.lower().strip() == "speed_dating":
        limit_records = cfg.dataset.speed_dating.max_records_per_distribution
        seed = cfg.dataset.speed_dating.random_seed_sampling
    # end if

    if "sub" in analysis_scope and limit_records is not None and limit_records > 0:
        container_analysis = container.create_subsample_container(
            max_records_per_distribution=limit_records,
            random_seed=seed,
        )
    else:
        container_analysis = container
    # end if
    logger.info(
        f"Variable analysis ({analysis_scope}) dataset: X shape {container_analysis.sample_matrix_x.shape}, "
        f"Y shape {container_analysis.sample_matrix_y.shape}"
    )

    scaler = ZScoreFeatureScaler()
    scaled = scaler.scale_features_zscore(container_analysis)

    db = DuckDBStorageManager(path_database=cfg.get_database_path())
    df_sel = db.fetch_records_sql("SELECT id_variable, name_variable, weight FROM analysis_variable_selection")
    anchor_indices = df_sel["id_variable"].tolist()

    if not anchor_indices:
        db.close_connection_database()
        raise RuntimeError("No anchor variables found in database. Run 'variable-detection' first.")
    # end if

    # 1. Correlation analysis
    matrix_pooled = np.vstack([scaled.sample_matrix_x_scaled, scaled.sample_matrix_y_scaled])
    corr_analyzer = CorrelationAnalyzer()
    corr_result = corr_analyzer.compute_matrix_correlation(
        matrix_pooled=matrix_pooled,
        names_variables=scaled.name_features,
        method=cfg.variable_analysis.correlation_method,
        threshold_edge=cfg.variable_analysis.correlation_threshold,
    )
    db.insert_records_correlation(corr_result)

    # 2. Variable clustering
    clusterer = VariableClusterer()
    clust_result = clusterer.cluster_variables_relationship(
        matrix_rel=corr_result.matrix_correlation,
        list_selected_anchors=anchor_indices,
        names_variables=scaled.name_features,
        num_clusters=cfg.variable_analysis.num_clusters,
    )
    db.insert_records_clustering(clust_result)

    # 3. Prototype extraction
    extractor = PrototypeExtractor()
    proto_hat_s = extractor.extract_samples_prototype(
        sample_x=scaled.sample_matrix_x_scaled,
        sample_y=scaled.sample_matrix_y_scaled,
        names_features=scaled.name_features,
        list_subspace_indices=anchor_indices,
        type_subspace="hat_S",
        top_n=5,
    )
    proto_s_tilde = extractor.extract_samples_prototype(
        sample_x=scaled.sample_matrix_x_scaled,
        sample_y=scaled.sample_matrix_y_scaled,
        names_features=scaled.name_features,
        list_subspace_indices=clust_result.indices_augmented_s_tilde,
        type_subspace="hat_S_augmented",
        top_n=5,
    )
    combined_prototypes = PrototypeSampleResult(
        list_prototype_records=proto_hat_s.list_prototype_records + proto_s_tilde.list_prototype_records
    )
    db.insert_records_prototypes(combined_prototypes)

    # Update metadata in DuckDB
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_analysis_x', ?)",
        [str(container_analysis.sample_matrix_x.shape[0])]
    )
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_analysis_y', ?)",
        [str(container_analysis.sample_matrix_y.shape[0])]
    )
    is_sub_analysis = (limit_records is not None and limit_records > 0 and (container_analysis.sample_matrix_x.shape[0] < container.sample_matrix_x.shape[0]))
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('scope_variable_analysis', ?)",
        ["subset" if is_sub_analysis else "whole"]
    )
    db.close_connection_database()

    typer.echo(
        f"✓ Variable analysis complete ({analysis_scope} scope, {container_analysis.sample_matrix_x.shape[0] + container_analysis.sample_matrix_y.shape[0]} samples). "
        f"Found {len(clust_result.dict_cluster_to_variables)} variable clusters. "
        f"Augmented feature set (S_tilde) contains {len(clust_result.indices_augmented_s_tilde)} variables."
    )

    copy_config_file(config, cfg)
    # end def cmd_variable_analysis


def _get_dataset_report_generator(cfg: PipelineCliConfig) -> BaseDatasetReportGenerator:
    """Instantiates the dataset-specific report generator based on configuration."""
    dataset_name = cfg.project.dataset_name.lower().strip()
    if dataset_name == "ames_housing":
        ames_cfg = AmesPreprocessingConfig(
            path_data_file=cfg.dataset.ames_housing.raw_data_path,
            max_records_per_distribution=cfg.dataset.ames_housing.max_records_per_distribution,
            random_seed_sampling=cfg.dataset.ames_housing.random_seed_sampling,
        )
        return AmesHousingReportGenerator(config=ames_cfg)
    elif dataset_name == "speed_dating":
        sd_cfg = SpeedDatingPreprocessingConfig(
            path_data_file=cfg.dataset.speed_dating.raw_data_path,
            max_records_per_distribution=cfg.dataset.speed_dating.max_records_per_distribution,
            random_seed_sampling=cfg.dataset.speed_dating.random_seed_sampling,
        )
        return SpeedDatingReportGenerator(config=sd_cfg)
    else:
        raise ValueError(f"Dataset '{dataset_name}' does not have a dataset report generator implemented.")
    # end if

    copy_config_file(config, cfg)
    # end def _get_dataset_report_generator


@app.command("generate-report")
def cmd_generate_report(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Generate visual plots, multi-sheet Excel workbooks, and executive reports (dataset-specific and analysis pipeline)."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)
    output_dir = cfg.project.output_directory
    os.makedirs(output_dir, exist_ok=True)

    db = DuckDBStorageManager(path_database=cfg.get_database_path())

    # Check that analytical records exist
    df_sel = db.fetch_records_sql("SELECT * FROM analysis_variable_selection")
    df_clust = db.fetch_records_sql("SELECT * FROM analysis_variable_clustering")
    df_corr = db.fetch_records_sql("SELECT * FROM analysis_variable_correlation")
    df_proto = db.fetch_records_sql("SELECT * FROM analysis_representative_samples")

    if df_sel.empty or df_clust.empty:
        db.close_connection_database()
        raise RuntimeError("Incomplete analysis tables in database. Run 'variable-detection' and 'variable-analysis' first.")
    # end if

    # Reconstruct data objects for artifact plotter
    npz_path = cfg.get_features_container_path()
    container = TwoSampleDataContainer.load_from_npz(npz_path) if os.path.exists(npz_path) else db.fetch_preprocessed_features()

    limit_records = None
    seed = 42
    if cfg.project.dataset_name.lower().strip() == "ames_housing":
        limit_records = cfg.dataset.ames_housing.max_records_per_distribution
        seed = cfg.dataset.ames_housing.random_seed_sampling
    elif cfg.project.dataset_name.lower().strip() == "speed_dating":
        limit_records = cfg.dataset.speed_dating.max_records_per_distribution
        seed = cfg.dataset.speed_dating.random_seed_sampling
    # end if

    # 1. Resolve analysis container (for correlation matrix & clustering)
    analysis_scope = cfg.variable_analysis.sample_scope.lower().strip()
    if "sub" in analysis_scope and limit_records is not None and limit_records > 0:
        container_analysis = container.create_subsample_container(
            max_records_per_distribution=limit_records,
            random_seed=seed,
        )
    else:
        container_analysis = container
    # end if

    scaler = ZScoreFeatureScaler()
    scaled_analysis = scaler.scale_features_zscore(container_analysis)

    matrix_pooled = np.vstack([scaled_analysis.sample_matrix_x_scaled, scaled_analysis.sample_matrix_y_scaled])
    corr_analyzer = CorrelationAnalyzer()
    corr_result = corr_analyzer.compute_matrix_correlation(
        matrix_pooled=matrix_pooled,
        names_variables=scaled_analysis.name_features,
        method=cfg.variable_analysis.correlation_method,
        threshold_edge=cfg.variable_analysis.correlation_threshold,
    )

    clusterer = VariableClusterer()
    clust_result = clusterer.cluster_variables_relationship(
        matrix_rel=corr_result.matrix_correlation,
        list_selected_anchors=df_sel["id_variable"].tolist(),
        names_variables=scaled_analysis.name_features,
        num_clusters=cfg.variable_analysis.num_clusters,
    )

    # 2. Resolve report container (for marginal distributions, prototype exemplars, persona radar charts)
    report_scope = cfg.report.sample_scope.lower().strip()
    if "sub" in report_scope and limit_records is not None and limit_records > 0:
        container_report = container.create_subsample_container(
            max_records_per_distribution=limit_records,
            random_seed=seed,
        )
    else:
        container_report = container
    # end if

    scaled_report = scaler.scale_features_zscore(container_report)

    sel_result = VariableSelectionResult(
        indices_selected=df_sel["id_variable"].tolist(),
        names_selected=df_sel["name_variable"].tolist(),
        weights_selected=df_sel["weight"].tolist(),
    )

    extractor = PrototypeExtractor()
    proto_hat_s = extractor.extract_samples_prototype(
        sample_x=scaled_report.sample_matrix_x_scaled,
        sample_y=scaled_report.sample_matrix_y_scaled,
        names_features=scaled_report.name_features,
        list_subspace_indices=sel_result.indices_selected,
        type_subspace="hat_S",
        top_n=5,
    )
    proto_s_tilde = extractor.extract_samples_prototype(
        sample_x=scaled_report.sample_matrix_x_scaled,
        sample_y=scaled_report.sample_matrix_y_scaled,
        names_features=scaled_report.name_features,
        list_subspace_indices=clust_result.indices_augmented_s_tilde,
        type_subspace="hat_S_augmented",
        top_n=5,
    )
    combined_prototypes = PrototypeSampleResult(
        list_prototype_records=proto_hat_s.list_prototype_records + proto_s_tilde.list_prototype_records
    )

    # Update prototype records and report scope in database
    db.insert_records_prototypes(combined_prototypes)
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_report_x', ?)",
        [str(container_report.sample_matrix_x.shape[0])]
    )
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('n_samples_report_y', ?)",
        [str(container_report.sample_matrix_y.shape[0])]
    )
    is_sub_report = (limit_records is not None and limit_records > 0 and (container_report.sample_matrix_x.shape[0] < container.sample_matrix_x.shape[0]))
    db.connection_db.execute(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES ('scope_report', ?)",
        ["subset" if is_sub_report else "whole"]
    )

    dict_artifacts: ty.Dict[str, ty.Any] = {}
    if cfg.report.export_plots:
        exporter = ArtifactExporter()
        dict_artifacts = exporter.export_all_artifacts(
            db_manager=db,
            correlation_result=corr_result,
            clustering_result=clust_result,
            selection_result=sel_result,
            prototype_result=combined_prototypes,
            directory_output=output_dir,
            container=container_report,
        )
        typer.echo(f"✓ Generated visual network, tornado, and radar charts in: {output_dir}")
    # end if

    # 1. Dataset-specific Report (Exploratory & Shallow Statistics)
    path_dataset_md: ty.Optional[str] = None
    if cfg.report.export_dataset_report:
        try:
            generator = _get_dataset_report_generator(cfg)
            dataset_title = cfg.report.dataset_report_title or f"{cfg.project.dataset_name.replace('_', ' ').title()} Dataset Exploratory Report"
            if cfg.project.dataset_name == "ames_housing":
                raw_path = cfg.dataset.ames_housing.raw_data_path
            elif cfg.project.dataset_name == "speed_dating":
                raw_path = cfg.dataset.speed_dating.raw_data_path
            else:
                raw_path = None
            # end if
            artifacts_dataset = generator.generate_dataset_report(
                directory_output=output_dir,
                path_raw_data=raw_path,
                title_report=dataset_title,
                export_excel=cfg.report.export_excel,
            )
            path_dataset_md = artifacts_dataset.path_report_markdown
            typer.echo(f"✓ Generated dataset-specific Markdown report at: {path_dataset_md}")
            if artifacts_dataset.path_report_excel:
                typer.echo(f"✓ Generated dataset-specific Excel workbook at: {artifacts_dataset.path_report_excel}")
            # end if
        except Exception as err:
            logger.warning(f"Could not generate dataset-specific report: {err}")
            typer.echo(f"⚠ Skipping dataset report: {err}")
        # end try
    # end if

    # 2. Analysis Pipeline Report (Statistical Engine, MMD, Clusters, Prototypes)
    synthesizer = ReportSynthesizer()

    if cfg.report.export_excel:
        path_excel = os.path.join(output_dir, "analysis_report.xlsx")
        synthesizer.generate_excel_workbook(db_manager=db, path_output_excel=path_excel)
        typer.echo(f"✓ Generated multi-sheet Excel workbook at: {path_excel}")
    # end if

    if cfg.report.export_markdown:
        path_md = os.path.join(output_dir, "report.md")
        synthesizer.generate_markdown_report(
            db_manager=db,
            path_output_markdown=path_md,
            title_report=cfg.report.report_title,
            dict_artifacts=dict_artifacts,
            path_dataset_report=path_dataset_md,
        )
        typer.echo(f"✓ Generated executive Markdown report at: {path_md}")
    # end if

    db.close_connection_database()

    copy_config_file(config, cfg)
    # end def cmd_generate_report


@app.command("generate-dataset-report")
def cmd_generate_dataset_report(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Generate dataset-specific exploratory report with shallow-level statistics."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)
    output_dir = cfg.project.output_directory
    os.makedirs(output_dir, exist_ok=True)

    generator = _get_dataset_report_generator(cfg)
    dataset_title = cfg.report.dataset_report_title or f"{cfg.project.dataset_name.replace('_', ' ').title()} Dataset Exploratory Report"
    if cfg.project.dataset_name == "ames_housing":
        raw_path = cfg.dataset.ames_housing.raw_data_path
    elif cfg.project.dataset_name == "speed_dating":
        raw_path = cfg.dataset.speed_dating.raw_data_path
    else:
        raw_path = None
    # end if
    artifacts_dataset = generator.generate_dataset_report(
        directory_output=output_dir,
        path_raw_data=raw_path,
        title_report=dataset_title,
        export_excel=cfg.report.export_excel,
    )
    typer.echo(f"✓ Generated dataset-specific Markdown report at: {artifacts_dataset.path_report_markdown}")
    if artifacts_dataset.path_report_excel:
        typer.echo(f"✓ Generated dataset-specific Excel workbook at: {artifacts_dataset.path_report_excel}")
    # end if

    copy_config_file(config, cfg)
    # end def cmd_generate_dataset_report


@app.command("run-all")
def cmd_run_all(
    config: str = typer.Argument(help="Path to TOML configuration file.")
) -> None:
    """Execute all pipeline stages sequentially: setup, preprocess, variable-detection, variable-analysis, generate-report."""
    typer.echo("=== [Step 1/5] Setup ===")
    cmd_setup(config=config)

    typer.echo("\n=== [Step 2/5] Preprocess ===")
    cmd_preprocess(config=config)

    typer.echo("\n=== [Step 3/5] Variable Detection ===")
    cmd_variable_detection(config=config)

    typer.echo("\n=== [Step 4/5] Variable Analysis ===")
    cmd_variable_analysis(config=config)

    typer.echo("\n=== [Step 5/5] Generate Report ===")
    cmd_generate_report(config=config)

    typer.echo("\n🎉 Pipeline complete! All deliverables generated successfully.")
    # end def cmd_run_all


def main():
    app()


if __name__ == "__main__":
    main()
