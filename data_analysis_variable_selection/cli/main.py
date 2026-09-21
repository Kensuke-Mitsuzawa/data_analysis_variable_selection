import logging
import os
import sys
import typing as ty
import typer
import numpy as np

from .cli_config import PipelineCliConfig, load_toml_config
from ..datasets.setup_handler import DatasetSetupHandler
from ..datasets.ames_housing.config import AmesPreprocessingConfig
from ..datasets.ames_housing.preprocessor import AmesHousingPreprocessor
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


@app.command("setup")
def cmd_setup(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
) -> None:
    """Setup datasets: download and uncompress raw files as specified in TOML config."""
    cfg = load_toml_config(config)
    configure_pipeline_logging(cfg)
    os.makedirs(cfg.project.output_directory, exist_ok=True)
    handler = DatasetSetupHandler()
    raw_path = handler.setup_dataset(cfg)
    typer.echo(f"✓ Setup complete. Verified raw dataset at: {raw_path}")
    # end def cmd_setup


@app.command("preprocess")
def cmd_preprocess(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
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
    else:
        raise ValueError(f"Dataset '{dataset_name}' not yet supported for preprocessing.")
    # end if

    container = preprocessor.prepare_two_sample_data()

    # 1. Save human-readable table into DuckDB
    db = DuckDBStorageManager(path_database=cfg.get_database_path())
    db.initialize_database_schema()
    db.insert_preprocessed_features(container)
    db.close_connection_database()

    # 2. Save array container to working directory
    npz_path = cfg.get_features_container_path()
    container.save_to_npz(npz_path)

    typer.echo(
        f"✓ Preprocessing complete. X shape: {container.sample_matrix_x.shape}, "
        f"Y shape: {container.sample_matrix_y.shape}, Features: {len(container.name_features)}"
    )
    typer.echo(f"✓ Saved preprocessed human-readable table to DuckDB: {cfg.get_database_path()}")
    typer.echo(f"✓ Cached preprocessed arrays to: {npz_path}")
    # end def cmd_preprocess


@app.command("variable-detection")
def cmd_variable_detection(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
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
    logger.info(f"loading dataset. {container.sample_matrix_x.shape}, {container.sample_matrix_y.shape}, {len(container.name_features)}")

    # Standardize
    logger.info("Standardizing dataset.")
    scaler = ZScoreFeatureScaler()
    scaled = scaler.scale_features_zscore(container)
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
            cv_stability_threshold=cfg.variable_detection.mmd.cv_stability_threshold,
            use_fused_kernel=cfg.variable_detection.mmd.use_fused_kernel,
            is_use_local_dask_cluster=cfg.variable_detection.mmd.is_use_local_dask_cluster,
            dask_scheduler_host=cfg.variable_detection.mmd.dask_scheduler_host,
            dask_scheduler_port=cfg.variable_detection.mmd.dask_scheduler_port,
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
    db.close_connection_database()

    typer.echo(f"✓ Variable detection complete ({method}). Discovered {len(result.indices_selected)} anchor variables:")
    for name_var, weight in zip(result.names_selected, result.weights_selected):
        typer.echo(f"  - {name_var}: weight = {weight:.4f}")
    # end for
    logger.info("Persisting variable detection results to DuckDB. Done.")
    # end def cmd_variable_detection


@app.command("variable-analysis")
def cmd_variable_analysis(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
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

    scaler = ZScoreFeatureScaler()
    scaled = scaler.scale_features_zscore(container)

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
    db.close_connection_database()

    typer.echo(
        f"✓ Variable analysis complete. Found {len(clust_result.dict_cluster_to_variables)} variable clusters. "
        f"Augmented feature set (S_tilde) contains {len(clust_result.indices_augmented_s_tilde)} variables."
    )
    # end def cmd_variable_analysis


@app.command("generate-report")
def cmd_generate_report(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
) -> None:
    """Generate visual plots, multi-sheet Excel workbook, and executive Markdown report."""
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
    scaler = ZScoreFeatureScaler()
    scaled = scaler.scale_features_zscore(container)

    matrix_pooled = np.vstack([scaled.sample_matrix_x_scaled, scaled.sample_matrix_y_scaled])
    corr_analyzer = CorrelationAnalyzer()
    corr_result = corr_analyzer.compute_matrix_correlation(
        matrix_pooled=matrix_pooled,
        names_variables=scaled.name_features,
        method=cfg.variable_analysis.correlation_method,
        threshold_edge=cfg.variable_analysis.correlation_threshold,
    )

    clusterer = VariableClusterer()
    clust_result = clusterer.cluster_variables_relationship(
        matrix_rel=corr_result.matrix_correlation,
        list_selected_anchors=df_sel["id_variable"].tolist(),
        names_variables=scaled.name_features,
        num_clusters=cfg.variable_analysis.num_clusters,
    )

    sel_result = VariableSelectionResult(
        indices_selected=df_sel["id_variable"].tolist(),
        names_selected=df_sel["name_variable"].tolist(),
        weights_selected=df_sel["weight"].tolist(),
    )

    extractor = PrototypeExtractor()
    proto_hat_s = extractor.extract_samples_prototype(
        sample_x=scaled.sample_matrix_x_scaled,
        sample_y=scaled.sample_matrix_y_scaled,
        names_features=scaled.name_features,
        list_subspace_indices=sel_result.indices_selected,
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
        )
        typer.echo(f"✓ Generated visual network, tornado, and radar charts in: {output_dir}")
    # end if

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
        )
        typer.echo(f"✓ Generated executive Markdown report at: {path_md}")
    # end if

    db.close_connection_database()
    # end def cmd_generate_report


@app.command("run-all")
def cmd_run_all(
    config: str = typer.Option("config.toml", "--config", "-c", help="Path to TOML configuration file.")
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
