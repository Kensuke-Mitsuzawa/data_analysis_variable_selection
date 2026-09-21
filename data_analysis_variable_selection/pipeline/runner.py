import os
import logging
import typing as ty
import numpy as np

from ..common.base_preprocessor import BaseDatasetPreprocessor
from ..common.models import (
    PrototypeSampleResult,
)
from ..common.scaler import ZScoreFeatureScaler
from ..variable_selection.mmd.selector import MMDVariableSelector
from ..variable_selection.mmd.config import MMDSelectionConfig
from ..correlation.analyzer import CorrelationAnalyzer
from ..clustering.clusterer import VariableClusterer
from ..common.base_selector import BaseVariableSelector
from ..prototype.extractor import PrototypeExtractor
from ..database.manager import DuckDBStorageManager
from ..export.artifact_exporter import ArtifactExporter

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates end-to-end data transformation, variable selection, clustering, persistence, and artifact generation.
    """

    def __init__(
        self,
        default_mmd_config: ty.Optional[MMDSelectionConfig] = None,
        variable_selector: ty.Optional[BaseVariableSelector] = None,
    ):
        """Initializes the orchestrator with common pipeline components.

        Args:
            default_mmd_config: Optional default MMDSelectionConfig.
            variable_selector: Optional custom BaseVariableSelector (defaults to MMDVariableSelector).
        """
        self.scaler = ZScoreFeatureScaler()
        self.variable_selector = variable_selector or MMDVariableSelector(config=default_mmd_config)
        self.mmd_selector = self.variable_selector  # Backwards compatibility alias
        self.correlation_analyzer = CorrelationAnalyzer()
        self.variable_clusterer = VariableClusterer()
        self.prototype_extractor = PrototypeExtractor()
        self.artifact_exporter = ArtifactExporter()
        # end def __init__

    def run_pipeline_analysis(
        self,
        preprocessor: BaseDatasetPreprocessor,
        path_output_directory: str,
        path_raw_data: ty.Optional[str] = None,
        variable_selector: ty.Optional[BaseVariableSelector] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        max_mmd_epochs: int = 30,
        top_k_anchors: int = 5,
        num_clusters: int = 5,
        correlation_method: str = "graphical_lasso",
        correlation_threshold: float = 0.3,
        mmd_config: ty.Optional[MMDSelectionConfig] = None,
        mmd_algorithm: str = "algorithm_one",
        mmd_device: str = "auto"
    ) -> ty.Dict[str, ty.Any]:
        """Executes the complete analysis pipeline from raw data to deliverables.

        Args:
            preprocessor: Dataset adapter implementing BaseDatasetPreprocessor.
            path_output_directory: Target folder to store DuckDB database and deliverables.
            path_raw_data: Optional path to raw dataset file.
            max_records_per_distribution: Optional maximum records per sample distribution to limit O(N^2) complexity.
            max_mmd_epochs: Number of epochs to optimize ARD kernel weights.
            top_k_anchors: Target number of anchor variables (hat_S).
            num_clusters: Target number of clusters.
            correlation_method: 'graphical_lasso' or 'pearson'.
            correlation_threshold: Cutoff score for network edges.
            mmd_config: Optional full MMDSelectionConfig.
            mmd_algorithm: MMD algorithm choice ('algorithm_one' or 'mmd_cv').
            mmd_device: Accelerator device ('auto', 'cpu', 'cuda').

        Returns:
            Dictionary summarizing execution results and artifact paths.
        """
        os.makedirs(path_output_directory, exist_ok=True)
        path_db = os.path.join(path_output_directory, "analysis_warehouse.duckdb")

        logger.info("Step 1: Ingesting and transforming raw dataset...")
        data_container = preprocessor.prepare_two_sample_data(
            path_data=path_raw_data,
            max_records_per_distribution=max_records_per_distribution
        )
        logger.info(
            f"Dataset prepared. X shape: {data_container.sample_matrix_x.shape}, "
            f"Y shape: {data_container.sample_matrix_y.shape}, "
            f"Features: {len(data_container.name_features)}"
        )

        logger.info("Step 2: Performing Z-score standardization across pooled features...")
        scaled_container = self.scaler.scale_features_zscore(data_container)

        logger.info("Step 3: Running variable selection...")
        selector = variable_selector or self.variable_selector
        mmd_kwargs = {
            "max_epochs": max_mmd_epochs,
            "top_k_fallback": top_k_anchors,
            "algorithm": mmd_algorithm,
            "device": mmd_device,
        }
        if mmd_config is not None:
            selection_result = selector.select_variables(
                sample_x=scaled_container.sample_matrix_x_scaled,
                sample_y=scaled_container.sample_matrix_y_scaled,
                names_variables=scaled_container.name_features,
                **mmd_config.model_dump(),
            )
        else:
            selection_result = selector.select_variables(
                sample_x=scaled_container.sample_matrix_x_scaled,
                sample_y=scaled_container.sample_matrix_y_scaled,
                names_variables=scaled_container.name_features,
                **mmd_kwargs,
            )
        # end if
        logger.info(f"Anchor variables selected (hat_S): {selection_result.names_selected}")

        logger.info("Step 4: Analyzing pairwise relationships and precision matrix...")
        matrix_pooled = np.vstack([
            scaled_container.sample_matrix_x_scaled,
            scaled_container.sample_matrix_y_scaled
        ])
        correlation_result = self.correlation_analyzer.compute_matrix_correlation(
            matrix_pooled=matrix_pooled,
            names_variables=scaled_container.name_features,
            method=correlation_method,
            threshold_edge=correlation_threshold,
        )
        logger.info(f"Correlation graph generated with {len(correlation_result.list_edges)} edges.")

        logger.info("Step 5: Clustering variables and identifying augmented variable sets (S_tilde)...")
        clustering_result = self.variable_clusterer.cluster_variables_relationship(
            matrix_rel=correlation_result.matrix_correlation,
            list_selected_anchors=selection_result.indices_selected,
            names_variables=scaled_container.name_features,
            num_clusters=num_clusters,
        )
        logger.info(f"Augmented variable set (S_tilde) contains {len(clustering_result.indices_augmented_s_tilde)} features.")

        logger.info("Step 6: Extracting prototypical exemplar samples for X and Y...")
        # 6a. Prototypes with hat_S
        proto_hat_s = self.prototype_extractor.extract_samples_prototype(
            sample_x=scaled_container.sample_matrix_x_scaled,
            sample_y=scaled_container.sample_matrix_y_scaled,
            names_features=scaled_container.name_features,
            list_subspace_indices=selection_result.indices_selected,
            type_subspace="hat_S",
            top_n=5,
        )

        # 6b. Prototypes with S_tilde
        proto_s_tilde = self.prototype_extractor.extract_samples_prototype(
            sample_x=scaled_container.sample_matrix_x_scaled,
            sample_y=scaled_container.sample_matrix_y_scaled,
            names_features=scaled_container.name_features,
            list_subspace_indices=clustering_result.indices_augmented_s_tilde,
            type_subspace="hat_S_augmented",
            top_n=5,
        )

        combined_prototypes = PrototypeSampleResult(
            list_prototype_records=proto_hat_s.list_prototype_records + proto_s_tilde.list_prototype_records
        )

        logger.info("Step 7: Persisting analytical results to DuckDB warehouse...")
        db_manager = DuckDBStorageManager(path_database=path_db)
        db_manager.initialize_database_schema()
        db_manager.insert_records_selection(selection_result)
        db_manager.insert_records_correlation(correlation_result)
        db_manager.insert_records_clustering(clustering_result)
        db_manager.insert_records_prototypes(combined_prototypes)

        logger.info("Step 8: Exporting tabular and visual deliverables...")
        dict_artifacts = self.artifact_exporter.export_all_artifacts(
            db_manager=db_manager,
            correlation_result=correlation_result,
            clustering_result=clustering_result,
            selection_result=selection_result,
            prototype_result=combined_prototypes,
            directory_output=path_output_directory,
        )
        db_manager.close_connection_database()

        return {
            "path_database": path_db,
            "anchors_selected": selection_result.names_selected,
            "augmented_features": clustering_result.names_augmented_s_tilde,
            "selection_result": selection_result,
            "artifacts": dict_artifacts,
        }
        # end def run_pipeline_analysis
# end class PipelineOrchestrator
