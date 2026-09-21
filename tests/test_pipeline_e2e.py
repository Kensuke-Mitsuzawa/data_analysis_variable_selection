import os
import tempfile
from data_analysis_variable_selection.pipeline.runner import PipelineOrchestrator
from data_analysis_variable_selection.datasets.ames_housing.preprocessor import AmesHousingPreprocessor
from data_analysis_variable_selection.database.manager import DuckDBStorageManager


def test_pipeline_orchestrator_e2e():
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor = AmesHousingPreprocessor()
        orchestrator = PipelineOrchestrator()

        res = orchestrator.run_pipeline_analysis(
            preprocessor=preprocessor,
            path_output_directory=temp_dir,
            max_mmd_epochs=5,
            top_k_anchors=3,
            num_clusters=3,
            correlation_method="pearson",
            correlation_threshold=0.2
        )

        assert "path_database" in res
        assert os.path.exists(res["path_database"])
        assert len(res["anchors_selected"]) > 0

        # Verify DuckDB persistence
        db = DuckDBStorageManager(res["path_database"])
        df_sel = db.fetch_records_sql("SELECT * FROM analysis_variable_selection")
        df_corr = db.fetch_records_sql("SELECT * FROM analysis_variable_correlation")
        df_clust = db.fetch_records_sql("SELECT * FROM analysis_variable_clustering")
        df_proto = db.fetch_records_sql("SELECT * FROM analysis_representative_samples")

        assert len(df_sel) > 0
        assert len(df_clust) > 0
        assert len(df_proto) > 0
        db.close_connection_database()

        # Verify visual and tabular deliverables
        artifacts = res["artifacts"]
        assert os.path.exists(artifacts["cluster_analysis_csv"])
        assert os.path.exists(artifacts["representative_samples_csv"])
        assert os.path.exists(artifacts["constellation_network"])
        assert len(artifacts["tornado_charts"]) > 0
        assert all(os.path.exists(p) for p in artifacts["tornado_charts"])
