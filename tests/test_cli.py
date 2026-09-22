import os
import logging
import tempfile
import pytest
from typer.testing import CliRunner

from data_analysis_variable_selection.cli.main import app
from data_analysis_variable_selection.database.manager import DuckDBStorageManager

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_logging_handlers():
    yield
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
            # end try
        # end if
    # end for
# end def clean_logging_handlers


def test_cli_step_by_step_mmd():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.toml")
        output_dir = os.path.join(tmpdir, "output")
        db_file = "warehouse.duckdb"

        toml_content = f"""
[project]
output_directory = "{output_dir}"
database_file = "{db_file}"
dataset_name = "ames_housing"

[dataset.ames_housing]
max_records_per_distribution = 20
random_seed_sampling = 42

[variable_detection]
method = "mmd"

[variable_detection.mmd]
algorithm = "algorithm_one"
device = "cpu"
max_epochs = 5
top_k_fallback = 3

[variable_analysis]
correlation_method = "pearson"
correlation_threshold = 0.2
num_clusters = 3

[report]
report_title = "Test Ames Analysis Report"
export_excel = true
export_plots = true
export_markdown = true
"""
        with open(config_path, "w") as f_toml:
            f_toml.write(toml_content)
        # end with

        # 1. Test setup
        res_setup = runner.invoke(app, ["setup", "--config", config_path])
        assert res_setup.exit_code == 0, f"Setup failed: {res_setup.stdout}"

        # 2. Test preprocess
        res_prep = runner.invoke(app, ["preprocess", "--config", config_path])
        assert res_prep.exit_code == 0, f"Preprocess failed: {res_prep.stdout}"

        # Verify DuckDB table and .npz
        db_path = os.path.join(output_dir, db_file)
        assert os.path.exists(db_path)
        npz_path = os.path.join(output_dir, "preprocessed_features.npz")
        assert os.path.exists(npz_path)

        db = DuckDBStorageManager(db_path)
        df_feat = db.fetch_records_sql("SELECT * FROM dataset_features_preprocessed")
        assert len(df_feat) > 0
        assert "sample_id" in df_feat.columns
        assert "distribution_label" in df_feat.columns
        db.close_connection_database()

        # 3. Test variable-detection
        res_det = runner.invoke(app, ["variable-detection", "--config", config_path])
        assert res_det.exit_code == 0, f"Variable-detection failed: {res_det.stdout}"

        db = DuckDBStorageManager(db_path)
        df_sel = db.fetch_records_sql("SELECT * FROM analysis_variable_selection")
        assert len(df_sel) > 0
        db.close_connection_database()

        # 4. Test variable-analysis
        res_ana = runner.invoke(app, ["variable-analysis", "--config", config_path])
        assert res_ana.exit_code == 0, f"Variable-analysis failed: {res_ana.stdout}"

        db = DuckDBStorageManager(db_path)
        df_clust = db.fetch_records_sql("SELECT * FROM analysis_variable_clustering")
        assert len(df_clust) > 0
        db.close_connection_database()

        # 5. Test generate-report
        res_rep = runner.invoke(app, ["generate-report", "--config", config_path])
        assert res_rep.exit_code == 0, f"Generate-report failed: {res_rep.stdout}"

        # Check excel, markdown, and log deliverables
        path_excel = os.path.join(output_dir, "analysis_report.xlsx")
        path_md = os.path.join(output_dir, "report.md")
        path_dataset_md = os.path.join(output_dir, "dataset_report.md")
        path_dataset_excel = os.path.join(output_dir, "dataset_report.xlsx")
        path_log = os.path.join(output_dir, "log", "pipeline_run.log")
        assert os.path.exists(path_excel)
        assert os.path.exists(path_md)
        assert os.path.exists(path_dataset_md)
        assert os.path.exists(path_dataset_excel)
        assert os.path.exists(path_log)
        with open(path_log, "r", encoding="utf-8") as f_log:
            log_text = f_log.read()
            assert len(log_text) > 0
        # end with

        with open(path_md, "r") as f_md:
            md_content = f_md.read()
            assert "# Test Ames Analysis Report" in md_content
            assert "Discovered Anchor Variables" in md_content
            assert "dataset_report.md" in md_content
        # end with

        with open(path_dataset_md, "r") as f_ds_md:
            ds_content = f_ds_md.read()
            assert "Sample Partitioning & Temporal Split" in ds_content
        # end with


def test_cli_generate_dataset_report_standalone():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config_ds.toml")
        output_dir = os.path.join(tmpdir, "output_ds")

        toml_content = f"""
[project]
output_directory = "{output_dir}"
dataset_name = "ames_housing"

[dataset.ames_housing]
max_records_per_distribution = 20

[report]
export_excel = true
"""
        with open(config_path, "w") as f_toml:
            f_toml.write(toml_content)
        # end with

        res_rep = runner.invoke(app, ["generate-dataset-report", "--config", config_path])
        assert res_rep.exit_code == 0, f"Generate-dataset-report failed: {res_rep.stdout}"

        path_dataset_md = os.path.join(output_dir, "dataset_report.md")
        path_dataset_excel = os.path.join(output_dir, "dataset_report.xlsx")
        assert os.path.exists(path_dataset_md)
        assert os.path.exists(path_dataset_excel)

        with open(path_dataset_md, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Ames Housing" in content
        # end with
    # end with


def test_cli_run_all_wasserstein():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_wasserstein.toml")
        output_dir = os.path.join(tmpdir, "output_wass")
        db_file = "warehouse_wass.duckdb"

        toml_content = f"""
[project]
output_directory = "{output_dir}"
database_file = "{db_file}"
dataset_name = "ames_housing"

[dataset.ames_housing]
max_records_per_distribution = 20
random_seed_sampling = 42

[variable_detection]
method = "wasserstein"

[variable_detection.wasserstein]
distributed_backend = "single"
variable_detection_approach = "hist_based"
top_k_fallback = 3

[variable_analysis]
correlation_method = "pearson"
correlation_threshold = 0.2
num_clusters = 3

[report]
report_title = "Wasserstein Ames Analysis Report"
export_excel = true
export_plots = true
export_markdown = true
"""
        with open(config_path, "w") as f_toml:
            f_toml.write(toml_content)
        # end with

        # Test run-all
        res_run_all = runner.invoke(app, ["run-all", "--config", config_path])
        assert res_run_all.exit_code == 0, f"run-all failed: {res_run_all.stdout}"

        path_excel = os.path.join(output_dir, "analysis_report.xlsx")
        path_md = os.path.join(output_dir, "report.md")
        path_db = os.path.join(output_dir, db_file)

        assert os.path.exists(path_db)
        assert os.path.exists(path_excel)
        assert os.path.exists(path_md)
    # end with
# end def test_cli_run_all_wasserstein


def test_cli_relative_output_directory_validation():
    """Verifies that relative output_directory raises an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "relative_config.toml")
        with open(config_path, "w") as f_toml:
            f_toml.write("""
[project]
output_directory = "relative/path/to/output"
database_file = "test.duckdb"
dataset_name = "ames_housing"
""")
        # end with

        res = runner.invoke(app, ["setup", "--config", config_path])
        assert res.exit_code != 0
        assert "output_directory" in str(res.exception) or "output_directory" in res.stdout
    # end with
# end def test_cli_relative_output_directory_validation
