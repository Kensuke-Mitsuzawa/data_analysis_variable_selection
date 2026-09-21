import numpy as np
import pytest
from data_analysis_variable_selection.variable_selection.mmd.config import MMDSelectionConfig
from data_analysis_variable_selection.variable_selection.mmd.selector import MMDVariableSelector


def test_mmd_config_defaults():
    config = MMDSelectionConfig()
    # Verify recommended defaults
    assert config.algorithm == "algorithm_one"
    assert config.device == "auto"
    assert config.regularizer_l1 is None
    assert config.regularizer_l2 is None
    assert config.max_epochs == 9999
    assert config.trainer_backend == "pure_pytorch"
    assert config.variable_detection_approach == "hist_based"
    assert config.top_k_fallback == 5
    assert config.n_cv_subsampling == 5
    assert config.normalize_algorithm_name() == "algorithm_one"

    # Verify user can explicitly set regularizers
    config_custom = MMDSelectionConfig(regularizer_l1=0.05, regularizer_l2=0.02)
    assert config_custom.regularizer_l1 == 0.05
    assert config_custom.regularizer_l2 == 0.02


def test_mmd_algorithm_aliases():
    cfg_dash = MMDSelectionConfig(algorithm="algorithm-one")
    assert cfg_dash.normalize_algorithm_name() == "algorithm_one"

    cfg_cv = MMDSelectionConfig(algorithm="mmd-cv")
    assert cfg_cv.normalize_algorithm_name() == "mmd_cv"

    cfg_cv_sel = MMDSelectionConfig(algorithm="cv_selection")
    assert cfg_cv_sel.normalize_algorithm_name() == "mmd_cv"

    with pytest.raises(ValueError):
        MMDSelectionConfig(algorithm="invalid_algo").normalize_algorithm_name()


def test_algorithm_one_execution():
    np.random.seed(42)
    d = 4
    n = 30
    x = np.random.randn(n, d)
    y = np.random.randn(n, d)
    y[:, 1] += 2.5  # shift on var 1
    names = ["v0", "v1", "v2", "v3"]

    selector = MMDVariableSelector(
        algorithm="algorithm-one",
        device="cpu",
        max_epochs=8,
        top_k_fallback=2
    )
    result = selector.select_variables_mmd(x, y, names)

    assert len(result.indices_selected) > 0
    assert len(result.names_selected) == len(result.indices_selected)
    assert len(result.weights_selected) == len(result.indices_selected)
    assert result.metadata_selection["algorithm"] == "algorithm_one"
    assert result.metadata_selection["device_resolved"] == "cpu"


def test_mmd_cv_execution():
    np.random.seed(42)
    d = 4
    n = 40
    x = np.random.randn(n, d)
    y = np.random.randn(n, d)
    y[:, 0] += 3.0  # shift on var 0
    names = ["v0", "v1", "v2", "v3"]

    config = MMDSelectionConfig(
        algorithm="mmd_cv",
        device="cpu",
        max_epochs=5,
        n_cv_subsampling=2,
        subsampling_ratio=0.75,
        cv_stability_threshold=0.3,
        top_k_fallback=2
    )
    selector = MMDVariableSelector(config=config)
    result = selector.select_variables_mmd(x, y, names)

    assert len(result.indices_selected) > 0
    assert result.metadata_selection["algorithm"] == "mmd_cv"
    assert result.metadata_selection["device_resolved"] == "cpu"
    assert "selection_frequencies" in result.metadata_selection
    assert len(result.metadata_selection["selection_frequencies"]) == d


def test_mmd_distributed_config_customization():
    config = MMDSelectionConfig(
        random_seed=123,
        is_use_local_dask_cluster=True,
        dask_n_workers=8,
        dask_threads_per_worker=3,
        dask_dashboard_address=":9999",
        dask_memory_limit="2GB",
    )
    selector = MMDVariableSelector(config=config)
    dist_cfg = selector._create_distributed_config(config)

    assert dist_cfg.distributed_mode == "dask"
    assert dist_cfg.is_use_local_dask_cluster is True
    assert dist_cfg.dask_n_workers == 8
    assert dist_cfg.dask_threads_per_worker == 3
    assert dist_cfg.dask_dashboard_address == ":9999"
    assert dist_cfg.dask_memory_limit == "2GB"
    assert config.random_seed == 123


def test_mmd_toml_config_parsing(tmp_path):
    import os
    from data_analysis_variable_selection.cli.cli_config import load_toml_config

    toml_path = os.path.join(tmp_path, "test_distributed.toml")
    toml_content = f"""
[project]
output_directory = "{tmp_path}"
database_file = "test.duckdb"

[variable_detection.mmd]
algorithm = "mmd_cv"
random_seed = 99
is_use_local_dask_cluster = true
dask_n_workers = 6
dask_threads_per_worker = 1
dask_memory_limit = "4GB"
"""
    with open(toml_path, "w") as f:
        f.write(toml_content)

    cfg = load_toml_config(toml_path)
    mmd_cfg = cfg.variable_detection.mmd

    assert mmd_cfg.random_seed == 99
    assert mmd_cfg.is_use_local_dask_cluster is True
    assert mmd_cfg.dask_n_workers == 6
    assert mmd_cfg.dask_threads_per_worker == 1
    assert mmd_cfg.dask_memory_limit == "4GB"

