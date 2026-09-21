import os
import tempfile
import numpy as np
import pytest

from data_analysis_variable_selection.common.base_selector import BaseVariableSelector
from data_analysis_variable_selection.variable_selection.wasserstein.config import WassersteinSelectionConfig
from data_analysis_variable_selection.variable_selection.wasserstein.selector import WassersteinVariableSelector
from data_analysis_variable_selection.pipeline.runner import PipelineOrchestrator
from data_analysis_variable_selection.datasets.ames_housing.preprocessor import AmesHousingPreprocessor
from data_analysis_variable_selection.datasets.ames_housing.config import AmesPreprocessingConfig


def test_wasserstein_config_defaults():
    config = WassersteinSelectionConfig()
    assert config.distributed_backend == "single"
    assert config.variable_detection_approach == "hist_based"
    assert config.threshold_weights == 0.1
    assert config.top_k_fallback == 5
    assert config.is_normalize_weights is True

    custom_cfg = WassersteinSelectionConfig(
        distributed_backend="single",
        threshold_weights=0.25,
        top_k_fallback=3
    )
    assert custom_cfg.threshold_weights == 0.25
    assert custom_cfg.top_k_fallback == 3


def test_wasserstein_selector_synthetic():
    np.random.seed(42)
    n_samples = 60
    n_features = 5
    names = [f"feat_{i}" for i in range(n_features)]

    # Distribution X is standard normal
    sample_x = np.random.randn(n_samples, n_features)
    # Distribution Y is identical to X except feat_0 and feat_2 have shifted means
    sample_y = np.random.randn(n_samples, n_features)
    sample_y[:, 0] += 3.0  # Large shift in feat_0
    sample_y[:, 2] += 2.0  # Moderate shift in feat_2

    selector = WassersteinVariableSelector()
    assert isinstance(selector, BaseVariableSelector)

    result = selector.select_variables(
        sample_x=sample_x,
        sample_y=sample_y,
        names_variables=names,
    )

    assert result.metadata_selection["algorithm"] == "1d_wasserstein"
    assert len(result.weights_selected) == len(result.indices_selected)
    assert len(result.names_selected) == len(result.indices_selected)

    # All weights must be normalized between 0 and 1
    all_weights = np.array(result.metadata_selection["all_weights"])
    assert np.all(all_weights >= 0.0)
    assert np.all(all_weights <= 1.0 + 1e-6)

    # Shifted feature 0 should have highest weight
    assert np.argmax(all_weights) == 0
    assert 0 in result.indices_selected
    assert "feat_0" in result.names_selected

    # Test alias
    result_alias = selector.select_variables_wasserstein(
        sample_x=sample_x,
        sample_y=sample_y,
        names_variables=names,
    )
    assert result_alias.indices_selected == result.indices_selected


def test_pipeline_with_wasserstein_selector():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AmesPreprocessingConfig(max_records_per_distribution=30)
        preprocessor = AmesHousingPreprocessor(config=config)
        wasserstein_selector = WassersteinVariableSelector()

        orchestrator = PipelineOrchestrator(variable_selector=wasserstein_selector)
        results = orchestrator.run_pipeline_analysis(
            preprocessor=preprocessor,
            path_output_directory=tmpdir,
            top_k_anchors=3,
            correlation_method="pearson",
        )

        assert os.path.exists(results["path_database"])
        assert "selection_result" in results
        assert results["selection_result"].metadata_selection["algorithm"] == "1d_wasserstein"
        assert len(results["selection_result"].indices_selected) > 0
