import numpy as np
from data_analysis_variable_selection.common.models import TwoSampleDataContainer
from data_analysis_variable_selection.common.scaler import ZScoreFeatureScaler


def test_zscore_feature_scaler():
    np.random.seed(42)
    x = np.random.normal(loc=10.0, scale=2.0, size=(50, 4))
    y = np.random.normal(loc=15.0, scale=3.0, size=(60, 4))
    names = ["feat_0", "feat_1", "feat_2", "feat_3"]

    container = TwoSampleDataContainer(
        sample_matrix_x=x,
        sample_matrix_y=y,
        name_features=names
    )

    scaler = ZScoreFeatureScaler()
    scaled_res = scaler.scale_features_zscore(container)

    # Check shapes
    assert scaled_res.sample_matrix_x_scaled.shape == (50, 4)
    assert scaled_res.sample_matrix_y_scaled.shape == (60, 4)

    # Pooled mean should be near 0 and std near 1
    pooled = np.vstack([scaled_res.sample_matrix_x_scaled, scaled_res.sample_matrix_y_scaled])
    np.testing.assert_allclose(np.mean(pooled, axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(pooled, axis=0), 1.0, atol=1e-6)

    # Test structured array conversion
    arr_struct = scaler.convert_matrix_to_structured_array(scaled_res.sample_matrix_x_scaled, names)
    assert arr_struct.shape == (50,)
    assert arr_struct.dtype.names == tuple(names)
