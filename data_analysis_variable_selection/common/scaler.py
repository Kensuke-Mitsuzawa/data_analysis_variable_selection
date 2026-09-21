import typing as ty
import numpy as np
from .models import TwoSampleDataContainer, ScaledDataContainer


class ZScoreFeatureScaler:
    """Standardizes feature matrices across two distributions using pooled statistics.
    """

    def __init__(self, epsilon_std: float = 1e-8):
        """Initializes the scaler.

        Args:
            epsilon_std: Small value to prevent division by zero for invariant features.
        """
        self.epsilon_std = epsilon_std
        # end def __init__

    def scale_features_zscore(self, data_container: TwoSampleDataContainer) -> ScaledDataContainer:
        """Standardizes matrices X and Y based on mean and standard deviation of pooled Z = X union Y.

        Args:
            data_container: TwoSampleDataContainer holding sample matrices X and Y.

        Returns:
            ScaledDataContainer holding standardized matrices, scaling parameters, and feature names.
        """
        matrix_x = data_container.sample_matrix_x
        matrix_y = data_container.sample_matrix_y

        matrix_pooled = np.vstack([matrix_x, matrix_y])
        vector_mean = np.mean(matrix_pooled, axis=0)
        vector_std = np.std(matrix_pooled, axis=0)

        # Avoid zero division
        vector_std_adjusted = np.where(vector_std < self.epsilon_std, 1.0, vector_std)

        matrix_x_scaled = (matrix_x - vector_mean) / vector_std_adjusted
        matrix_y_scaled = (matrix_y - vector_mean) / vector_std_adjusted

        return ScaledDataContainer(
            sample_matrix_x_scaled=matrix_x_scaled,
            sample_matrix_y_scaled=matrix_y_scaled,
            vector_mean=vector_mean,
            vector_std=vector_std,
            name_features=data_container.name_features,
        )
        # end def scale_features_zscore

    def convert_matrix_to_structured_array(
        self,
        matrix_data: np.ndarray,
        names_features: ty.List[str]
    ) -> np.ndarray:
        """Converts a 2D numeric matrix into a NumPy structured array with named fields.

        Args:
            matrix_data: 2D numpy array of shape (N, d).
            names_features: List of column names matching the d features.

        Returns:
            1D numpy structured array with field names matching names_features.
        """
        dtype_fields = [(name, np.float64) for name in names_features]
        array_structured = np.empty(matrix_data.shape[0], dtype=dtype_fields)

        for idx_col, name_col in enumerate(names_features):
            array_structured[name_col] = matrix_data[:, idx_col]
        # end for idx_col

        return array_structured
        # end def convert_matrix_to_structured_array
# end class ZScoreFeatureScaler
