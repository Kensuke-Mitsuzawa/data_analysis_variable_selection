import abc
import typing as ty
from .models import TwoSampleDataContainer


class BaseDatasetPreprocessor(abc.ABC):
    """Abstract base class for dataset-specific ingestion, transformation, and two-sample partitioning.
    """

    @abc.abstractmethod
    def prepare_two_sample_data(
        self,
        path_data: ty.Optional[str] = None,
        **kwargs: ty.Any
    ) -> TwoSampleDataContainer:
        """Loads and processes the raw dataset into a two-sample data container.

        Args:
            path_data: Optional file path or identifier for the dataset.
            **kwargs: Additional dataset-specific preprocessing or sampling arguments.

        Returns:
            TwoSampleDataContainer holding matrices X, Y, and feature names.
        """
        raise NotImplementedError()
        # end def prepare_two_sample_data

    def track_feature_operations(
        self,
        container: ty.Optional[TwoSampleDataContainer] = None,
        include_removed: bool = True
    ) -> ty.List[ty.Any]:
        """Tracks and returns the lineage, source columns, and data types of all processed features.

        Args:
            container: Optional pre-computed TwoSampleDataContainer.
            include_removed: Whether to include raw features removed during preprocessing.

        Returns:
            List of FeatureItemData entries.
        """
        raise NotImplementedError()
        # end def track_feature_operations
# end class BaseDatasetPreprocessor

