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
# end class BaseDatasetPreprocessor
