import logging
import typing as ty
import numpy as np
import pandas as pd

from ...common.base_preprocessor import BaseDatasetPreprocessor
from ...common.models import TwoSampleDataContainer
from .config import SpeedDatingPreprocessingConfig
from .loader import SpeedDatingDataLoader
from .cleaner import SpeedDatingDataCleaner
from .pair_builder import SpeedDatingPairBuilder
from .encoder import SpeedDatingFeatureEncoder
from .splitter import SpeedDatingLabelSplitter

logger = logging.getLogger(__name__)


class SpeedDatingPreprocessor(BaseDatasetPreprocessor):
    """Coordinates the end-to-end transformation of raw speed dating surveys into a standardized two-sample container.
    """

    def __init__(
        self,
        config: ty.Optional[SpeedDatingPreprocessingConfig] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        random_seed_sampling: ty.Optional[int] = None
    ):
        """Initializes the preprocessor.

        Args:
            config: Optional SpeedDatingPreprocessingConfig.
            max_records_per_distribution: Optional maximum sample limit per distribution to control O(N^2) complexity.
            random_seed_sampling: Optional random seed for reproducible sampling.
        """
        self.config = config or SpeedDatingPreprocessingConfig()
        if max_records_per_distribution is not None:
            self.config.max_records_per_distribution = max_records_per_distribution
        # end if
        if random_seed_sampling is not None:
            self.config.random_seed_sampling = random_seed_sampling
        # end if

        self.loader = SpeedDatingDataLoader()
        self.cleaner = SpeedDatingDataCleaner()
        self.pair_builder = SpeedDatingPairBuilder()
        self.encoder = SpeedDatingFeatureEncoder()
        self.splitter = SpeedDatingLabelSplitter()
        self._feature_operations: ty.Optional[ty.List[ty.Any]] = None
        # end def __init__

    def prepare_two_sample_data(
        self,
        path_data: ty.Optional[str] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        apply_subsampling: bool = True,
        **kwargs: ty.Any
    ) -> TwoSampleDataContainer:
        """Executes ingestion, cleaning, reciprocal pair formation, joint feature encoding, and splitting.

        Args:
            path_data: Optional file path or URL to raw Speed Dating CSV.
            max_records_per_distribution: Optional sample limit per distribution (overrides config).
            apply_subsampling: Whether to apply subsampling according to max_records_per_distribution (default True).
            **kwargs: Extra parameters for compatibility.

        Returns:
            TwoSampleDataContainer holding Matched (X) and Unmatched (Y) feature matrices.
        """
        path_to_load = path_data or self.config.path_data_file
        df_raw = self.loader.load_data_raw(path_source=path_to_load)

        from ..feature_tracker import FeatureOperationRecorder

        recorder = FeatureOperationRecorder()

        # 1. Clean survey fields and impute missing ratings
        df_clean = self.cleaner.clean_dataset_survey_fields(df_raw, self.config)

        # 2. Reconstruct reciprocal date interaction pairs
        df_pairs = self.pair_builder.build_pairs_reciprocal(df_clean)

        # 3. Construct joint profile vectors and homophily interaction features
        df_joint = self.encoder.encode_features_joint(df_pairs, self.config, recorder=recorder)

        # 4. Partition into Distribution X (match = 1) and Distribution Y (match = 0)
        df_x, df_y = self.splitter.split_samples_by_match(df_joint)

        # Align columns between X and Y
        all_cols = sorted(list(set(df_x.columns).intersection(set(df_y.columns))))
        df_x = df_x[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df_y = df_y[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        matrix_x = df_x.to_numpy(dtype=np.float64)
        matrix_y = df_y.to_numpy(dtype=np.float64)
        name_features = all_cols

        # Record removed raw columns
        used_sources: ty.Set[str] = set()
        for item in recorder.get_feature_items(include_removed=False):
            if item.feature_original.startswith("[") and item.feature_original.endswith("]"):
                for c in df_raw.columns:
                    if f"'{c}'" in item.feature_original or f'"{c}"' in item.feature_original:
                        used_sources.add(c)
                    # end if
                # end for c
            else:
                used_sources.add(item.feature_original)
            # end if
        # end for item

        # Also account for columns merged/transformed into pairs
        used_sources.add("field_cd")

        removed_cols = sorted(list(set(df_raw.columns) - used_sources))
        for rem_col in removed_cols:
            recorder.record_removed(source_original=rem_col)
        # end for rem_col

        self._feature_operations = recorder.get_feature_items(include_removed=True)

        limit_records = (
            max_records_per_distribution
            if max_records_per_distribution is not None
            else self.config.max_records_per_distribution
        )

        n_orig_x = int(matrix_x.shape[0])
        n_orig_y = int(matrix_y.shape[0])

        if apply_subsampling and limit_records is not None and limit_records > 0:
            rng = np.random.RandomState(self.config.random_seed_sampling)
            if matrix_x.shape[0] > limit_records:
                idx_x = rng.choice(matrix_x.shape[0], size=limit_records, replace=False)
                matrix_x = matrix_x[idx_x]
            # end if
            if matrix_y.shape[0] > limit_records:
                idx_y = rng.choice(matrix_y.shape[0], size=limit_records, replace=False)
                matrix_y = matrix_y[idx_y]
            # end if
        # end if

        return TwoSampleDataContainer(
            sample_matrix_x=matrix_x,
            sample_matrix_y=matrix_y,
            name_features=name_features,
            metadata_dataset={
                "dataset_name": "Columbia Speed Dating Experiment",
                "n_samples_x": int(matrix_x.shape[0]),
                "n_samples_y": int(matrix_y.shape[0]),
                "n_samples_x_original": n_orig_x,
                "n_samples_y_original": n_orig_y,
                "max_records_per_distribution": limit_records,
                "num_features": int(matrix_x.shape[1]),
                "label_x_description": "Mutual Match (X: match = 1)",
                "label_y_description": "No Mutual Match (Y: match = 0)",
                "feature_operations": [f.model_dump() for f in self._feature_operations]
            }
        )
        # end def prepare_two_sample_data

    def track_feature_operations(
        self,
        container: ty.Optional[TwoSampleDataContainer] = None,
        include_removed: bool = True
    ) -> ty.List[ty.Any]:
        """Tracks the lineage, source columns, and data types of all Speed Dating processed features.

        Args:
            container: Optional preprocessed container.
            include_removed: Whether to include removed raw features.

        Returns:
            List of FeatureItemData specifications stored during preprocessing.
        """
        if self._feature_operations is None:
            self.prepare_two_sample_data(apply_subsampling=False)
        # end if

        if include_removed:
            return list(self._feature_operations)
        # end if
        return [f for f in self._feature_operations if f.type_feature != "removed"]
        # end def track_feature_operations
# end class SpeedDatingPreprocessor
