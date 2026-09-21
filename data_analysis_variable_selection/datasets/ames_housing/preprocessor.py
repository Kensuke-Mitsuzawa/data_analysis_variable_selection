import typing as ty
import numpy as np
import pandas as pd

from ...common.base_preprocessor import BaseDatasetPreprocessor
from ...common.models import TwoSampleDataContainer
from .config import AmesPreprocessingConfig
from .loader import AmesHousingDataLoader
from .cleaner import AmesHousingDataCleaner
from .encoder import AmesHousingFeatureEncoder
from .splitter import AmesHousingTemporalSplitter


class AmesHousingPreprocessor(BaseDatasetPreprocessor):
    """Coordinates the end-to-end transformation of the Ames Housing dataset into a standardized two-sample container.
    """

    def __init__(
        self,
        config: ty.Optional[AmesPreprocessingConfig] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        random_seed_sampling: ty.Optional[int] = None
    ):
        """Initializes the preprocessor.

        Args:
            config: Optional custom AmesPreprocessingConfig. Defaults to standard configuration.
            max_records_per_distribution: Optional maximum sample limit per distribution to manage O(N^2) MMD complexity.
            random_seed_sampling: Optional random seed for reproducible sampling.
        """
        self.config = config or AmesPreprocessingConfig()
        if max_records_per_distribution is not None:
            self.config.max_records_per_distribution = max_records_per_distribution
        # end if
        if random_seed_sampling is not None:
            self.config.random_seed_sampling = random_seed_sampling
        # end if
        self.loader = AmesHousingDataLoader()
        self.cleaner = AmesHousingDataCleaner()
        self.encoder = AmesHousingFeatureEncoder()
        self.splitter = AmesHousingTemporalSplitter()
        # end def __init__

    def prepare_two_sample_data(
        self,
        path_data: ty.Optional[str] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        **kwargs: ty.Any
    ) -> TwoSampleDataContainer:
        """Executes ingestion, Domain NA cleanup, neighborhood imputation, encoding, and temporal splitting.

        Args:
            path_data: Optional file path to raw Ames CSV. If None, uses config or OpenML.
            max_records_per_distribution: Optional sample limit per distribution (overrides config if provided).
            **kwargs: Extra parameters for compatibility.

        Returns:
            TwoSampleDataContainer holding Pre-Crash (X) and Post-Crash (Y) matrices and feature names.
        """
        path_to_load = path_data or self.config.path_data_file
        df_raw = self.loader.load_data_raw(path_source=path_to_load)

        # 1. Clean Domain NAs
        df_clean = self.cleaner.clean_dataset_domain_nas(df_raw, self.config)

        # 2. Impute LotFrontage by Neighborhood and remaining columns
        df_imputed = self.cleaner.impute_missing_values_neighborhood(df_clean)
        df_imputed = self.cleaner.impute_remaining_features_generic(df_imputed)

        # 3. Ordinal mapping
        df_ordinal = self.encoder.encode_features_ordinal(df_imputed, self.config.ordinal_mapping_dicts)

        # 4. One-Hot Encoding for nominal features
        df_encoded = self.encoder.encode_features_nominal_onehot(df_ordinal)

        # 5. Temporal Two-Sample Split
        df_x, df_y = self.splitter.split_samples_temporal(df_encoded, self.config)

        # Align columns between X and Y
        all_cols = sorted(list(set(df_x.columns).union(set(df_y.columns))))
        df_x = df_x.reindex(columns=all_cols, fill_value=0.0)
        df_y = df_y.reindex(columns=all_cols, fill_value=0.0)

        # Ensure all columns are purely numeric
        df_x_num = df_x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df_y_num = df_y.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        matrix_x = df_x_num.to_numpy(dtype=np.float64)
        matrix_y = df_y_num.to_numpy(dtype=np.float64)
        name_features = list(df_x_num.columns)

        limit_records = (
            max_records_per_distribution
            if max_records_per_distribution is not None
            else self.config.max_records_per_distribution
        )

        n_orig_x = int(matrix_x.shape[0])
        n_orig_y = int(matrix_y.shape[0])

        if limit_records is not None and limit_records > 0:
            rng = np.random.RandomState(self.config.random_seed_sampling)
            if matrix_x.shape[0] > limit_records:
                chosen_idx_x = rng.choice(matrix_x.shape[0], size=limit_records, replace=False)
                matrix_x = matrix_x[chosen_idx_x]
            # end if
            if matrix_y.shape[0] > limit_records:
                chosen_idx_y = rng.choice(matrix_y.shape[0], size=limit_records, replace=False)
                matrix_y = matrix_y[chosen_idx_y]
            # end if
        # end if

        return TwoSampleDataContainer(
            sample_matrix_x=matrix_x,
            sample_matrix_y=matrix_y,
            name_features=name_features,
            metadata_dataset={
                "dataset_name": "Ames Housing Dataset",
                "n_samples_x": int(matrix_x.shape[0]),
                "n_samples_y": int(matrix_y.shape[0]),
                "n_samples_x_original": n_orig_x,
                "n_samples_y_original": n_orig_y,
                "max_records_per_distribution": limit_records,
                "num_features": int(matrix_x.shape[1]),
                "label_x_description": f"Pre-Crash Market ({self.config.years_pre_crash})",
                "label_y_description": f"Post-Crash Market ({self.config.years_post_crash})",
            }
        )
        # end def prepare_two_sample_data
# end class AmesHousingPreprocessor
