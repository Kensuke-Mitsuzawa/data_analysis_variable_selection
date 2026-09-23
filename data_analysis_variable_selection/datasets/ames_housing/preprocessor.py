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
        self._feature_operations: ty.Optional[ty.List[ty.Any]] = None
        # end def __init__

    def prepare_two_sample_data(
        self,
        path_data: ty.Optional[str] = None,
        max_records_per_distribution: ty.Optional[int] = None,
        apply_subsampling: bool = True,
        **kwargs: ty.Any
    ) -> TwoSampleDataContainer:
        """Executes ingestion, Domain NA cleanup, neighborhood imputation, encoding, and temporal splitting.

        Args:
            path_data: Optional file path to raw Ames CSV. If None, uses config or OpenML.
            max_records_per_distribution: Optional sample limit per distribution (overrides config if provided).
            apply_subsampling: Whether to apply subsampling according to max_records_per_distribution (default True).
            **kwargs: Extra parameters for compatibility.

        Returns:
            TwoSampleDataContainer holding Pre-Crash (X) and Post-Crash (Y) matrices and feature names.
        """
        path_to_load = path_data or self.config.path_data_file
        df_raw = self.loader.load_data_raw(path_source=path_to_load)

        from ..feature_tracker import FeatureOperationRecorder, FeatureItemData

        recorder = FeatureOperationRecorder()

        # 1. Clean Domain NAs
        df_clean = self.cleaner.clean_dataset_domain_nas(df_raw, self.config)

        # 2. Impute LotFrontage by Neighborhood and remaining columns
        df_imputed = self.cleaner.impute_missing_values_neighborhood(df_clean)
        df_imputed = self.cleaner.impute_remaining_features_generic(df_imputed)

        # Record LotFrontage imputation operation
        recorder.record_feature(
            name_processed="LotFrontage",
            source_original=["LotFrontage", "Neighborhood"],
            type_feature="float"
        )

        # 3. Ordinal mapping
        df_ordinal = self.encoder.encode_features_ordinal(df_imputed, self.config.ordinal_mapping_dicts)
        for col_ord in self.config.ordinal_mapping_dicts.keys():
            if col_ord in df_ordinal.columns:
                recorder.record_feature(
                    name_processed=col_ord,
                    source_original=col_ord,
                    type_feature="int"
                )
            # end if
        # end for col_ord

        # 4. One-Hot Encoding for nominal features
        df_encoded, dict_onehot_lineage = self.encoder.encode_features_nominal_onehot(
            df_ordinal,
            return_lineage=True
        )
        for src_col, dummy_cols in dict_onehot_lineage.items():
            recorder.record_onehot_expansion(
                source_column=src_col,
                generated_columns=dummy_cols
            )
        # end for src_col

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

        # Record pass-through continuous and discrete numeric features
        int_cols = {
            "YearBuilt", "YearRemodAdd", "GarageYrBlt", "BedroomAbvGr", "KitchenAbvGr",
            "TotRmsAbvGrd", "Fireplaces", "GarageCars", "FullBath", "HalfBath",
            "BsmtFullBath", "BsmtHalfBath", "OverallQual", "OverallCond", "MoSold"
        }
        for feat in name_features:
            if feat not in recorder._processed_names:
                type_feat = "int" if feat in int_cols else "float"
                recorder.record_feature(
                    name_processed=feat,
                    source_original=feat,
                    type_feature=type_feat
                )
            # end if
        # end for feat

        # Record removed raw columns
        used_raw_sources: ty.Set[str] = set(dict_onehot_lineage.keys())
        for item in recorder.get_feature_items(include_removed=False):
            if item.feature_original.startswith("[") and item.feature_original.endswith("]"):
                for raw_col in df_raw.columns:
                    if f"'{raw_col}'" in item.feature_original or f'"{raw_col}"' in item.feature_original:
                        used_raw_sources.add(raw_col)
                    # end if
                # end for raw_col
            else:
                used_raw_sources.add(item.feature_original)
            # end if
        # end for item

        removed_cols = sorted(list(set(df_raw.columns) - used_raw_sources))
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
                "feature_operations": [f.model_dump() for f in self._feature_operations]
            }
        )
        # end def prepare_two_sample_data

    def track_feature_operations(
        self,
        container: ty.Optional[TwoSampleDataContainer] = None,
        include_removed: bool = True
    ) -> ty.List[ty.Any]:
        """Tracks the lineage, source columns, and data types of all Ames Housing processed features.

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
# end class AmesHousingPreprocessor
