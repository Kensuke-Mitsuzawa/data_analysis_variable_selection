import pandas as pd
import numpy as np
from .config import AmesPreprocessingConfig


class AmesHousingDataCleaner:
    """Cleans domain-specific missing values (the 'Domain NA' trap) and performs stratified imputation.
    """

    def clean_dataset_domain_nas(
        self,
        df_raw: pd.DataFrame,
        config: AmesPreprocessingConfig
    ) -> pd.DataFrame:
        """Fixes Domain NAs where missingness denotes absence of physical structures.

        Args:
            df_raw: Raw housing dataframe.
            config: AmesPreprocessingConfig holding column groups.

        Returns:
            Cleaned DataFrame with Domain NAs resolved to 'None' or 0.
        """
        df_clean = df_raw.copy()

        # Categorical absence -> 'None'
        for col in config.categorical_na_to_none_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("None").astype(str)
            # end if
        # end for col

        # Continuous absence -> 0.0
        for col in config.continuous_na_to_zero_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)
            # end if
        # end for col

        return df_clean
        # end def clean_dataset_domain_nas

    def impute_missing_values_neighborhood(
        self,
        df_clean: pd.DataFrame,
        column_target: str = "LotFrontage",
        column_group: str = "Neighborhood"
    ) -> pd.DataFrame:
        """Imputes genuine missing values using neighborhood-level medians.

        Args:
            df_clean: Cleaned DataFrame.
            column_target: Feature to impute (default: LotFrontage).
            column_group: Grouping column (default: Neighborhood).

        Returns:
            DataFrame with target column imputed.
        """
        df_imputed = df_clean.copy()

        if column_target in df_imputed.columns and column_group in df_imputed.columns:
            global_median = float(df_imputed[column_target].median())
            if np.isnan(global_median):
                global_median = 60.0
            # end if

            medians_by_neighborhood = df_imputed.groupby(column_group)[column_target].transform("median")
            df_imputed[column_target] = df_imputed[column_target].fillna(medians_by_neighborhood)
            df_imputed[column_target] = df_imputed[column_target].fillna(global_median)
        # end if

        return df_imputed
        # end def impute_missing_values_neighborhood

    def impute_remaining_features_generic(self, df_imputed: pd.DataFrame) -> pd.DataFrame:
        """Imputes any leftover missing values using column-wise median (numeric) or mode (categorical).

        Args:
            df_imputed: Partially imputed DataFrame.

        Returns:
            Fully populated DataFrame with zero NaNs.
        """
        df_final = df_imputed.copy()

        for col in df_final.columns:
            if df_final[col].isna().any():
                if pd.api.types.is_numeric_dtype(df_final[col]):
                    col_med = df_final[col].median()
                    df_final[col] = df_final[col].fillna(col_med if pd.notna(col_med) else 0.0)
                else:
                    col_mode = df_final[col].mode()
                    fill_val = col_mode.iloc[0] if len(col_mode) > 0 else "None"
                    df_final[col] = df_final[col].fillna(fill_val)
                # end if
            # end if
        # end for col

        return df_final
        # end def impute_remaining_features_generic
# end class AmesHousingDataCleaner
