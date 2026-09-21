import typing as ty
import pandas as pd
from .config import AmesPreprocessingConfig


class AmesHousingTemporalSplitter:
    """Partitions the housing market observations into pre-crash and post-crash distributions and strips leakage.
    """

    def split_samples_temporal(
        self,
        df_encoded: pd.DataFrame,
        config: AmesPreprocessingConfig
    ) -> ty.Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into Pre-Crash (X) and Post-Crash (Y) subsets based on YrSold.

        Args:
            df_encoded: DataFrame containing encoded features and YrSold.
            config: AmesPreprocessingConfig holding temporal split parameters.

        Returns:
            Tuple of (df_x, df_y) representing the two distributions.
        """
        assert "YrSold" in df_encoded.columns, "YrSold column required to perform temporal two-sample split."

        series_yr = pd.to_numeric(df_encoded["YrSold"], errors="coerce")

        mask_x = series_yr.isin(config.years_pre_crash)
        mask_y = series_yr.isin(config.years_post_crash)

        df_x = df_encoded[mask_x].copy()
        df_y = df_encoded[mask_y].copy()

        # Strip temporal and price leakage columns
        df_x_clean = self.drop_leakage_columns(df_x, config.columns_to_drop)
        df_y_clean = self.drop_leakage_columns(df_y, config.columns_to_drop)

        return df_x_clean, df_y_clean
        # end def split_samples_temporal

    def drop_leakage_columns(
        self,
        df_data: pd.DataFrame,
        columns_to_drop: ty.List[str]
    ) -> pd.DataFrame:
        """Removes columns associated with label leakage, timestamps, or arbitrary row IDs.

        Args:
            df_data: Input DataFrame.
            columns_to_drop: List of column names to eliminate.

        Returns:
            Clean feature DataFrame.
        """
        existing_cols = [c for c in columns_to_drop if c in df_data.columns]
        df_pruned = df_data.drop(columns=existing_cols)
        return df_pruned
        # end def drop_leakage_columns
# end class AmesHousingTemporalSplitter
