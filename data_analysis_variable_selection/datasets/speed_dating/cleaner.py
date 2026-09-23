import logging
import typing as ty
import numpy as np
import pandas as pd

from .config import SpeedDatingPreprocessingConfig

logger = logging.getLogger(__name__)


class SpeedDatingDataCleaner:
    """Handles survey data cleaning, median and mode imputation, and type coercion.
    """

    def clean_dataset_survey_fields(
        self,
        df_raw: pd.DataFrame,
        config: SpeedDatingPreprocessingConfig
    ) -> pd.DataFrame:
        """Cleans survey ratings, demographics, and fills missing values.

        Args:
            df_raw: Raw scorecard DataFrame.
            config: SpeedDatingPreprocessingConfig instance.

        Returns:
            Cleaned DataFrame with numeric coercions and imputed values.
        """
        df_clean = df_raw.copy()

        # Coerce core numeric columns
        numeric_targets = (
            config.columns_demographics
            + config.columns_interests
            + config.columns_self_ratings
            + config.columns_stated_preferences
        )
        for col in numeric_targets:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            # end if
        # end for col

        # Ensure ID, wave, and gender columns are valid integers
        id_cols = ["iid", "pid", "wave", "gender", "match", "dec", "dec_o"]
        for col in id_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            # end if
        # end for col

        # Drop rows with missing critical identifiers
        df_clean = df_clean.dropna(subset=["iid", "pid", "gender"]).copy()
        df_clean["iid"] = df_clean["iid"].astype(int)
        df_clean["pid"] = df_clean["pid"].astype(int)
        df_clean["gender"] = df_clean["gender"].astype(int)
        if "wave" in df_clean.columns:
            df_clean["wave"] = df_clean["wave"].fillna(1).astype(int)
        # end if

        # 1. Impute 1-10 Likert ratings and continuous variables via median
        impute_median_cols = (
            config.columns_interests
            + config.columns_self_ratings
            + config.columns_stated_preferences
            + ["age", "imprace", "imprelig", "date", "go_out", "exphappy", "expnum"]
        )
        df_clean = self.impute_missing_ratings_median(df_clean, impute_median_cols)

        # 2. Impute nominal categorical columns via mode
        impute_mode_cols = ["goal", "race", "field_cd"]
        df_clean = self.impute_missing_categoricals_mode(df_clean, impute_mode_cols)

        return df_clean
        # end def clean_dataset_survey_fields

    def impute_missing_ratings_median(
        self,
        df_data: pd.DataFrame,
        columns_to_impute: ty.List[str]
    ) -> pd.DataFrame:
        """Fills missing numeric values with column medians.

        Args:
            df_data: DataFrame to process.
            columns_to_impute: List of numeric column names.

        Returns:
            DataFrame with median-imputed values.
        """
        df_out = df_data.copy()
        for col in columns_to_impute:
            if col in df_out.columns:
                med_val = df_out[col].median()
                if pd.isna(med_val):
                    med_val = 5.0
                # end if
                df_out[col] = df_out[col].fillna(med_val)
            # end if
        # end for col
        return df_out
        # end def impute_missing_ratings_median

    def impute_missing_categoricals_mode(
        self,
        df_data: pd.DataFrame,
        columns_to_impute: ty.List[str]
    ) -> pd.DataFrame:
        """Fills missing categorical codes with column modes or fallback default.

        Args:
            df_data: DataFrame to process.
            columns_to_impute: List of categorical column names.

        Returns:
            DataFrame with mode-imputed categories.
        """
        df_out = df_data.copy()
        for col in columns_to_impute:
            if col in df_out.columns:
                series_non_na = df_out[col].dropna()
                if not series_non_na.empty:
                    mode_val = series_non_na.mode()[0]
                else:
                    mode_val = 1
                # end if
                df_out[col] = df_out[col].fillna(mode_val)
            # end if
        # end for col
        return df_out
        # end def impute_missing_categoricals_mode
# end class SpeedDatingDataCleaner
