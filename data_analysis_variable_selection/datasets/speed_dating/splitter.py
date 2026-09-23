import logging
import typing as ty
import pandas as pd

logger = logging.getLogger(__name__)


class SpeedDatingLabelSplitter:
    """Partitions joint date interaction pairs into Distribution X (match = 1) and Distribution Y (match = 0).
    """

    def split_samples_by_match(
        self,
        df_joint: pd.DataFrame
    ) -> ty.Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits joint dataframe into two sample distributions and purges leakage columns.

        Args:
            df_joint: DataFrame containing joint features and the 'match' column.

        Returns:
            Tuple of (df_x, df_y) representing Matched and Unmatched feature spaces.
        """
        if "match" not in df_joint.columns:
            raise KeyError("Column 'match' is required in joint dataframe to partition distributions.")
        # end if

        mask_x = (df_joint["match"] == 1)
        mask_y = (df_joint["match"] == 0)

        df_x = df_joint[mask_x].copy()
        df_y = df_joint[mask_y].copy()

        # Purge leakage columns
        df_x = self.drop_leakage_columns(df_x)
        df_y = self.drop_leakage_columns(df_y)

        logger.info(
            f"Split into Distribution X ({len(df_x)} matched pairs) and "
            f"Distribution Y ({len(df_y)} unmatched pairs)."
        )
        return df_x, df_y
        # end def split_samples_by_match

    def drop_leakage_columns(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Removes outcome labels and experimental tracking identifiers from feature matrix.

        Args:
            df_data: DataFrame with features.

        Returns:
            DataFrame with leakage columns removed.
        """
        leakage_cols = [
            "match", "dec", "dec_o", "dec_male", "dec_female", "dec_o_male", "dec_o_female",
            "match_male", "match_female", "pair_id_hash", "iid_male", "pid_male",
            "iid_female", "pid_female"
        ]
        cols_to_drop = [col for col in leakage_cols if col in df_data.columns]
        return df_data.drop(columns=cols_to_drop)
        # end def drop_leakage_columns
# end class SpeedDatingLabelSplitter
