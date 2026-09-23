import hashlib
import logging
import typing as ty
import pandas as pd

logger = logging.getLogger(__name__)


class SpeedDatingPairBuilder:
    """Merges individual participant scorecards into reciprocal date interaction pairs.
    """

    def build_pairs_reciprocal(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Forms unified date encounters by inner joining male and female scorecards.

        Args:
            df_clean: Cleaned scorecard DataFrame with gender, iid, pid, and wave.

        Returns:
            DataFrame where each row represents an intact reciprocal date encounter.
        """
        df_male = df_clean[df_clean["gender"] == 1].copy()
        df_female = df_clean[df_clean["gender"] == 0].copy()

        # Inner join to ensure both participants handed in evaluations
        join_keys = ["wave"]
        if "wave" in df_clean.columns:
            df_pairs = df_male.merge(
                df_female,
                left_on=["iid", "pid", "wave"],
                right_on=["pid", "iid", "wave"],
                suffixes=("_male", "_female")
            )
        else:
            df_pairs = df_male.merge(
                df_female,
                left_on=["iid", "pid"],
                right_on=["pid", "iid"],
                suffixes=("_male", "_female")
            )
        # end if

        # Generate symmetric pair hash IDs
        list_hash_ids: ty.List[str] = []
        for _, row in df_pairs.iterrows():
            id_m = int(row["iid_male"])
            id_f = int(row["iid_female"])
            list_hash_ids.append(self.generate_id_pair_hash(id_m, id_f))
        # end for row
        df_pairs["pair_id_hash"] = list_hash_ids

        # Ensure consistent mutual match label: both male and female agreed
        if "dec_male" in df_pairs.columns and "dec_female" in df_pairs.columns:
            df_pairs["match"] = (
                (df_pairs["dec_male"] == 1) & (df_pairs["dec_female"] == 1)
            ).astype(int)
        elif "match_male" in df_pairs.columns:
            df_pairs["match"] = df_pairs["match_male"].astype(int)
        elif "match_female" in df_pairs.columns:
            df_pairs["match"] = df_pairs["match_female"].astype(int)
        else:
            df_pairs["match"] = 0
        # end if

        logger.info(
            f"Formed {len(df_pairs)} reciprocal date pairs (Matches: {(df_pairs['match'] == 1).sum()}, "
            f"Non-matches: {(df_pairs['match'] == 0).sum()})."
        )
        return df_pairs
        # end def build_pairs_reciprocal

    def generate_id_pair_hash(self, id_person_a: int, id_person_b: int) -> str:
        """Generates a symmetric hash identifier for a date pair.

        Args:
            id_person_a: First participant ID.
            id_person_b: Second participant ID.

        Returns:
            16-character hexadecimal hash string.
        """
        pair_key = f"{min(id_person_a, id_person_b)}_{max(id_person_a, id_person_b)}"
        return hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:16]
        # end def generate_id_pair_hash
# end class SpeedDatingPairBuilder
