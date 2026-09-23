import logging
import typing as ty
import numpy as np
import pandas as pd

from .config import SpeedDatingPreprocessingConfig

logger = logging.getLogger(__name__)


class SpeedDatingFeatureEncoder:
    """Encodes joint reciprocal features, homophily interaction deltas, and cosine similarities.
    """

    def encode_features_joint(
        self,
        df_pairs: pd.DataFrame,
        config: SpeedDatingPreprocessingConfig
    ) -> pd.DataFrame:
        """Constructs joint profile vectors and interaction difference features.

        Args:
            df_pairs: Paired DataFrame from SpeedDatingPairBuilder.
            config: SpeedDatingPreprocessingConfig instance.

        Returns:
            DataFrame containing joint features and the 'match' outcome column.
        """
        dict_features: ty.Dict[str, ty.Any] = {}

        # Retain match target
        dict_features["match"] = df_pairs["match"].values

        # 1. Direct Concatenation of Participant Features
        core_columns = (
            config.columns_demographics
            + config.columns_interests
            + config.columns_self_ratings
            + config.columns_stated_preferences
        )

        for col in core_columns:
            col_m = f"{col}_male"
            col_f = f"{col}_female"
            if col_m in df_pairs.columns:
                dict_features[f"Male_{col}"] = df_pairs[col_m].astype(float).values
            # end if
            if col_f in df_pairs.columns:
                dict_features[f"Female_{col}"] = df_pairs[col_f].astype(float).values
            # end if
        # end for col

        # 2. Homophily & Demographic Differences
        if "age_male" in df_pairs.columns and "age_female" in df_pairs.columns:
            dict_features["Age_Gap"] = np.abs(
                df_pairs["age_male"].astype(float).values - df_pairs["age_female"].astype(float).values
            )
        # end if

        if "race_male" in df_pairs.columns and "race_female" in df_pairs.columns:
            same_race = (df_pairs["race_male"] == df_pairs["race_female"]).astype(float).values
            dict_features["Same_Race"] = same_race

            if "imprace_male" in df_pairs.columns and "imprace_female" in df_pairs.columns:
                avg_imprace = (
                    df_pairs["imprace_male"].astype(float).values + df_pairs["imprace_female"].astype(float).values
                ) / 2.0
                dict_features["Race_Preference_Conflict"] = (1.0 - same_race) * avg_imprace
            # end if
        # end if

        if "goal_male" in df_pairs.columns and "goal_female" in df_pairs.columns:
            dict_features["Same_Goal"] = (
                df_pairs["goal_male"] == df_pairs["goal_female"]
            ).astype(float).values
        # end if

        if "field_cd_male" in df_pairs.columns and "field_cd_female" in df_pairs.columns:
            dict_features["Same_Field"] = (
                df_pairs["field_cd_male"] == df_pairs["field_cd_female"]
            ).astype(float).values
        # end if

        # 3. Leisure & Activity Differences (17 domains)
        interest_names = config.columns_interests
        m_interests_matrix: ty.List[np.ndarray] = []
        f_interests_matrix: ty.List[np.ndarray] = []

        for inter in interest_names:
            col_m = f"{inter}_male"
            col_f = f"{inter}_female"
            if col_m in df_pairs.columns and col_f in df_pairs.columns:
                vals_m = df_pairs[col_m].astype(float).values
                vals_f = df_pairs[col_f].astype(float).values
                dict_features[f"Diff_{inter}"] = np.abs(vals_m - vals_f)
                m_interests_matrix.append(vals_m)
                f_interests_matrix.append(vals_f)
            # end if
        # end for inter

        # 4. Cosine Similarity Across Holistic Interest Vectors
        if m_interests_matrix and f_interests_matrix:
            mat_m = np.column_stack(m_interests_matrix)
            mat_f = np.column_stack(f_interests_matrix)
            dict_features["Interest_Cosine_Sim"] = self.compute_similarity_interest_cosine(mat_m, mat_f)
        # end if

        # 5. Preference-Trait Alignment Deltas
        # Normalize stated preference weights (summing to ~100) to 1-10 scale
        if "attr3_1_male" in df_pairs.columns and "attr1_1_female" in df_pairs.columns:
            pref_attr_f = df_pairs["attr1_1_female"].astype(float).values / 10.0
            dict_features["Delta_Male_Attr_Align"] = np.abs(
                df_pairs["attr3_1_male"].astype(float).values - pref_attr_f
            )
        # end if

        if "attr3_1_female" in df_pairs.columns and "attr1_1_male" in df_pairs.columns:
            pref_attr_m = df_pairs["attr1_1_male"].astype(float).values / 10.0
            dict_features["Delta_Female_Attr_Align"] = np.abs(
                df_pairs["attr3_1_female"].astype(float).values - pref_attr_m
            )
        # end if

        if "intel3_1_male" in df_pairs.columns and "intel1_1_female" in df_pairs.columns:
            pref_intel_f = df_pairs["intel1_1_female"].astype(float).values / 10.0
            dict_features["Delta_Male_Intel_Align"] = np.abs(
                df_pairs["intel3_1_male"].astype(float).values - pref_intel_f
            )
        # end if

        if "intel3_1_female" in df_pairs.columns and "intel1_1_male" in df_pairs.columns:
            pref_intel_m = df_pairs["intel1_1_male"].astype(float).values / 10.0
            dict_features["Delta_Female_Intel_Align"] = np.abs(
                df_pairs["intel3_1_female"].astype(float).values - pref_intel_m
            )
        # end if

        df_joint = pd.DataFrame(dict_features)
        # Ensure purely numeric representation and fill any remaining NaNs with 0.0
        df_joint = df_joint.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        logger.info(f"Encoded joint speed dating feature space with {df_joint.shape[1] - 1} features.")
        return df_joint
        # end def encode_features_joint

    def compute_similarity_interest_cosine(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray
    ) -> np.ndarray:
        """Computes row-wise cosine similarity between two feature matrices.

        Args:
            matrix_a: 2D array of shape (N, D).
            matrix_b: 2D array of shape (N, D).

        Returns:
            1D array of cosine similarities of length N bounded in [-1.0, 1.0].
        """
        dot_product = np.sum(matrix_a * matrix_b, axis=1)
        norm_a = np.linalg.norm(matrix_a, axis=1)
        norm_b = np.linalg.norm(matrix_b, axis=1)
        denom = norm_a * norm_b
        zero_mask = denom < 1e-9
        denom[zero_mask] = 1.0

        similarity = dot_product / denom
        similarity[zero_mask] = 0.0
        return np.clip(similarity, -1.0, 1.0)
        # end def compute_similarity_interest_cosine
# end class SpeedDatingFeatureEncoder
