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
        config: SpeedDatingPreprocessingConfig,
        recorder: ty.Optional[ty.Any] = None
    ) -> pd.DataFrame:
        """Constructs joint profile vectors and interaction difference features.

        Args:
            df_pairs: Paired DataFrame from SpeedDatingPairBuilder.
            config: SpeedDatingPreprocessingConfig instance.
            recorder: Optional FeatureOperationRecorder to record transformations.

        Returns:
            DataFrame containing joint features and the 'match' outcome column.
        """
        dict_features: ty.Dict[str, ty.Any] = {}

        # Retain match target
        dict_features["match"] = df_pairs["match"].values

        # 1. Age Difference
        if "age_male" in df_pairs.columns and "age_female" in df_pairs.columns:
            dict_features["Diff_age"] = np.abs(
                df_pairs["age_male"].astype(float).values - df_pairs["age_female"].astype(float).values
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Diff_age",
                    source_original="age",
                    type_feature="float"
                )
            # end if
        # end if

        # 2. Leisure & Activity Differences (17 domains)
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
                if recorder is not None:
                    recorder.record_feature(
                        name_processed=f"Diff_{inter}",
                        source_original=inter,
                        type_feature="float"
                    )
                # end if
            # end if
        # end for inter

        # 3. Cosine Similarity Across Holistic Interest Vectors
        if m_interests_matrix and f_interests_matrix:
            mat_m = np.column_stack(m_interests_matrix)
            mat_f = np.column_stack(f_interests_matrix)
            dict_features["Interest_Cosine_Sim"] = self.compute_similarity_interest_cosine(mat_m, mat_f)
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Interest_Cosine_Sim",
                    source_original=sorted(interest_names),
                    type_feature="float"
                )
            # end if
        # end if

        # 4. Trait Differences (attr*_1, sinc*_1, intel*_1, fun*_1, amb*_1, shar*_1)
        trait_columns = config.columns_stated_preferences + config.columns_self_ratings
        for trait in trait_columns:
            col_m = f"{trait}_male"
            col_f = f"{trait}_female"
            if col_m in df_pairs.columns and col_f in df_pairs.columns:
                vals_m = df_pairs[col_m].astype(float).values
                vals_f = df_pairs[col_f].astype(float).values
                dict_features[f"Diff_{trait}"] = np.abs(vals_m - vals_f)
                if recorder is not None:
                    recorder.record_feature(
                        name_processed=f"Diff_{trait}",
                        source_original=trait,
                        type_feature="float"
                    )
                # end if
            # end if
        # end for trait

        # 5. Survey Field Differences (goal, date, go_out, exphappy, expnum)
        survey_diff_columns = ["goal", "date", "go_out", "exphappy", "expnum"]
        for s_col in survey_diff_columns:
            col_m = f"{s_col}_male"
            col_f = f"{s_col}_female"
            if col_m in df_pairs.columns and col_f in df_pairs.columns:
                vals_m = df_pairs[col_m].astype(float).values
                vals_f = df_pairs[col_f].astype(float).values
                dict_features[f"Diff_{s_col}"] = np.abs(vals_m - vals_f)
                if recorder is not None:
                    recorder.record_feature(
                        name_processed=f"Diff_{s_col}",
                        source_original=s_col,
                        type_feature="float"
                    )
                # end if
            # end if
        # end for s_col

        # 6. Demographics Differences (imprace, imprelig)
        for col in ["imprace", "imprelig"]:
            col_m = f"{col}_male"
            col_f = f"{col}_female"
            if col_m in df_pairs.columns and col_f in df_pairs.columns:
                vals_m = df_pairs[col_m].astype(float).values
                vals_f = df_pairs[col_f].astype(float).values
                dict_features[f"Diff_{col}"] = np.abs(vals_m - vals_f)
                if recorder is not None:
                    recorder.record_feature(
                        name_processed=f"Diff_{col}",
                        source_original=col,
                        type_feature="float"
                    )
                # end if
            # end if
        # end for col

        # 7. Career Grouping (Tiers 1-5)
        career_tier_map: ty.Dict[int, int] = {
            1: 1, 4: 1, 7: 1,
            5: 2, 17: 2,
            2: 3, 3: 3, 9: 3, 11: 3, 12: 3, 13: 3, 16: 3,
            6: 4, 8: 4, 14: 4,
            10: 5, 15: 5,
        }
        if "career_c_male" in df_pairs.columns:
            m_codes = pd.to_numeric(df_pairs["career_c_male"], errors="coerce").fillna(10).astype(int)
            dict_features["career_group_male"] = m_codes.map(career_tier_map).fillna(5).astype(int).values
            if recorder is not None:
                recorder.record_feature(
                    name_processed="career_group_male",
                    source_original="career_c",
                    type_feature="int"
                )
            # end if
        # end if

        if "career_c_female" in df_pairs.columns:
            f_codes = pd.to_numeric(df_pairs["career_c_female"], errors="coerce").fillna(10).astype(int)
            dict_features["career_group_female"] = f_codes.map(career_tier_map).fillna(5).astype(int).values
            if recorder is not None:
                recorder.record_feature(
                    name_processed="career_group_female",
                    source_original="career_c",
                    type_feature="int"
                )
            # end if
        # end if

        # 8. Homophily & Agreement Indicators
        if "race_male" in df_pairs.columns and "race_female" in df_pairs.columns:
            same_race = (df_pairs["race_male"] == df_pairs["race_female"]).astype(float).values
            dict_features["Same_Race"] = same_race
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Same_Race",
                    source_original="race",
                    type_feature="category"
                )
            # end if

            if "imprace_male" in df_pairs.columns and "imprace_female" in df_pairs.columns:
                avg_imprace = (
                    df_pairs["imprace_male"].astype(float).values + df_pairs["imprace_female"].astype(float).values
                ) / 2.0
                dict_features["Race_Preference_Conflict"] = (1.0 - same_race) * avg_imprace
                if recorder is not None:
                    recorder.record_feature(
                        name_processed="Race_Preference_Conflict",
                        source_original=["race", "imprace"],
                        type_feature="float"
                    )
                # end if
            # end if
        # end if

        if "goal_male" in df_pairs.columns and "goal_female" in df_pairs.columns:
            dict_features["Same_Goal"] = (
                df_pairs["goal_male"] == df_pairs["goal_female"]
            ).astype(float).values
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Same_Goal",
                    source_original="goal",
                    type_feature="category"
                )
            # end if
        # end if

        if "field_cd_male" in df_pairs.columns and "field_cd_female" in df_pairs.columns:
            dict_features["Same_Field"] = (
                df_pairs["field_cd_male"] == df_pairs["field_cd_female"]
            ).astype(float).values
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Same_Field",
                    source_original="field_cd",
                    type_feature="category"
                )
            # end if

            dict_features["Field_Similarity"] = self.compute_field_similarity(
                df_pairs["field_cd_male"],
                df_pairs["field_cd_female"]
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Field_Similarity",
                    source_original="field_cd",
                    type_feature="float"
                )
            # end if
        # end if

        if "zipcode_male" in df_pairs.columns and "zipcode_female" in df_pairs.columns:
            dict_features["Same_region"] = self.compute_same_region(
                df_pairs["zipcode_male"],
                df_pairs["zipcode_female"]
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Same_region",
                    source_original="zipcode",
                    type_feature="category"
                )
            # end if
        # end if

        # 9. Preference-Trait Alignment Deltas
        # Normalize stated preference weights (summing to ~100) to 1-10 scale
        if "attr3_1_male" in df_pairs.columns and "attr1_1_female" in df_pairs.columns:
            pref_attr_f = df_pairs["attr1_1_female"].astype(float).values / 10.0
            dict_features["Delta_Male_Attr_Align"] = np.abs(
                df_pairs["attr3_1_male"].astype(float).values - pref_attr_f
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Delta_Male_Attr_Align",
                    source_original=["attr3_1", "attr1_1"],
                    type_feature="float"
                )
            # end if
        # end if

        if "attr3_1_female" in df_pairs.columns and "attr1_1_male" in df_pairs.columns:
            pref_attr_m = df_pairs["attr1_1_male"].astype(float).values / 10.0
            dict_features["Delta_Female_Attr_Align"] = np.abs(
                df_pairs["attr3_1_female"].astype(float).values - pref_attr_m
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Delta_Female_Attr_Align",
                    source_original=["attr3_1", "attr1_1"],
                    type_feature="float"
                )
            # end if
        # end if

        if "intel3_1_male" in df_pairs.columns and "intel1_1_female" in df_pairs.columns:
            pref_intel_f = df_pairs["intel1_1_female"].astype(float).values / 10.0
            dict_features["Delta_Male_Intel_Align"] = np.abs(
                df_pairs["intel3_1_male"].astype(float).values - pref_intel_f
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Delta_Male_Intel_Align",
                    source_original=["intel3_1", "intel1_1"],
                    type_feature="float"
                )
            # end if
        # end if

        if "intel3_1_female" in df_pairs.columns and "intel1_1_male" in df_pairs.columns:
            pref_intel_m = df_pairs["intel1_1_male"].astype(float).values / 10.0
            dict_features["Delta_Female_Intel_Align"] = np.abs(
                df_pairs["intel3_1_female"].astype(float).values - pref_intel_m
            )
            if recorder is not None:
                recorder.record_feature(
                    name_processed="Delta_Female_Intel_Align",
                    source_original=["intel3_1", "intel1_1"],
                    type_feature="float"
                )
            # end if
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

    def compute_field_similarity(
        self,
        series_male: pd.Series,
        series_female: pd.Series
    ) -> np.ndarray:
        """Computes deterministic domain-knowledge similarity between two field codes.

        Returns:
            1.0: Exact field match
            0.5: Different fields sharing the same epistemological macro-domain
            0.0: Divergent macro-domains or missing/unclassified codes
        """
        arr_m = pd.to_numeric(series_male, errors="coerce").fillna(-1).astype(int).values
        arr_f = pd.to_numeric(series_female, errors="coerce").fillna(-1).astype(int).values

        unclassified = {12, 18, -1}
        knowledge_domains = [
            {2, 4, 5, 10},
            {1, 3, 8, 9, 11, 13},
            {6, 7, 16},
            {14, 15, 17}
        ]

        n_samples = len(arr_m)
        similarities = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            fm = arr_m[i]
            ff = arr_f[i]

            if fm in unclassified or ff in unclassified:
                similarities[i] = 0.0
                continue
            # end if

            if fm == ff:
                similarities[i] = 1.0
                continue
            # end if

            matched_domain = False
            for domain_codes in knowledge_domains:
                if fm in domain_codes and ff in domain_codes:
                    similarities[i] = 0.5
                    matched_domain = True
                    break
                # end if
            # end for domain_codes

            if not matched_domain:
                similarities[i] = 0.0
            # end if
        # end for i

        return similarities
        # end def compute_field_similarity

    @staticmethod
    def zip_to_8_categories(zip_code: ty.Any) -> str:
        """Extracts the first digit of a US ZIP code and aggregates
        them into 8 broad geographic categories.
        Handles numeric, string, and dirty input formats (e.g., ZIP+4).
        """
        if pd.isna(zip_code):
            return "Unknown"
        # end if

        # Clean and zero-pad to handle missing leading zeros from numeric inputs
        zip_str = str(zip_code).strip().replace(",", "").split("-")[0].split(".")[0]
        if not zip_str:
            return "Unknown"
        # end if
        zip_str = zip_str.zfill(5)

        first_digit = zip_str[0]

        mapping = {
            "0": "Northeast",         # New England, NJ, PR
            "1": "Northeast",         # NY, PA, DE
            "2": "Mid-Atlantic",      # DC, MD, NC, SC, VA, WV
            "3": "Southeast",         # AL, FL, GA, MS, TN
            "4": "Southeast",         # IN, KY, MI, OH
            "5": "Northern Plains",   # IA, MN, MT, ND, SD, WI
            "6": "Central Plains",    # IL, KS, MO, NE
            "7": "South Central",     # AR, LA, OK, TX
            "8": "Mountain West",     # AZ, CO, ID, NM, NV, UT, WY
            "9": "Pacific West",      # AK, CA, HI, OR, WA
        }

        return mapping.get(first_digit, "Other")
        # end def zip_to_8_categories

    def compute_same_region(
        self,
        series_male: pd.Series,
        series_female: pd.Series
    ) -> np.ndarray:
        """Computes binary same-region indicator based on 8 geographic categories.

        Args:
            series_male: Male participant ZIP codes.
            series_female: Female participant ZIP codes.

        Returns:
            Binary float ndarray: 1.0 if converted region is identical, 0.0 otherwise.
        """
        reg_m = series_male.apply(self.zip_to_8_categories)
        reg_f = series_female.apply(self.zip_to_8_categories)
        return (reg_m == reg_f).astype(float).values
        # end def compute_same_region
# end class SpeedDatingFeatureEncoder

zip_to_8_categories = SpeedDatingFeatureEncoder.zip_to_8_categories
