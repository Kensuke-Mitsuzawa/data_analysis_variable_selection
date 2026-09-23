import typing as ty
import pandas as pd


class AmesHousingFeatureEncoder:
    """Encodes ordinal qualities monotonically to integers and expands nominal categories to one-hot dummies.
    """

    def encode_features_ordinal(
        self,
        df_data: pd.DataFrame,
        mapping_dicts: ty.Dict[str, ty.Dict[str, int]]
    ) -> pd.DataFrame:
        """Applies monotonic integer scales to ordinal quality and condition columns.

        Args:
            df_data: Input DataFrame with cleaned columns.
            mapping_dicts: Mapping dictionary for each ordinal column.

        Returns:
            DataFrame with specified ordinal columns mapped to integer values.
        """
        df_encoded = df_data.copy()

        for col_name, dict_map in mapping_dicts.items():
            if col_name in df_encoded.columns:
                df_encoded[col_name] = df_encoded[col_name].map(dict_map).fillna(0).astype(float)
            # end if
        # end for col_name

        return df_encoded
        # end def encode_features_ordinal

    def encode_features_nominal_onehot(
        self,
        df_data: pd.DataFrame,
        columns_nominal: ty.Optional[ty.List[str]] = None,
        return_lineage: bool = False
    ) -> ty.Union[pd.DataFrame, ty.Tuple[pd.DataFrame, ty.Dict[str, ty.List[str]]]]:
        """One-hot encodes nominal categorical columns into binary indicators.

        Args:
            df_data: Input DataFrame with numerical and categorical features.
            columns_nominal: Explicit list of nominal columns to encode, or None to auto-detect object/category columns.
            return_lineage: If True, also returns dictionary mapping original column to dummy columns.

        Returns:
            High-dimensional DataFrame with one-hot encoded dummy indicators (and lineage dict if requested).
        """
        if columns_nominal is None:
            # Auto-detect all string/object/category columns
            cols_to_encode = df_data.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cols_to_encode = [c for c in columns_nominal if c in df_data.columns]
        # end if

        if not cols_to_encode:
            return (df_data.copy(), {}) if return_lineage else df_data.copy()
        # end if

        dict_lineage: ty.Dict[str, ty.List[str]] = {}
        for col in cols_to_encode:
            categories = df_data[col].dropna().unique()
            # Sort categories for consistent column order
            categories_sorted = sorted([str(c) for c in categories])
            dict_lineage[col] = [f"{col}_{cat}" for cat in categories_sorted]
        # end for col

        df_encoded = pd.get_dummies(
            df_data,
            columns=cols_to_encode,
            drop_first=False,
            dtype=float
        )

        if return_lineage:
            # Reconcile exact columns present in df_encoded
            for col in cols_to_encode:
                dict_lineage[col] = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
            # end for col
            return df_encoded, dict_lineage
        # end if

        return df_encoded
        # end def encode_features_nominal_onehot
# end class AmesHousingFeatureEncoder
