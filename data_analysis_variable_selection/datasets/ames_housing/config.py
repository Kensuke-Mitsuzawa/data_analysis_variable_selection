import typing as ty
from pydantic import BaseModel, Field


class AmesPreprocessingConfig(BaseModel):
    """Configuration options, column groupings, and categorical maps for Ames Housing preprocessing.
    """
    path_data_file: ty.Optional[str] = Field(default=None, description="Local path to Ames Housing CSV file")
    years_pre_crash: ty.List[int] = Field(default=[2006, 2007], description="Years defining distribution X")
    years_post_crash: ty.List[int] = Field(default=[2009, 2010], description="Years defining distribution Y")
    drop_transition_year_2008: bool = Field(default=True, description="Whether to drop 2008 crash transition epoch")
    columns_to_drop: ty.List[str] = Field(
        default=["YrSold", "MoSold", "SalePrice", "Order", "PID", "SaleType", "SaleCondition", "Id"],
        description="Target, identification, and temporal leakage columns to drop"
    )
    max_records_per_distribution: ty.Optional[int] = Field(
        default=None,
        description="Maximum number of records to sample per distribution for MMD variable selection to manage O(N^2) complexity."
    )
    random_seed_sampling: int = Field(
        default=42,
        description="Random seed for reproducible subsampling of records."
    )

    categorical_na_to_none_cols: ty.List[str] = Field(
        default=[
            "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"
        ],
        description="Categorical columns where NA denotes absence of physical feature"
    )

    continuous_na_to_zero_cols: ty.List[str] = Field(
        default=[
            "GarageArea", "GarageCars", "TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2",
            "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
        ],
        description="Continuous columns where NA denotes absence of physical feature"
    )

    ordinal_mapping_dicts: ty.Dict[str, ty.Dict[str, int]] = Field(
        default={
            "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "PoolQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
            "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
            "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
        },
        description="Monotonic integer mappings for ordinal quality and condition attributes"
    )
