import os
import logging
import typing as ty
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AmesHousingDataLoader:
    """Loads raw Ames Housing dataset records from local CSV or OpenML repository.
    """

    def load_data_raw(self, path_source: ty.Optional[str] = None) -> pd.DataFrame:
        """Loads raw housing dataframe from file or remote source.

        Args:
            path_source: Local filesystem path to Ames Housing CSV.

        Returns:
            Pandas DataFrame containing raw property sales.
        """
        if path_source is not None and os.path.isfile(path_source):
            logger.info(f"Loading Ames Housing dataset from local file: {path_source}")
            df_raw = pd.read_csv(path_source)
            return df_raw
        # end if

        # Fallback to OpenML
        try:
            from sklearn.datasets import fetch_openml
            logger.info("Attempting to load Ames Housing from OpenML (data_id=42165)...")
            dataset_openml = fetch_openml(data_id=42165, as_frame=True, parser="auto")
            df_raw = dataset_openml.frame
            return df_raw
        except Exception as exc:
            logger.warning(
                f"Failed to fetch from OpenML ({exc}). Generating representative synthetic Ames Housing data."
            )
            df_synthetic = self._generate_synthetic_ames_data(n_samples=200)
            return df_synthetic
        # end try
        # end def load_data_raw

    def _generate_synthetic_ames_data(self, n_samples: int = 200) -> pd.DataFrame:
        """Generates synthetic housing data matching the real schema for offline testing.
        """
        np.random.seed(42)
        years = np.random.choice([2006, 2007, 2008, 2009, 2010], size=n_samples)
        neighborhoods = np.random.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"], size=n_samples)
        qualities = np.random.choice(["Ex", "Gd", "TA", "Fa", "Po", None], size=n_samples)
        bldg_type = np.random.choice(["1Fam", "2fmCon", "Duplex", "Twnhs"], size=n_samples)

        dict_data = {
            "Order": np.arange(1, n_samples + 1),
            "PID": np.random.randint(500000, 999999, size=n_samples),
            "YrSold": years,
            "MoSold": np.random.randint(1, 13, size=n_samples),
            "SalePrice": np.random.randint(100000, 400000, size=n_samples),
            "LotFrontage": np.where(np.random.rand(n_samples) > 0.2, np.random.randint(30, 120, size=n_samples).astype(float), np.nan),
            "LotArea": np.random.randint(5000, 20000, size=n_samples),
            "GrLivArea": np.random.randint(800, 3000, size=n_samples),
            "TotalBsmtSF": np.random.randint(500, 2000, size=n_samples),
            "GarageArea": np.random.randint(0, 800, size=n_samples),
            "OverallQual": np.random.randint(3, 10, size=n_samples),
            "OverallCond": np.random.randint(3, 10, size=n_samples),
            "ExterQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], size=n_samples),
            "ExterCond": np.random.choice(["Ex", "Gd", "TA", "Fa"], size=n_samples),
            "BsmtQual": qualities,
            "BsmtCond": qualities,
            "HeatingQC": np.random.choice(["Ex", "Gd", "TA", "Fa"], size=n_samples),
            "KitchenQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], size=n_samples),
            "FireplaceQu": qualities,
            "GarageQual": qualities,
            "GarageCond": qualities,
            "PoolQC": np.random.choice(["Ex", "Gd", None], p=[0.05, 0.05, 0.9], size=n_samples),
            "BsmtExposure": np.random.choice(["Gd", "Av", "Mn", "No", None], size=n_samples),
            "BsmtFinType1": np.random.choice(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", None], size=n_samples),
            "BsmtFinType2": np.random.choice(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", None], size=n_samples),
            "Neighborhood": neighborhoods,
            "BldgType": bldg_type,
            "HouseStyle": np.random.choice(["1Story", "2Story", "1.5Fin"], size=n_samples),
            "Foundation": np.random.choice(["PConc", "CBlock", "BrkTil"], size=n_samples),
            "RoofStyle": np.random.choice(["Gable", "Hip", "Gambrel"], size=n_samples),
        }
        return pd.DataFrame(dict_data)
        # end def _generate_synthetic_ames_data
# end class AmesHousingDataLoader
