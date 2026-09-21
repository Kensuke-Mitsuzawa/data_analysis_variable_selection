import numpy as np
import pandas as pd
from data_analysis_variable_selection.datasets.ames_housing.config import AmesPreprocessingConfig
from data_analysis_variable_selection.datasets.ames_housing.cleaner import AmesHousingDataCleaner
from data_analysis_variable_selection.datasets.ames_housing.encoder import AmesHousingFeatureEncoder
from data_analysis_variable_selection.datasets.ames_housing.splitter import AmesHousingTemporalSplitter
from data_analysis_variable_selection.datasets.ames_housing.preprocessor import AmesHousingPreprocessor


def test_ames_housing_adapter():
    config = AmesPreprocessingConfig()
    cleaner = AmesHousingDataCleaner()
    encoder = AmesHousingFeatureEncoder()
    splitter = AmesHousingTemporalSplitter()

    # Create dummy Ames dataframe
    df_raw = pd.DataFrame({
        "Order": [1, 2, 3, 4],
        "PID": [101, 102, 103, 104],
        "YrSold": [2006, 2007, 2008, 2009],
        "MoSold": [5, 6, 7, 8],
        "SalePrice": [200000, 250000, 180000, 220000],
        "PoolQC": [np.nan, "Ex", np.nan, "Gd"],
        "GarageArea": [np.nan, 400.0, 500.0, np.nan],
        "Neighborhood": ["CollgCr", "CollgCr", "Veenker", "Veenker"],
        "LotFrontage": [60.0, np.nan, 80.0, np.nan],
        "KitchenQual": ["TA", "Gd", "Ex", "Fa"],
        "HouseStyle": ["1Story", "2Story", "1Story", "2Story"],
    })

    # 1. Clean Domain NAs
    df_clean = cleaner.clean_dataset_domain_nas(df_raw, config)
    assert df_clean["PoolQC"].iloc[0] == "None"
    assert df_clean["PoolQC"].iloc[1] == "Ex"
    assert df_clean["GarageArea"].iloc[0] == 0.0

    # 2. Neighborhood Imputation
    df_imputed = cleaner.impute_missing_values_neighborhood(df_clean)
    assert df_imputed["LotFrontage"].iloc[1] == 60.0  # CollgCr median
    assert df_imputed["LotFrontage"].iloc[3] == 80.0  # Veenker median

    # 3. Ordinal Encoding
    df_ordinal = encoder.encode_features_ordinal(df_imputed, config.ordinal_mapping_dicts)
    assert df_ordinal["KitchenQual"].iloc[0] == 3.0  # TA
    assert df_ordinal["KitchenQual"].iloc[1] == 4.0  # Gd
    assert df_ordinal["KitchenQual"].iloc[2] == 5.0  # Ex
    assert df_ordinal["KitchenQual"].iloc[3] == 2.0  # Fa

    # 4. One-Hot Encoding
    df_onehot = encoder.encode_features_nominal_onehot(df_ordinal)
    assert "HouseStyle_1Story" in df_onehot.columns
    assert "HouseStyle_2Story" in df_onehot.columns

    # 5. Temporal Splitting
    df_x, df_y = splitter.split_samples_temporal(df_onehot, config)
    assert len(df_x) == 2  # 2006, 2007
    assert len(df_y) == 1  # 2009 (2008 dropped)
    assert "YrSold" not in df_x.columns
    assert "SalePrice" not in df_x.columns
    assert "Order" not in df_x.columns


def test_ames_preprocessor_end_to_end():
    preprocessor = AmesHousingPreprocessor()
    container = preprocessor.prepare_two_sample_data()

    assert container.sample_matrix_x.ndim == 2
    assert container.sample_matrix_y.ndim == 2
    assert container.sample_matrix_x.shape[1] == container.sample_matrix_y.shape[1]
    assert len(container.name_features) == container.sample_matrix_x.shape[1]

    # Assert no NaNs anywhere in matrices
    assert not np.isnan(container.sample_matrix_x).any()
    assert not np.isnan(container.sample_matrix_y).any()


def test_ames_preprocessor_record_limit():
    config = AmesPreprocessingConfig(max_records_per_distribution=25, random_seed_sampling=123)
    preprocessor = AmesHousingPreprocessor(config=config)
    container = preprocessor.prepare_two_sample_data()

    assert container.sample_matrix_x.shape[0] <= 25
    assert container.sample_matrix_y.shape[0] <= 25
    assert container.metadata_dataset["max_records_per_distribution"] == 25
    assert container.metadata_dataset["n_samples_x_original"] >= container.sample_matrix_x.shape[0]

    # Test dynamic override via parameter
    container_override = preprocessor.prepare_two_sample_data(max_records_per_distribution=10)
    assert container_override.sample_matrix_x.shape[0] <= 10
    assert container_override.sample_matrix_y.shape[0] <= 10
    assert container_override.metadata_dataset["max_records_per_distribution"] == 10

