import os
import tempfile
import pytest
from pathlib import Path

from data_analysis_variable_selection.cli.cli_config import load_toml_config
from data_analysis_variable_selection.datasets.feature_tracker import FeatureOperationTracker, FeatureItemData
from data_analysis_variable_selection.datasets.setup_handler import DatasetSetupHandler
from data_analysis_variable_selection.datasets.speed_dating.config import SpeedDatingPreprocessingConfig
from data_analysis_variable_selection.datasets.speed_dating.preprocessor import SpeedDatingPreprocessor
from data_analysis_variable_selection.datasets.ames_housing.config import AmesPreprocessingConfig
from data_analysis_variable_selection.datasets.ames_housing.preprocessor import AmesHousingPreprocessor


def test_speed_dating_feature_list_extraction():
    """Verify speed dating feature list contains all 47 active features, removed features, and valid 3-column attributes."""
    cfg = load_toml_config("plans/config_speed_dating_mmd_cv.toml")
    preprocessor = SpeedDatingPreprocessor()
    tracker = FeatureOperationTracker()

    # 1. Verify active features only
    active_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg, include_removed=False)
    assert len(active_features) == 49
    feature_dict = {f.feature_processed: f for f in active_features}

    # Verify combined features are formatted as list expressions
    assert "Delta_Male_Attr_Align" in feature_dict
    assert feature_dict["Delta_Male_Attr_Align"].feature_original == "['attr3_1', 'attr1_1']"
    assert feature_dict["Delta_Male_Attr_Align"].type_feature == "float"

    assert "Delta_Female_Intel_Align" in feature_dict
    assert feature_dict["Delta_Female_Intel_Align"].feature_original == "['intel3_1', 'intel1_1']"
    assert feature_dict["Delta_Female_Intel_Align"].type_feature == "float"

    assert "Race_Preference_Conflict" in feature_dict
    assert feature_dict["Race_Preference_Conflict"].feature_original == "['race', 'imprace']"
    assert feature_dict["Race_Preference_Conflict"].type_feature == "float"

    assert "Interest_Cosine_Sim" in feature_dict
    assert feature_dict["Interest_Cosine_Sim"].feature_original.startswith("['")
    assert feature_dict["Interest_Cosine_Sim"].feature_original.endswith("']")
    assert feature_dict["Interest_Cosine_Sim"].type_feature == "float"

    # Verify career tiers and field similarity
    assert "career_group_male" in feature_dict
    assert feature_dict["career_group_male"].feature_original == "career_c"
    assert feature_dict["career_group_male"].type_feature == "int"

    assert "career_group_female" in feature_dict
    assert feature_dict["career_group_female"].feature_original == "career_c"
    assert feature_dict["career_group_female"].type_feature == "int"

    assert "Field_Similarity" in feature_dict
    assert feature_dict["Field_Similarity"].feature_original == "field_cd"
    assert feature_dict["Field_Similarity"].type_feature == "float"

    # Verify Same_region
    assert "Same_region" in feature_dict
    assert feature_dict["Same_region"].feature_original == "zipcode"
    assert feature_dict["Same_region"].type_feature == "category"

    # Verify diff features for age, interests, traits, and survey expectations
    assert "Diff_age" in feature_dict
    assert feature_dict["Diff_age"].feature_original == "age"
    assert feature_dict["Diff_age"].type_feature == "float"

    assert "Diff_imprace" in feature_dict
    assert feature_dict["Diff_imprace"].feature_original == "imprace"
    assert feature_dict["Diff_imprace"].type_feature == "float"

    assert "Diff_imprelig" in feature_dict
    assert feature_dict["Diff_imprelig"].feature_original == "imprelig"
    assert feature_dict["Diff_imprelig"].type_feature == "float"

    assert "Diff_art" in feature_dict
    assert feature_dict["Diff_art"].feature_original == "art"
    assert feature_dict["Diff_art"].type_feature == "float"

    assert "Diff_attr1_1" in feature_dict
    assert feature_dict["Diff_attr1_1"].feature_original == "attr1_1"
    assert feature_dict["Diff_attr1_1"].type_feature == "float"

    assert "Diff_attr3_1" in feature_dict
    assert feature_dict["Diff_attr3_1"].feature_original == "attr3_1"
    assert feature_dict["Diff_attr3_1"].type_feature == "float"

    assert "Diff_goal" in feature_dict
    assert feature_dict["Diff_goal"].feature_original == "goal"
    assert feature_dict["Diff_goal"].type_feature == "float"

    assert "Diff_date" in feature_dict
    assert feature_dict["Diff_date"].feature_original == "date"
    assert feature_dict["Diff_date"].type_feature == "float"

    assert "Diff_go_out" in feature_dict
    assert feature_dict["Diff_go_out"].feature_original == "go_out"
    assert feature_dict["Diff_go_out"].type_feature == "float"

    assert "Diff_exphappy" in feature_dict
    assert feature_dict["Diff_exphappy"].feature_original == "exphappy"
    assert feature_dict["Diff_exphappy"].type_feature == "float"

    assert "Diff_expnum" in feature_dict
    assert feature_dict["Diff_expnum"].feature_original == "expnum"
    assert feature_dict["Diff_expnum"].type_feature == "float"

    # Verify individual gender features were removed
    assert "Male_age" not in feature_dict
    assert "Female_age" not in feature_dict
    assert "Male_imprace" not in feature_dict
    assert "Female_imprace" not in feature_dict
    assert "Male_imprelig" not in feature_dict
    assert "Female_imprelig" not in feature_dict
    assert "Male_art" not in feature_dict
    assert "Female_art" not in feature_dict
    assert "Male_attr1_1" not in feature_dict
    assert "Female_attr1_1" not in feature_dict
    assert "Male_goal" not in feature_dict
    assert "Female_goal" not in feature_dict

    # Check valid types across all active features
    valid_active_types = {"int", "float", "category", "str"}
    for f in active_features:
        assert f.type_feature in valid_active_types, f"Invalid type {f.type_feature} for {f.feature_processed}"
    # end for

    # 2. Verify all features with include_removed=True (default)
    all_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg, include_removed=True)
    assert len(all_features) > 47
    removed_features = [f for f in all_features if f.feature_processed == "removed"]
    assert len(removed_features) > 150
    for rem in removed_features:
        assert rem.feature_processed == "removed"
        assert rem.type_feature == "removed"
    # end for
    removed_cols = {f.feature_original for f in removed_features}
    assert "dec" in removed_cols
    assert "id" in removed_cols
    assert "match" in removed_cols
# end def test_speed_dating_feature_list_extraction


def test_ames_housing_feature_list_extraction():
    """Verify Ames Housing feature list contains 243/244 features, removed features, and valid 3-column attributes."""
    cfg = load_toml_config("plans/config_ames_housing_mmd_cv.toml")
    preprocessor = AmesHousingPreprocessor()
    tracker = FeatureOperationTracker()

    # 1. Verify active features only
    active_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg, include_removed=False)
    assert len(active_features) in {243, 244}
    feature_dict = {f.feature_processed: f for f in active_features}

    # Verify LotFrontage combines LotFrontage and Neighborhood as list expression
    assert "LotFrontage" in feature_dict
    assert feature_dict["LotFrontage"].feature_original == "['LotFrontage', 'Neighborhood']"
    assert feature_dict["LotFrontage"].type_feature == "float"

    # Verify one-hot category feature
    assert "Neighborhood_CollgCr" in feature_dict
    assert feature_dict["Neighborhood_CollgCr"].feature_original == "Neighborhood"
    assert feature_dict["Neighborhood_CollgCr"].type_feature == "category"

    # Verify ordinal integer feature
    assert "OverallQual" in feature_dict
    assert feature_dict["OverallQual"].type_feature == "int"

    # Check valid types across all active features
    valid_active_types = {"int", "float", "category", "str"}
    for f in active_features:
        assert f.type_feature in valid_active_types, f"Invalid type {f.type_feature} for {f.feature_processed}"
    # end for

    # 2. Verify all features with include_removed=True (default)
    all_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg, include_removed=True)
    assert len(all_features) >= 247
    removed_features = [f for f in all_features if f.feature_processed == "removed"]
    assert len(removed_features) >= 4
    for rem in removed_features:
        assert rem.feature_processed == "removed"
        assert rem.type_feature == "removed"
    # end for
    removed_cols = {f.feature_original for f in removed_features}
    assert {"SalePrice", "Id", "YrSold", "MoSold"}.issubset(removed_cols)
# end def test_ames_housing_feature_list_extraction


def test_export_feature_list_markdown_speed_dating():
    """Verify export_feature_list_markdown produces valid markdown with the required 3 columns and removed notation."""
    cfg = load_toml_config("plans/config_speed_dating_mmd_cv.toml")
    preprocessor = SpeedDatingPreprocessor()
    tracker = FeatureOperationTracker()
    list_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_md = os.path.join(tmp_dir, "features.md")
        result_path = tracker.export_feature_list_markdown(
            list_features=list_features,
            config=cfg,
            path_output_markdown=path_md
        )

        assert os.path.exists(result_path)
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        # end with

        # Verify header metadata
        assert "Generation Date" in content
        assert "Git Commit ID" in content
        assert "Source URL" in content
        assert "kaggle.com" in content

        # Verify table header with exact 3 required columns
        assert "| processed feature | original feature | type of the processed feature |" in content
        assert "| :--- | :--- | :--- |" in content

        # Verify sample active diff rows
        assert "| Diff_age | age | float |" in content
        assert "| Diff_imprace | imprace | float |" in content
        assert "| Diff_imprelig | imprelig | float |" in content
        assert "| Diff_goal | goal | float |" in content
        assert "| Diff_date | date | float |" in content
        assert "| Diff_exphappy | exphappy | float |" in content
        assert "| Diff_expnum | expnum | float |" in content
        assert "| Delta_Female_Attr_Align | ['attr3_1', 'attr1_1'] | float |" in content
        assert "| Race_Preference_Conflict | ['race', 'imprace'] | float |" in content
        assert "| Same_Race | race | category |" in content
        assert "| Same_region | zipcode | category |" in content
        assert "| career_group_male | career_c | int |" in content
        assert "| career_group_female | career_c | int |" in content
        assert "| Field_Similarity | field_cd | float |" in content

        # Verify removed feature notation
        assert "| removed | match | removed |" in content
        assert "| removed | dec | removed |" in content
    # end with
# end def test_export_feature_list_markdown_speed_dating


def test_export_feature_list_markdown_ames_housing():
    """Verify export_feature_list_markdown for Ames Housing produces valid markdown with required 3 columns and removed notation."""
    cfg = load_toml_config("plans/config_ames_housing_mmd_cv.toml")
    preprocessor = AmesHousingPreprocessor()
    tracker = FeatureOperationTracker()
    list_features = tracker.track_features_dataset(preprocessor=preprocessor, config=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_md = os.path.join(tmp_dir, "features.md")
        result_path = tracker.export_feature_list_markdown(
            list_features=list_features,
            config=cfg,
            path_output_markdown=path_md
        )

        assert os.path.exists(result_path)
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        # end with

        # Verify header metadata
        assert "Generation Date" in content
        assert "Git Commit ID" in content
        assert "Source URL" in content

        # Verify table header with exact 3 required columns
        assert "| processed feature | original feature | type of the processed feature |" in content
        assert "| :--- | :--- | :--- |" in content

        # Verify sample active rows
        assert "| LotFrontage | ['LotFrontage', 'Neighborhood'] | float |" in content
        assert "| Neighborhood_CollgCr | Neighborhood | category |" in content
        assert "| OverallQual | OverallQual | int |" in content

        # Verify removed feature notation
        assert "| removed | SalePrice | removed |" in content
        assert "| removed | YrSold | removed |" in content
    # end with
# end def test_export_feature_list_markdown_ames_housing


def test_setup_handler_does_not_export_markdown():
    """Verify DatasetSetupHandler does not perform feature markdown export."""
    handler = DatasetSetupHandler()
    assert not hasattr(handler, "export_feature_list_markdown")
# end def test_setup_handler_does_not_export_markdown
