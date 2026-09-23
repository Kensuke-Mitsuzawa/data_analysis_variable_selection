import os
import openpyxl
import numpy as np
import pandas as pd
from data_analysis_variable_selection.datasets.speed_dating.config import SpeedDatingPreprocessingConfig
from data_analysis_variable_selection.datasets.speed_dating.loader import SpeedDatingDataLoader
from data_analysis_variable_selection.datasets.speed_dating.cleaner import SpeedDatingDataCleaner
from data_analysis_variable_selection.datasets.speed_dating.pair_builder import SpeedDatingPairBuilder
from data_analysis_variable_selection.datasets.speed_dating.encoder import SpeedDatingFeatureEncoder
from data_analysis_variable_selection.datasets.speed_dating.splitter import SpeedDatingLabelSplitter
from data_analysis_variable_selection.datasets.speed_dating.preprocessor import SpeedDatingPreprocessor
from data_analysis_variable_selection.datasets.speed_dating.report_generator import SpeedDatingReportGenerator


def test_speed_dating_loader():
    loader = SpeedDatingDataLoader()
    df = loader.load_data_raw()

    assert not df.empty
    assert "iid" in df.columns
    assert "gender" in df.columns
    assert "pid" in df.columns
    assert "match" in df.columns
    assert "age" in df.columns
    assert "sports" in df.columns
    assert "attr3_1" in df.columns
    assert "attr1_1" in df.columns
    assert (df["match"] == 1).sum() > 0
    # end def test_speed_dating_loader


def test_speed_dating_cleaning_and_pairing():
    config = SpeedDatingPreprocessingConfig()
    cleaner = SpeedDatingDataCleaner()
    pair_builder = SpeedDatingPairBuilder()
    encoder = SpeedDatingFeatureEncoder()
    splitter = SpeedDatingLabelSplitter()

    # Create small test dataset
    df_raw = pd.DataFrame([
        {
            "iid": 1, "pid": 2, "gender": 1, "wave": 1, "match": 1, "dec": 1, "dec_o": 1,
            "age": 28, "race": 2, "imprace": 5, "imprelig": 4, "date": 3, "go_out": 2, "goal": 1, "field_cd": 1,
            "sports": 8, "dining": 7, "art": 6, "yoga": 5, "attr3_1": 8, "sinc3_1": 7, "intel3_1": 8, "fun3_1": 8, "amb3_1": 7,
            "attr1_1": 20.0, "sinc1_1": 20.0, "intel1_1": 20.0, "fun1_1": 20.0, "amb1_1": 10.0, "shar1_1": 10.0
        },
        {
            "iid": 2, "pid": 1, "gender": 0, "wave": 1, "match": 1, "dec": 1, "dec_o": 1,
            "age": 25, "race": 2, "imprace": 6, "imprelig": 5, "date": 4, "go_out": 3, "goal": 1, "field_cd": 2,
            "sports": 6, "dining": 9, "art": 8, "yoga": 9, "attr3_1": 7, "sinc3_1": 8, "intel3_1": 9, "fun3_1": 7, "amb3_1": 8,
            "attr1_1": 25.0, "sinc1_1": 15.0, "intel1_1": 20.0, "fun1_1": 20.0, "amb1_1": 10.0, "shar1_1": 10.0
        },
        {
            "iid": 3, "pid": 4, "gender": 1, "wave": 1, "match": 0, "dec": 0, "dec_o": 1,
            "age": 30, "race": 4, "imprace": 8, "imprelig": 7, "date": 2, "go_out": 1, "goal": 2, "field_cd": 3,
            "sports": 5, "dining": 6, "art": 4, "yoga": 3, "attr3_1": 6, "sinc3_1": 6, "intel3_1": 7, "fun3_1": 6, "amb3_1": 6,
            "attr1_1": 30.0, "sinc1_1": 10.0, "intel1_1": 20.0, "fun1_1": 20.0, "amb1_1": 10.0, "shar1_1": 10.0
        },
        {
            "iid": 4, "pid": 3, "gender": 0, "wave": 1, "match": 0, "dec": 1, "dec_o": 0,
            "age": 26, "race": 2, "imprace": 7, "imprelig": 6, "date": 5, "go_out": 4, "goal": 3, "field_cd": 3,
            "sports": 4, "dining": 8, "art": 7, "yoga": 8, "attr3_1": 8, "sinc3_1": 8, "intel3_1": 8, "fun3_1": 8, "amb3_1": 7,
            "attr1_1": 20.0, "sinc1_1": 20.0, "intel1_1": 20.0, "fun1_1": 20.0, "amb1_1": 10.0, "shar1_1": 10.0
        }
    ])

    # 1. Clean
    df_clean = cleaner.clean_dataset_survey_fields(df_raw, config)
    assert len(df_clean) == 4

    # 2. Pair
    df_pairs = pair_builder.build_pairs_reciprocal(df_clean)
    assert len(df_pairs) == 2
    assert "pair_id_hash" in df_pairs.columns

    # 3. Encode
    df_joint = encoder.encode_features_joint(df_pairs, config)
    assert "Age_Gap" in df_joint.columns
    assert "Same_Race" in df_joint.columns
    assert "Race_Preference_Conflict" in df_joint.columns
    assert "Male_age" in df_joint.columns
    assert "Female_age" in df_joint.columns

    # 4. Split
    df_x, df_y = splitter.split_samples_by_match(df_joint)
    assert len(df_x) == 1
    assert len(df_y) == 1
    assert "match" not in df_x.columns
    assert "match" not in df_y.columns
    # end def test_speed_dating_cleaning_and_pairing


def test_speed_dating_preprocessor_end_to_end():
    config = SpeedDatingPreprocessingConfig(max_records_per_distribution=50)
    preprocessor = SpeedDatingPreprocessor(config=config)
    container = preprocessor.prepare_two_sample_data(apply_subsampling=True)

    assert container.sample_matrix_x.ndim == 2
    assert container.sample_matrix_y.ndim == 2
    assert container.sample_matrix_x.shape[1] == container.sample_matrix_y.shape[1]
    assert len(container.name_features) == container.sample_matrix_x.shape[1]
    assert not np.isnan(container.sample_matrix_x).any()
    assert not np.isnan(container.sample_matrix_y).any()
    assert container.sample_matrix_x.shape[0] <= 50
    assert container.sample_matrix_y.shape[0] <= 50
    # end def test_speed_dating_preprocessor_end_to_end


def test_speed_dating_report_generator(tmp_path):
    output_dir = str(tmp_path / "reports")
    generator = SpeedDatingReportGenerator()
    artifacts = generator.generate_dataset_report(
        directory_output=output_dir,
        export_excel=True,
    )

    assert os.path.isfile(artifacts.path_report_markdown)
    assert artifacts.path_report_excel is not None
    assert os.path.isfile(artifacts.path_report_excel)

    with open(artifacts.path_report_markdown, "r", encoding="utf-8") as f:
        content = f.read()
    # end with
    assert "# Speed Dating Experiment" in content
    assert "Generated At" in content
    assert "Git Commit" in content
    assert "Dataset Source URL" in content
    assert "Demographics & Social Habits" in content
    assert "Leisure & Lifestyle Activity Ratings" in content
    assert "Self-Perception / Self-Concept Ratings" in content

    # Verify Excel sheets
    wb = openpyxl.load_workbook(artifacts.path_report_excel)
    expected_sheets = [
        "Overview",
        "Demographics & Habits",
        "Leisure Interests",
        "Self-Perception",
        "Mate Preferences",
        "Missingness Audit",
    ]
    for s in expected_sheets:
        assert s in wb.sheetnames
    # end for s
    # end def test_speed_dating_report_generator
