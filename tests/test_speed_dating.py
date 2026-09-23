import os
import openpyxl
import pandas as pd
from data_analysis_variable_selection.datasets.speed_dating.config import SpeedDatingPreprocessingConfig
from data_analysis_variable_selection.datasets.speed_dating.loader import SpeedDatingDataLoader
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
