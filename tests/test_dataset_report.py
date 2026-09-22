import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from data_analysis_variable_selection.datasets.base_report_generator import (
    BaseDatasetReportGenerator,
    NumericMetricSummary,
    CategoryProportionComparison,
    MissingnessSummary,
    DatasetReportArtifacts,
)
from data_analysis_variable_selection.datasets.ames_housing.report_generator import (
    AmesHousingReportGenerator,
)
from data_analysis_variable_selection.datasets.ames_housing.config import (
    AmesPreprocessingConfig,
)


class ConcreteDatasetReportGenerator(BaseDatasetReportGenerator):
    """Concrete test subclass of BaseDatasetReportGenerator."""
    def generate_markdown_report(
        self,
        path_output_markdown: str,
        title_report: str = "Test Report",
        **kwargs
    ) -> str:
        with open(path_output_markdown, "w", encoding="utf-8") as f:
            f.write(f"# {title_report}\n")
        # end with
        return path_output_markdown
        # end def generate_markdown_report
# end class ConcreteDatasetReportGenerator


def test_base_report_generator_statistics():
    """Verify generic helper calculations in BaseDatasetReportGenerator."""
    generator = ConcreteDatasetReportGenerator(name_dataset="Test Mock Dataset")

    df_x = pd.DataFrame({
        "Price": [100.0, 200.0, 300.0],
        "Category": ["A", "A", "B"],
        "MissingCol": [1.0, np.nan, 3.0],
    })
    df_y = pd.DataFrame({
        "Price": [150.0, 250.0, 350.0],
        "Category": ["A", "B", "B"],
        "MissingCol": [np.nan, np.nan, 3.0],
    })

    # Numeric summary
    num_summaries = generator.compute_numeric_metric_summary(df_x, df_y, ["Price"])
    assert len(num_summaries) == 1
    price_sum = num_summaries[0]
    assert price_sum.mean_x == 200.0
    assert price_sum.mean_y == 250.0
    assert price_sum.delta_mean == 50.0
    assert price_sum.percentage_change_mean == 25.0

    # Categorical summary
    cat_summaries = generator.compute_categorical_distribution_summary(df_x, df_y, "Category")
    assert len(cat_summaries) == 2
    cat_a = next(c for c in cat_summaries if c.name_category == "A")
    assert cat_a.count_x == 2
    assert cat_a.count_y == 1
    assert cat_a.delta_percentage_points < 0

    # Missingness summary
    df_all = pd.concat([df_x, df_y], ignore_index=True)
    miss_summaries = generator.compute_missingness_summary(df_all, df_x, df_y)
    assert len(miss_summaries) == 1
    assert miss_summaries[0].name_column == "MissingCol"
    assert miss_summaries[0].count_missing_total == 3
# end def test_base_report_generator_statistics


def test_ames_housing_report_generator_end_to_end():
    """Verify AmesHousingReportGenerator produces Markdown and Excel reports with expected sections."""
    config = AmesPreprocessingConfig()
    generator = AmesHousingReportGenerator(config=config)

    # Dummy raw Ames DataFrame
    df_raw = pd.DataFrame({
        "Order": [1, 2, 3, 4, 5],
        "PID": [101, 102, 103, 104, 105],
        "YrSold": [2006, 2007, 2008, 2009, 2010],
        "MoSold": [5, 6, 7, 8, 9],
        "SalePrice": [200000.0, 250000.0, 180000.0, 210000.0, 230000.0],
        "GrLivArea": [1500.0, 1800.0, 1400.0, 1600.0, 1750.0],
        "LotArea": [8000.0, 9500.0, 8200.0, 8900.0, 9100.0],
        "TotalBsmtSF": [800.0, 1000.0, 750.0, 850.0, 950.0],
        "GarageArea": [np.nan, 450.0, 400.0, 500.0, np.nan],
        "YearBuilt": [1995, 2002, 1980, 2005, 2008],
        "OverallQual": [7, 8, 5, 7, 8],
        "OverallCond": [5, 5, 6, 5, 6],
        "Neighborhood": ["CollgCr", "CollgCr", "OldTown", "Edwards", "Edwards"],
        "BldgType": ["1Fam", "1Fam", "Twnhs", "1Fam", "1Fam"],
        "SaleType": ["WD", "WD", "COD", "WD", "New"],
        "PoolQC": [np.nan, "Ex", np.nan, np.nan, "Gd"],
        "FireplaceQu": ["TA", np.nan, np.nan, "Gd", "Ex"],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts = generator.generate_dataset_report(
            directory_output=tmp_dir,
            df_raw=df_raw,
            title_report="Ames Housing Test Report",
            export_excel=True,
        )

        assert isinstance(artifacts, DatasetReportArtifacts)
        assert os.path.exists(artifacts.path_report_markdown)
        assert artifacts.path_report_excel is not None
        assert os.path.exists(artifacts.path_report_excel)

        # Inspect Markdown contents
        with open(artifacts.path_report_markdown, "r", encoding="utf-8") as f:
            md_text = f.read()
        # end with

        assert "Ames Housing Test Report" in md_text
        assert "Sample Partitioning & Temporal Split" in md_text
        assert "SalePrice" in md_text
        assert "GrLivArea" in md_text
        assert "OverallQual" in md_text
        assert "Neighborhood" in md_text
        assert "Domain NAs & Amenity Absence Rates" in md_text

        # Inspect Excel sheets
        excel_file = pd.ExcelFile(artifacts.path_report_excel)
        sheet_names = excel_file.sheet_names
        assert "Overview" in sheet_names
        assert "Financial & Physical Metrics" in sheet_names
        assert "Quality Ratings" in sheet_names
        assert "Neighborhood Distribution" in sheet_names
        assert "Domain NAs & Missing" in sheet_names

        df_overview = excel_file.parse("Overview")
        assert len(df_overview) >= 5
    # end with
# end def test_ames_housing_report_generator_end_to_end


def test_report_synthesizer_template_rendering():
    """Verify ReportSynthesizer populates placeholders in base_report.md template."""
    from data_analysis_variable_selection.database.manager import DuckDBStorageManager
    from data_analysis_variable_selection.export.report_synthesizer import ReportSynthesizer
    from data_analysis_variable_selection.common.models import (
        VariableSelectionResult,
        VariableClusteringResult,
        ClusterMembership,
        PrototypeSampleResult,
        PrototypeSampleRecord,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test.duckdb")
        db = DuckDBStorageManager(db_path)
        db.initialize_database_schema()

        # Insert test data
        db.insert_records_selection(
            VariableSelectionResult(
                indices_selected=[0, 1],
                names_selected=["FeatureA", "FeatureB"],
                weights_selected=[1.0, 0.8],
            )
        )
        db.insert_records_clustering(
            VariableClusteringResult(
                list_memberships=[
                    ClusterMembership(id_variable=0, name_variable="FeatureA", id_cluster=1, score_related=1.0),
                    ClusterMembership(id_variable=1, name_variable="FeatureB", id_cluster=1, score_related=0.8),
                ],
                dict_cluster_to_variables={1: [0, 1]},
                indices_augmented_s_tilde=[0, 1],
                names_augmented_s_tilde=["FeatureA", "FeatureB"],
                dict_anchor_to_cluster={0: 1, 1: 1},
            )
        )
        db.insert_records_prototypes(
            PrototypeSampleResult(
                list_prototype_records=[
                    PrototypeSampleRecord(
                        id_sample=12,
                        label_class="X",
                        is_prototype_for="X",
                        distance_score=0.1234,
                        type_subspace="hat_S",
                        dict_feature_values={"FeatureA": 1.0},
                    )
                ]
            )
        )

        synthesizer = ReportSynthesizer()
        path_md = os.path.join(tmp_dir, "test_report.md")
        synthesizer.generate_markdown_report(
            db_manager=db,
            path_output_markdown=path_md,
            title_report="Synthesizer Template Test",
            path_dataset_report="dataset_report.md",
        )

        assert os.path.exists(path_md)
        with open(path_md, "r", encoding="utf-8") as f:
            content = f.read()
        # end with

        assert "# Synthesizer Template Test" in content
        assert "FeatureA" in content
        assert "FeatureB" in content
        assert "Cluster 1" in content
        assert "#12" in content
        assert "dataset_report.md" in content

        db.close_connection_database()
    # end with
# end def test_report_synthesizer_template_rendering
