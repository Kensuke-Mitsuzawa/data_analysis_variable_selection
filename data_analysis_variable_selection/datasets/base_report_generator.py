import abc
import logging
import os
import typing as ty
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NumericMetricSummary(BaseModel):
    """Statistical summary comparing a numeric feature across two sample distributions.
    """
    name_metric: str = Field(description="Name of the numeric feature or metric.")
    mean_x: float = Field(description="Mean value in distribution X.")
    std_x: float = Field(description="Standard deviation in distribution X.")
    median_x: float = Field(description="Median value in distribution X.")
    min_x: float = Field(description="Minimum value in distribution X.")
    max_x: float = Field(description="Maximum value in distribution X.")
    mean_y: float = Field(description="Mean value in distribution Y.")
    std_y: float = Field(description="Standard deviation in distribution Y.")
    median_y: float = Field(description="Median value in distribution Y.")
    min_y: float = Field(description="Minimum value in distribution Y.")
    max_y: float = Field(description="Maximum value in distribution Y.")
    delta_mean: float = Field(description="Difference in means: mean_y - mean_x.")
    percentage_change_mean: ty.Optional[float] = Field(default=None, description="Percentage change in mean.")
# end class NumericMetricSummary


class CategoryProportionComparison(BaseModel):
    """Categorical value frequency and proportion comparison between two distributions.
    """
    name_category: str = Field(description="Category label or value representation.")
    count_x: int = Field(description="Frequency count in distribution X.")
    proportion_x: float = Field(description="Proportion in distribution X (0 to 1).")
    count_y: int = Field(description="Frequency count in distribution Y.")
    proportion_y: float = Field(description="Proportion in distribution Y (0 to 1).")
    delta_percentage_points: float = Field(description="Difference in percentage points: (proportion_y - proportion_x) * 100.")
# end class CategoryProportionComparison


class MissingnessSummary(BaseModel):
    """Missing value / Domain NA statistics for a dataset column.
    """
    name_column: str = Field(description="Column name.")
    count_missing_total: int = Field(description="Total missing count.")
    percentage_missing_total: float = Field(description="Percentage missing overall.")
    count_missing_x: int = Field(description="Missing count in distribution X.")
    percentage_missing_x: float = Field(description="Percentage missing in distribution X.")
    count_missing_y: int = Field(description="Missing count in distribution Y.")
    percentage_missing_y: float = Field(description="Percentage missing in distribution Y.")
# end class MissingnessSummary


class DatasetReportArtifacts(BaseModel):
    """Paths to generated dataset-specific report files.
    """
    path_report_markdown: str = Field(description="Path to generated Markdown report.")
    path_report_excel: ty.Optional[str] = Field(default=None, description="Path to generated Excel workbook.")
# end class DatasetReportArtifacts


class FeatureDerivationSummary(BaseModel):
    """Lineage and derivation specification for a preprocessed feature.
    """
    name_feature: str = Field(description="Name of the preprocessed feature in the dataset.")
    source_columns: ty.List[str] = Field(description="List of raw dataset column names from which the feature was derived.")
    process_description: str = Field(description="Brief 4-5 word description explaining the transformation process.")
# end class FeatureDerivationSummary


class BaseDatasetReportGenerator(abc.ABC):
    """Abstract base class for dataset-specific report generators producing shallow-level exploratory reports.
    """

    def __init__(self, name_dataset: str = "Dataset"):
        """Initializes the base report generator.

        Args:
            name_dataset: Descriptive human-readable dataset name.
        """
        self.name_dataset = name_dataset
        # end def __init__

    @abc.abstractmethod
    def generate_markdown_report(
        self,
        path_output_markdown: str,
        title_report: ty.Optional[str] = None,
        **kwargs: ty.Any
    ) -> str:
        """Synthesizes dataset-specific shallow statistics into a formatted Markdown report.

        Args:
            path_output_markdown: Destination file path.
            title_report: Optional custom report title.
            **kwargs: Dataset-specific data objects or paths.

        Returns:
            Path to generated Markdown report.
        """
        raise NotImplementedError()
        # end def generate_markdown_report

    def generate_excel_report(
        self,
        path_output_excel: str,
        **kwargs: ty.Any
    ) -> str:
        """Exports dataset-specific shallow statistics into an Excel workbook.

        Args:
            path_output_excel: Destination file path.
            **kwargs: Dataset-specific data objects or paths.

        Returns:
            Path to generated Excel workbook.
        """
        raise NotImplementedError("Excel export is not implemented for this dataset report generator.")
        # end def generate_excel_report

    def compute_numeric_metric_summary(
        self,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        list_columns: ty.List[str]
    ) -> ty.List[NumericMetricSummary]:
        """Calculates comparative summary statistics for numeric features between distributions X and Y.

        Args:
            df_x: DataFrame representing distribution X.
            df_y: DataFrame representing distribution Y.
            list_columns: List of numeric column names to analyze.

        Returns:
            List of NumericMetricSummary records.
        """
        list_summaries: ty.List[NumericMetricSummary] = []

        for col in list_columns:
            if col not in df_x.columns or col not in df_y.columns:
                continue
            # end if

            series_x = pd.to_numeric(df_x[col], errors="coerce").dropna()
            series_y = pd.to_numeric(df_y[col], errors="coerce").dropna()

            if series_x.empty or series_y.empty:
                continue
            # end if

            mean_x = float(series_x.mean())
            std_x = float(series_x.std(ddof=1)) if len(series_x) > 1 else 0.0
            median_x = float(series_x.median())
            min_x = float(series_x.min())
            max_x = float(series_x.max())

            mean_y = float(series_y.mean())
            std_y = float(series_y.std(ddof=1)) if len(series_y) > 1 else 0.0
            median_y = float(series_y.median())
            min_y = float(series_y.min())
            max_y = float(series_y.max())

            delta_mean = mean_y - mean_x
            pct_change = ((mean_y - mean_x) / abs(mean_x) * 100.0) if abs(mean_x) > 1e-9 else None

            list_summaries.append(
                NumericMetricSummary(
                    name_metric=col,
                    mean_x=mean_x,
                    std_x=std_x,
                    median_x=median_x,
                    min_x=min_x,
                    max_x=max_x,
                    mean_y=mean_y,
                    std_y=std_y,
                    median_y=median_y,
                    min_y=min_y,
                    max_y=max_y,
                    delta_mean=delta_mean,
                    percentage_change_mean=pct_change,
                )
            )
        # end for col

        return list_summaries
        # end def compute_numeric_metric_summary

    def compute_categorical_distribution_summary(
        self,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        column_name: str,
        top_n: int = 10
    ) -> ty.List[CategoryProportionComparison]:
        """Calculates categorical value counts and proportions across distributions X and Y.

        Args:
            df_x: DataFrame representing distribution X.
            df_y: DataFrame representing distribution Y.
            column_name: Categorical column name.
            top_n: Maximum number of top categories to compare.

        Returns:
            List of CategoryProportionComparison objects sorted by overall prevalence.
        """
        if column_name not in df_x.columns or column_name not in df_y.columns:
            return []
        # end if

        counts_x = df_x[column_name].fillna("Missing").value_counts()
        counts_y = df_y[column_name].fillna("Missing").value_counts()

        total_x = max(len(df_x), 1)
        total_y = max(len(df_y), 1)

        combined_categories = list(dict.fromkeys(list(counts_x.index) + list(counts_y.index)))
        # Sort by total count
        combined_categories.sort(
            key=lambda cat: (counts_x.get(cat, 0) + counts_y.get(cat, 0)),
            reverse=True
        )

        selected_categories = combined_categories[:top_n]
        list_comparisons: ty.List[CategoryProportionComparison] = []

        for cat in selected_categories:
            cnt_x = int(counts_x.get(cat, 0))
            prop_x = cnt_x / total_x
            cnt_y = int(counts_y.get(cat, 0))
            prop_y = cnt_y / total_y
            delta_pp = (prop_y - prop_x) * 100.0

            list_comparisons.append(
                CategoryProportionComparison(
                    name_category=str(cat),
                    count_x=cnt_x,
                    proportion_x=prop_x,
                    count_y=cnt_y,
                    proportion_y=prop_y,
                    delta_percentage_points=delta_pp,
                )
            )
        # end for cat

        return list_comparisons
        # end def compute_categorical_distribution_summary

    def compute_missingness_summary(
        self,
        df_raw: pd.DataFrame,
        df_x: ty.Optional[pd.DataFrame] = None,
        df_y: ty.Optional[pd.DataFrame] = None
    ) -> ty.List[MissingnessSummary]:
        """Calculates missing value frequencies and percentages overall and per distribution.

        Args:
            df_raw: Full raw DataFrame.
            df_x: Optional distribution X DataFrame.
            df_y: Optional distribution Y DataFrame.

        Returns:
            List of MissingnessSummary records sorted descending by total missing count.
        """
        total_rows = max(len(df_raw), 1)
        total_rows_x = len(df_x) if df_x is not None else 0
        total_rows_y = len(df_y) if df_y is not None else 0

        list_records: ty.List[MissingnessSummary] = []

        for col in df_raw.columns:
            missing_tot = int(df_raw[col].isna().sum())
            if missing_tot == 0:
                continue
            # end if

            pct_tot = (missing_tot / total_rows) * 100.0
            missing_x = int(df_x[col].isna().sum()) if df_x is not None and col in df_x.columns else 0
            pct_x = (missing_x / total_rows_x * 100.0) if total_rows_x > 0 else 0.0

            missing_y = int(df_y[col].isna().sum()) if df_y is not None and col in df_y.columns else 0
            pct_y = (missing_y / total_rows_y * 100.0) if total_rows_y > 0 else 0.0

            list_records.append(
                MissingnessSummary(
                    name_column=col,
                    count_missing_total=missing_tot,
                    percentage_missing_total=pct_tot,
                    count_missing_x=missing_x,
                    percentage_missing_x=pct_x,
                    count_missing_y=missing_y,
                    percentage_missing_y=pct_y,
                )
            )
        # end for col

        list_records.sort(key=lambda item: item.count_missing_total, reverse=True)
        return list_records
        # end def compute_missingness_summary
# end class BaseDatasetReportGenerator
