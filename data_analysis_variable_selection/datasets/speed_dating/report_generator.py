import logging
import os
import typing as ty
import numpy as np
import pandas as pd

from ..base_report_generator import (
    BaseDatasetReportGenerator,
    NumericMetricSummary,
    CategoryProportionComparison,
    MissingnessSummary,
    DatasetReportArtifacts,
)
from .config import SpeedDatingPreprocessingConfig
from .loader import SpeedDatingDataLoader

logger = logging.getLogger(__name__)


class SpeedDatingReportGenerator(BaseDatasetReportGenerator):
    """Generates exploratory shallow-level dataset reports tailored for the Speed Dating Experiment dataset.
    """

    def __init__(
        self,
        config: ty.Optional[SpeedDatingPreprocessingConfig] = None
    ):
        """Initializes the Speed Dating report generator.

        Args:
            config: Optional SpeedDatingPreprocessingConfig instance.
        """
        super().__init__(name_dataset="Speed Dating Experiment")
        self.config = config or SpeedDatingPreprocessingConfig()
        self.loader = SpeedDatingDataLoader()
        # end def __init__

    def generate_dataset_report(
        self,
        directory_output: str,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        title_report: ty.Optional[str] = None,
        export_excel: bool = True
    ) -> DatasetReportArtifacts:
        """Coordinates the end-to-end generation of Speed Dating dataset reports.

        Args:
            directory_output: Output directory where report files are stored.
            path_raw_data: Optional file path to raw CSV dataset.
            df_raw: Optional loaded raw DataFrame.
            title_report: Optional display title for the report.
            export_excel: Whether to also generate dataset_report.xlsx.

        Returns:
            DatasetReportArtifacts holding paths to created files.
        """
        os.makedirs(directory_output, exist_ok=True)
        path_md = os.path.join(directory_output, "dataset_report.md")
        self.generate_markdown_report(
            path_output_markdown=path_md,
            title_report=title_report,
            path_raw_data=path_raw_data,
            df_raw=df_raw
        )

        path_excel: ty.Optional[str] = None
        if export_excel:
            path_excel = os.path.join(directory_output, "dataset_report.xlsx")
            self.generate_excel_report(
                path_output_excel=path_excel,
                path_raw_data=path_raw_data,
                df_raw=df_raw
            )
        # end if

        return DatasetReportArtifacts(
            path_report_markdown=path_md,
            path_report_excel=path_excel
        )
        # end def generate_dataset_report

    def generate_markdown_report(
        self,
        path_output_markdown: str,
        title_report: ty.Optional[str] = None,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        **kwargs: ty.Any
    ) -> str:
        """Synthesizes Speed Dating dataset shallow statistics into a structured Markdown document.

        Args:
            path_output_markdown: Filepath for the generated Markdown report.
            title_report: Display title for the report.
            path_raw_data: Optional file path to raw CSV.
            df_raw: Optional pre-loaded raw DataFrame.
            **kwargs: Additional parameters.

        Returns:
            Path to the created Markdown report.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_markdown)), exist_ok=True)
        report_title = title_report or "Speed Dating Experiment Exploratory & Statistical Report"

        df_data = self._resolve_raw_dataframe(path_raw_data=path_raw_data, df_raw=df_raw)
        df_x, df_y = self._partition_raw_dataframe(df_data)

        # 1. Compute shallow statistics
        list_habits_cols = ["age", "imprace", "imprelig", "date", "go_out"]
        habits_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=list_habits_cols
        )

        interest_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_interests
        )

        self_rating_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_self_ratings
        )

        preference_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_stated_preferences
        )

        goal_comparison = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="goal",
            top_n=7
        )

        race_comparison = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="race",
            top_n=6
        )

        missingness_summaries = self.compute_missingness_summary(
            df_raw=df_data,
            df_x=df_x,
            df_y=df_y
        )

        # 2. Build Markdown content
        n_total = len(df_data)
        n_x = len(df_x)
        n_y = len(df_y)
        pct_x = (n_x / max(n_total, 1)) * 100.0
        pct_y = (n_y / max(n_total, 1)) * 100.0

        git_commit_id = self.get_git_commit_id()
        report_timestamp = self.get_generation_timestamp()
        source_url = self.config.url_download or "https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment"

        lines: ty.List[str] = [
            f"# {report_title}",
            "",
            "> **Dataset Overview**: The Columbia Business School Speed Dating Experiment dataset captures individual date scorecards across 21 experimental waves. This report provides baseline domain exploratory data analysis, shallow statistical distributions, and empirical shifts between **Mutually Matched Date Pairs ($X$: `match = 1`)** and **Unmatched Date Pairs ($Y$: `match = 0`)**.",
            "",
            "| Report Metadata | Description |",
            "| :--- | :--- |",
            f"| **Generated At** | {report_timestamp} |",
            f"| **Git Commit** | `{git_commit_id}` |",
            f"| **Dataset Source URL** | [{source_url}]({source_url}) |",
            "",
            "---",
            "",
            "## 1. Sample Partitioning & Two-Sample Definition",
            "",
            f"- **Total Raw Observations**: {n_total:,} scorecards",
            f"- **Distribution $X$ (Mutual Matches, `match = 1`)**: {n_x:,} records ({pct_x:.1f}%)",
            f"- **Distribution $Y$ (No Mutual Match, `match = 0`)**: {n_y:,} records ({pct_y:.1f}%)",
            f"- **Raw Feature Count**: {len(df_data.columns)} columns",
            "",
            "---",
            "",
            "## 2. Demographics & Social Habits",
            "",
            "Comparative central tendencies and spreads between Matched ($X$) and Unmatched ($Y$) encounters:",
            "",
            "| Feature / Metric | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | % Shift | Median ($X$) | Median ($Y$) | Min / Max ($X$) | Min / Max ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for num in habits_summaries:
            pct_str = f"{num.percentage_change_mean:+.2f}%" if num.percentage_change_mean is not None else "N/A"
            delta_str = f"{num.delta_mean:+.2f}"
            lines.append(
                f"| **{num.name_metric}** | {num.mean_x:,.2f} | {num.mean_y:,.2f} | {delta_str} | {pct_str} | "
                f"{num.median_x:,.1f} | {num.median_y:,.1f} | [{num.min_x:,.0f}, {num.max_x:,.0f}] | [{num.min_y:,.0f}, {num.max_y:,.0f}] |"
            )
        # end for num

        lines.extend([
            "",
            "---",
            "",
            "## 3. Leisure & Lifestyle Activity Ratings (1–10 Scale)",
            "",
            "Self-reported participation and interest across 17 leisure domains:",
            "",
            "| Activity Interest | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | Median ($X$) | Median ($Y$) | Std ($X$) | Std ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for inter in interest_summaries:
            lines.append(
                f"| **{inter.name_metric}** | {inter.mean_x:.2f} | {inter.mean_y:.2f} | {inter.delta_mean:+.2f} | "
                f"{inter.median_x:.1f} | {inter.median_y:.1f} | {inter.std_x:.2f} | {inter.std_y:.2f} |"
            )
        # end for inter

        lines.extend([
            "",
            "---",
            "",
            "## 4. Self-Perception / Self-Concept Ratings (1–10 Scale)",
            "",
            "Self-evaluations on physical attractiveness, sincerity, intelligence, fun, and ambition:",
            "",
            "| Trait Self-Rating | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | Median ($X$) | Median ($Y$) | Std ($X$) | Std ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for s_rate in self_rating_summaries:
            lines.append(
                f"| **{s_rate.name_metric}** | {s_rate.mean_x:.2f} | {s_rate.mean_y:.2f} | {s_rate.delta_mean:+.2f} | "
                f"{s_rate.median_x:.1f} | {s_rate.median_y:.1f} | {s_rate.std_x:.2f} | {s_rate.std_y:.2f} |"
            )
        # end for s_rate

        lines.extend([
            "",
            "---",
            "",
            "## 5. Stated Mate Preferences (Pre-Date Survey 1)",
            "",
            "Importance weights assigned to traits in prospective romantic partners:",
            "",
            "| Stated Preference | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | Median ($X$) | Median ($Y$) | Std ($X$) | Std ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for pref in preference_summaries:
            lines.append(
                f"| **{pref.name_metric}** | {pref.mean_x:.2f} | {pref.mean_y:.2f} | {pref.delta_mean:+.2f} | "
                f"{pref.median_x:.1f} | {pref.median_y:.1f} | {pref.std_x:.2f} | {pref.std_y:.2f} |"
            )
        # end for pref

        lines.extend([
            "",
            "---",
            "",
            "## 6. Categorical Distributions (Motivation & Ethnicity)",
            "",
            "### 6.1. Primary Dating Goal (`goal`)",
            "",
            "| Goal Category | Count ($X$) | Share ($X$) | Count ($Y$) | Share ($Y$) | Shift ($\Delta$ pp) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        goal_descriptions = {
            "1": "Fun night out (1)",
            "2": "Meet people (2)",
            "3": "Get a date (3)",
            "4": "Serious relationship (4)",
            "5": "To say I did it (5)",
            "6": "Other (6)",
        }

        for g in goal_comparison:
            desc = goal_descriptions.get(str(g.name_category).split(".")[0], str(g.name_category))
            lines.append(
                f"| {desc} | {g.count_x:,} | {g.proportion_x * 100.0:.1f}% | {g.count_y:,} | {g.proportion_y * 100.0:.1f}% | {g.delta_percentage_points:+.2f} pp |"
            )
        # end for g

        lines.extend([
            "",
            "### 6.2. Participant Ethnicity (`race`)",
            "",
            "| Ethnicity Code | Count ($X$) | Share ($X$) | Count ($Y$) | Share ($Y$) | Shift ($\Delta$ pp) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        race_descriptions = {
            "1": "Black/African American (1)",
            "2": "European/Caucasian (2)",
            "3": "Latino/Hispanic (3)",
            "4": "Asian/Pacific Islander (4)",
            "5": "Native American (5)",
            "6": "Other (6)",
        }

        for r in race_comparison:
            desc = race_descriptions.get(str(r.name_category).split(".")[0], str(r.name_category))
            lines.append(
                f"| {desc} | {r.count_x:,} | {r.proportion_x * 100.0:.1f}% | {r.count_y:,} | {r.proportion_y * 100.0:.1f}% | {r.delta_percentage_points:+.2f} pp |"
            )
        # end for r

        lines.extend([
            "",
            "---",
            "",
            "## 7. Missing Value & Survey Completeness Audit",
            "",
            "Top features exhibiting missing values in the raw scorecard data:",
            "",
            "| Column Name | Missing (Total) | Missing % (Total) | Missing ($X$) | Missing % ($X$) | Missing ($Y$) | Missing % ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for m in missingness_summaries[:15]:
            lines.append(
                f"| `{m.name_column}` | {m.count_missing_total:,} | {m.percentage_missing_total:.1f}% | "
                f"{m.count_missing_x:,} | {m.percentage_missing_x:.1f}% | {m.count_missing_y:,} | {m.percentage_missing_y:.1f}% |"
            )
        # end for m

        report_content = "\n".join(lines) + "\n"
        with open(path_output_markdown, "w", encoding="utf-8") as f_out:
            f_out.write(report_content)
        # end with

        logger.info(f"Speed Dating Markdown report generated at: {path_output_markdown}")
        return path_output_markdown
        # end def generate_markdown_report

    def generate_excel_report(
        self,
        path_output_excel: str,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        **kwargs: ty.Any
    ) -> str:
        """Exports Speed Dating shallow statistics into a multi-tab Excel workbook.

        Args:
            path_output_excel: Destination filepath for the Excel workbook.
            path_raw_data: Optional file path to raw CSV.
            df_raw: Optional pre-loaded DataFrame.
            **kwargs: Additional parameters.

        Returns:
            Path to created Excel workbook.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_excel)), exist_ok=True)
        df_data = self._resolve_raw_dataframe(path_raw_data=path_raw_data, df_raw=df_raw)
        df_x, df_y = self._partition_raw_dataframe(df_data)

        # Overview Sheet
        git_commit_id = self.get_git_commit_id()
        report_timestamp = self.get_generation_timestamp()
        source_url = self.config.url_download or "https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment"

        df_overview = pd.DataFrame([
            {"Metric": "Dataset Name", "Value": "Columbia Speed Dating Experiment"},
            {"Metric": "Report Generation Date", "Value": report_timestamp},
            {"Metric": "Git Commit ID", "Value": git_commit_id},
            {"Metric": "Dataset Source URL", "Value": source_url},
            {"Metric": "Total Raw Observations", "Value": len(df_data)},
            {"Metric": "Mutual Matches (X: match = 1)", "Value": len(df_x)},
            {"Metric": "Unmatched (Y: match = 0)", "Value": len(df_y)},
            {"Metric": "Match Rate", "Value": f"{(len(df_x) / max(len(df_data), 1) * 100.0):.2f}%"},
            {"Metric": "Raw Columns", "Value": len(df_data.columns)},
        ])

        # Dating Habits & Demographics Sheet
        habits_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=["age", "imprace", "imprelig", "date", "go_out"]
        )
        df_habits = pd.DataFrame([s.model_dump() for s in habits_summaries])

        # Leisure Interests Sheet
        interest_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_interests
        )
        df_interests = pd.DataFrame([s.model_dump() for s in interest_summaries])

        # Self-Perception Sheet
        self_rating_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_self_ratings
        )
        df_self_ratings = pd.DataFrame([s.model_dump() for s in self_rating_summaries])

        # Stated Preferences Sheet
        preference_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=self.config.columns_stated_preferences
        )
        df_preferences = pd.DataFrame([s.model_dump() for s in preference_summaries])

        # Missingness Sheet
        missingness_summaries = self.compute_missingness_summary(
            df_raw=df_data,
            df_x=df_x,
            df_y=df_y
        )
        df_missing = pd.DataFrame([s.model_dump() for s in missingness_summaries])

        with pd.ExcelWriter(path_output_excel, engine="openpyxl") as writer:
            df_overview.to_excel(writer, sheet_name="Overview", index=False)
            if not df_habits.empty:
                df_habits.to_excel(writer, sheet_name="Demographics & Habits", index=False)
            # end if
            if not df_interests.empty:
                df_interests.to_excel(writer, sheet_name="Leisure Interests", index=False)
            # end if
            if not df_self_ratings.empty:
                df_self_ratings.to_excel(writer, sheet_name="Self-Perception", index=False)
            # end if
            if not df_preferences.empty:
                df_preferences.to_excel(writer, sheet_name="Mate Preferences", index=False)
            # end if
            if not df_missing.empty:
                df_missing.to_excel(writer, sheet_name="Missingness Audit", index=False)
            # end if
        # end with

        logger.info(f"Speed Dating Excel report generated at: {path_output_excel}")
        return path_output_excel
        # end def generate_excel_report

    def _resolve_raw_dataframe(
        self,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Resolves the raw DataFrame from argument, config, or loader."""
        if df_raw is not None:
            return df_raw
        # end if
        target_path = path_raw_data or self.config.path_data_file
        return self.loader.load_data_raw(path_source=target_path)
        # end def _resolve_raw_dataframe

    def _partition_raw_dataframe(
        self,
        df_raw: pd.DataFrame
    ) -> ty.Tuple[pd.DataFrame, pd.DataFrame]:
        """Partitions raw speed dating records into Distribution X (match = 1) and Distribution Y (match = 0)."""
        if "match" in df_raw.columns:
            mask_x = (df_raw["match"] == 1)
            df_x = df_raw[mask_x].copy()
            df_y = df_raw[~mask_x].copy()
            return df_x, df_y
        # end if

        if "dec" in df_raw.columns and "dec_o" in df_raw.columns:
            mask_x = (df_raw["dec"] == 1) & (df_raw["dec_o"] == 1)
            df_x = df_raw[mask_x].copy()
            df_y = df_raw[~mask_x].copy()
            return df_x, df_y
        # end if

        half = len(df_raw) // 2
        return df_raw.iloc[:half].copy(), df_raw.iloc[half:].copy()
        # end def _partition_raw_dataframe
# end class SpeedDatingReportGenerator
