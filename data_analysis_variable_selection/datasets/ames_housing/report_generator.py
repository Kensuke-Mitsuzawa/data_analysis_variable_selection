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
    FeatureDerivationSummary,
)
from .config import AmesPreprocessingConfig
from .loader import AmesHousingDataLoader

logger = logging.getLogger(__name__)


class AmesHousingReportGenerator(BaseDatasetReportGenerator):
    """Generates exploratory shallow-level dataset reports tailored for the Ames Housing dataset.
    """

    def __init__(
        self,
        config: ty.Optional[AmesPreprocessingConfig] = None
    ):
        """Initializes the Ames Housing report generator.

        Args:
            config: Optional AmesPreprocessingConfig instance.
        """
        super().__init__(name_dataset="Ames Housing Dataset")
        self.config = config or AmesPreprocessingConfig()
        self.loader = AmesHousingDataLoader()
        # end def __init__

    def generate_markdown_report(
        self,
        path_output_markdown: str,
        title_report: ty.Optional[str] = None,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        **kwargs: ty.Any
    ) -> str:
        """Synthesizes Ames Housing dataset shallow statistics into a structured Markdown document.

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
        report_title = title_report or "Ames Housing Dataset Exploratory & Statistical Report"

        df_data = self._resolve_raw_dataframe(path_raw_data=path_raw_data, df_raw=df_raw)
        df_x, df_y, df_transition = self._partition_raw_dataframe(df_data)

        # 1. Compute shallow statistics
        list_financial_physical_cols = [
            "SalePrice", "GrLivArea", "LotArea", "TotalBsmtSF", "GarageArea",
            "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd", "TotRmsAbvGrd",
            "FullBath", "HalfBath", "BedroomAbvGr", "LotFrontage"
        ]
        numeric_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=list_financial_physical_cols
        )

        quality_cols = ["OverallQual", "OverallCond"]
        quality_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=quality_cols
        )

        neighborhood_comparison = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="Neighborhood",
            top_n=12
        )

        bldgtype_comparison = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="BldgType",
            top_n=6
        )

        saletype_comparison = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="SaleType",
            top_n=6
        )

        missingness_summaries = self.compute_missingness_summary(
            df_raw=df_data,
            df_x=df_x,
            df_y=df_y
        )

        # Filter domain NA feature summaries
        domain_na_cols = self.config.categorical_na_to_none_cols + self.config.continuous_na_to_zero_cols
        domain_na_summaries = [s for s in missingness_summaries if s.name_column in domain_na_cols]

        # 2. Build Markdown content
        lines: ty.List[str] = [
            f"# {report_title}",
            "",
            "> **Dataset Overview**: The Ames Housing dataset describes property sales in Ames, Iowa across 80+ nominal, ordinal, and continuous features. This report provides baseline domain exploratory data analysis, shallow statistical distributions, and empirical shifts between the **Pre-Crash market ($X$: 2006–2007)** and the **Post-Crash market ($Y$: 2009–2010)**.",
            "",
            "---",
            "",
            "## 1. Sample Partitioning & Temporal Split",
            "",
            f"- **Total Raw Observations**: {len(df_data):,} properties",
            f"- **Distribution $X$ (Pre-Crash Market, 2006–2007)**: {len(df_x):,} properties ({(len(df_x) / max(len(df_data), 1) * 100.0):.1f}%)",
            f"- **Distribution $Y$ (Post-Crash Market, 2009–2010)**: {len(df_y):,} properties ({(len(df_y) / max(len(df_data), 1) * 100.0):.1f}%)",
            f"- **Transition Epoch (2008, Excluded)**: {len(df_transition):,} properties ({(len(df_transition) / max(len(df_data), 1) * 100.0):.1f}%)",
            f"- **Raw Feature Count**: {len(df_data.columns)} columns",
            "",
            "---",
            "",
            "## 2. Key Financial & Structural Dimensions",
            "",
            "Comparative central tendencies and spreads between Pre-Crash ($X$) and Post-Crash ($Y$) property transactions:",
            "",
            "| Feature / Metric | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | % Shift | Median ($X$) | Median ($Y$) | Min / Max ($X$) | Min / Max ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for num in numeric_summaries:
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
            "## 3. Overall Quality & Condition Ratings",
            "",
            "Subjective rating scores (1–10 scale) comparing property standards:",
            "",
            "| Metric | Mean ($X$) | Mean ($Y$) | Shift ($Y - X$) | Median ($X$) | Median ($Y$) | Std ($X$) | Std ($Y$) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for q in quality_summaries:
            lines.append(
                f"| **{q.name_metric}** | {q.mean_x:.2f} | {q.mean_y:.2f} | {q.delta_mean:+.2f} | "
                f"{q.median_x:.1f} | {q.median_y:.1f} | {q.std_x:.2f} | {q.std_y:.2f} |"
            )
        # end for q

        lines.extend([
            "",
            "---",
            "",
            "## 4. Geographic & Neighborhood Prevalence Shifts",
            "",
            "Market transaction concentration across top Ames neighborhoods:",
            "",
            "| Neighborhood | Count ($X$) | Share ($X$) | Count ($Y$) | Share ($Y$) | Shift (% Points) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for neigh in neighborhood_comparison:
            lines.append(
                f"| **{neigh.name_category}** | {neigh.count_x} | {neigh.proportion_x * 100:.2f}% | "
                f"{neigh.count_y} | {neigh.proportion_y * 100:.2f}% | {neigh.delta_percentage_points:+.2f}% |"
            )
        # end for neigh

        lines.extend([
            "",
            "---",
            "",
            "## 5. Building Types & Sale Characteristics",
            "",
            "### Building Type Breakdown",
            "",
            "| Building Type | Count ($X$) | Share ($X$) | Count ($Y$) | Share ($Y$) | Shift (% Points) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for bldg in bldgtype_comparison:
            lines.append(
                f"| **{bldg.name_category}** | {bldg.count_x} | {bldg.proportion_x * 100:.2f}% | "
                f"{bldg.count_y} | {bldg.proportion_y * 100:.2f}% | {bldg.delta_percentage_points:+.2f}% |"
            )
        # end for bldg

        lines.extend([
            "",
            "### Sale Type Breakdown",
            "",
            "| Sale Type | Count ($X$) | Share ($X$) | Count ($Y$) | Share ($Y$) | Shift (% Points) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for st in saletype_comparison:
            lines.append(
                f"| **{st.name_category}** | {st.count_x} | {st.proportion_x * 100:.2f}% | "
                f"{st.count_y} | {st.proportion_y * 100:.2f}% | {st.delta_percentage_points:+.2f}% |"
            )
        # end for st

        lines.extend([
            "",
            "---",
            "",
            "## 6. Domain NAs & Amenity Absence Rates",
            "",
            "In Ames Housing, `NA` indicates the physical absence of an amenity rather than an unrecorded observation. Below is the prevalence of feature absences:",
            "",
            "| Feature | Missing / Absent Count | Missing Total (%) | Absent ($X$, %) | Absent ($Y$, %) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for dna in domain_na_summaries:
            lines.append(
                f"| **{dna.name_column}** | {dna.count_missing_total:,} | {dna.percentage_missing_total:.2f}% | "
                f"{dna.percentage_missing_x:.2f}% | {dna.percentage_missing_y:.2f}% |"
            )
        # end for dna

        # Feature Lineage Mapping
        feature_lineage_summaries = self.compute_feature_derivation_summary(df_data)
        lines.extend([
            "",
            "---",
            "",
            "## 7. Feature Derivation & Lineage Mapping",
            "",
            "Comprehensive transformation lineage mapping each preprocessed feature to its raw source columns and the engineering operation:",
            "",
            "| Feature Name | Source Column(s) | Transformation Process |",
            "| :--- | :--- | :--- |",
        ])

        for s in feature_lineage_summaries:
            src_str = f"`{s.source_columns}`"
            lines.append(f"| **{s.name_feature}** | {src_str} | {s.process_description} |")
        # end for s

        content_markdown = "\n".join(lines) + "\n"

        with open(path_output_markdown, "w", encoding="utf-8") as f_out:
            f_out.write(content_markdown)
        # end with

        logger.info(f"Ames Housing dataset report generated at: {path_output_markdown}")
        return path_output_markdown
        # end def generate_markdown_report

    def generate_excel_report(
        self,
        path_output_excel: str,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        **kwargs: ty.Any
    ) -> str:
        """Exports Ames Housing shallow-level statistics into a multi-sheet Excel workbook.

        Args:
            path_output_excel: Destination file path for .xlsx workbook.
            path_raw_data: Optional file path to raw CSV.
            df_raw: Optional pre-loaded raw DataFrame.
            **kwargs: Additional arguments.

        Returns:
            Path to generated Excel workbook.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_excel)), exist_ok=True)

        df_data = self._resolve_raw_dataframe(path_raw_data=path_raw_data, df_raw=df_raw)
        df_x, df_y, df_transition = self._partition_raw_dataframe(df_data)

        # Overview Sheet
        df_overview = pd.DataFrame([
            {"Metric": "Dataset Name", "Value": self.name_dataset},
            {"Metric": "Total Raw Observations", "Value": len(df_data)},
            {"Metric": "Pre-Crash Samples (X: 2006-2007)", "Value": len(df_x)},
            {"Metric": "Post-Crash Samples (Y: 2009-2010)", "Value": len(df_y)},
            {"Metric": "Transition Samples (2008 Excluded)", "Value": len(df_transition)},
            {"Metric": "Total Features (Raw)", "Value": len(df_data.columns)},
        ])

        # Financial & Physical Metrics Sheet
        list_financial_physical_cols = [
            "SalePrice", "GrLivArea", "LotArea", "TotalBsmtSF", "GarageArea",
            "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd", "TotRmsAbvGrd",
            "FullBath", "HalfBath", "BedroomAbvGr", "LotFrontage"
        ]
        numeric_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=list_financial_physical_cols
        )
        df_numeric = pd.DataFrame([s.model_dump() for s in numeric_summaries])

        # Quality Ratings Sheet
        quality_cols = ["OverallQual", "OverallCond"]
        quality_summaries = self.compute_numeric_metric_summary(
            df_x=df_x,
            df_y=df_y,
            list_columns=quality_cols
        )
        df_quality = pd.DataFrame([s.model_dump() for s in quality_summaries])

        # Neighborhoods Sheet
        neighborhood_summaries = self.compute_categorical_distribution_summary(
            df_x=df_x,
            df_y=df_y,
            column_name="Neighborhood",
            top_n=30
        )
        df_neighborhood = pd.DataFrame([s.model_dump() for s in neighborhood_summaries])

        # Domain NA Sheet
        missingness_summaries = self.compute_missingness_summary(
            df_raw=df_data,
            df_x=df_x,
            df_y=df_y
        )
        df_missing = pd.DataFrame([s.model_dump() for s in missingness_summaries])

        # Feature Lineage Sheet
        feature_lineage_summaries = self.compute_feature_derivation_summary(df_data)
        df_lineage = pd.DataFrame([
            {
                "Feature Name": s.name_feature,
                "Source Column(s)": ", ".join(s.source_columns),
                "Transformation Process": s.process_description,
            }
            for s in feature_lineage_summaries
        ])

        with pd.ExcelWriter(path_output_excel, engine="openpyxl") as writer:
            df_overview.to_excel(writer, sheet_name="Overview", index=False)
            if not df_numeric.empty:
                df_numeric.to_excel(writer, sheet_name="Financial & Physical Metrics", index=False)
            # end if
            if not df_quality.empty:
                df_quality.to_excel(writer, sheet_name="Quality Ratings", index=False)
            # end if
            if not df_neighborhood.empty:
                df_neighborhood.to_excel(writer, sheet_name="Neighborhood Distribution", index=False)
            # end if
            if not df_missing.empty:
                df_missing.to_excel(writer, sheet_name="Domain NAs & Missing", index=False)
            # end if
            if not df_lineage.empty:
                df_lineage.to_excel(writer, sheet_name="Feature Lineage", index=False)
            # end if
        # end with

        logger.info(f"Ames Housing Excel report generated at: {path_output_excel}")
        return path_output_excel
        # end def generate_excel_report

    def compute_feature_derivation_summary(
        self,
        df_raw: pd.DataFrame,
        list_feature_names: ty.Optional[ty.List[str]] = None,
    ) -> ty.List[FeatureDerivationSummary]:
        """Maps preprocessed features to their source raw columns and concise 4-5 word process descriptions.

        Args:
            df_raw: Raw housing dataframe.
            list_feature_names: Optional pre-computed feature names list.

        Returns:
            List of FeatureDerivationSummary instances.
        """
        if list_feature_names is None:
            from .preprocessor import AmesHousingPreprocessor
            preprocessor = AmesHousingPreprocessor(config=self.config)
            container = preprocessor.prepare_two_sample_data(df_raw=df_raw)
            feature_names = container.name_features
        else:
            feature_names = list_feature_names
        # end if

        raw_columns = list(df_raw.columns)
        nominal_columns = sorted(
            [
                col for col in raw_columns
                if col not in self.config.ordinal_mapping_dicts
                and col not in self.config.columns_to_drop
                and df_raw[col].dtype == "object"
            ],
            key=len,
            reverse=True
        )

        list_summaries: ty.List[FeatureDerivationSummary] = []
        for feat in feature_names:
            if feat == "LotFrontage":
                src = ["LotFrontage", "Neighborhood"]
                desc = "Neighborhood median stratified imputation"
            elif feat in self.config.ordinal_mapping_dicts:
                src = [feat]
                desc = "Monotonic integer ordinal rating mapping"
            elif feat in self.config.continuous_na_to_zero_cols:
                src = [feat]
                desc = "Zero-filled physical absence continuous value"
            elif feat in raw_columns:
                src = [feat]
                desc = "Direct numeric feature pass-through"
            else:
                matched_col = None
                for nom_col in nominal_columns:
                    if feat.startswith(nom_col + "_"):
                        matched_col = nom_col
                        break
                    # end if
                # end for
                if matched_col is None:
                    for raw_col in sorted(raw_columns, key=len, reverse=True):
                        if feat.startswith(raw_col + "_"):
                            matched_col = raw_col
                            break
                        # end if
                    # end for
                # end if

                src = [matched_col] if matched_col else [feat]
                desc = "One-hot binary categorical indicator"
            # end if

            list_summaries.append(
                FeatureDerivationSummary(
                    name_feature=feat,
                    source_columns=src,
                    process_description=desc
                )
            )
        # end for feat

        return list_summaries
        # end def compute_feature_derivation_summary

    def generate_dataset_report(
        self,
        directory_output: str,
        path_raw_data: ty.Optional[str] = None,
        df_raw: ty.Optional[pd.DataFrame] = None,
        title_report: ty.Optional[str] = None,
        export_excel: bool = True
    ) -> DatasetReportArtifacts:
        """Coordinates the end-to-end generation of Ames Housing dataset reports.

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
    ) -> ty.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partitions the raw dataset into Distribution X, Distribution Y, and transition samples."""
        if "YrSold" not in df_raw.columns:
            half = len(df_raw) // 2
            return df_raw.iloc[:half], df_raw.iloc[half:], pd.DataFrame()
        # end if

        mask_x = df_raw["YrSold"].isin(self.config.years_pre_crash)
        mask_y = df_raw["YrSold"].isin(self.config.years_post_crash)
        mask_transition = ~mask_x & ~mask_y

        df_x = df_raw[mask_x].copy()
        df_y = df_raw[mask_y].copy()
        df_transition = df_raw[mask_transition].copy()

        return df_x, df_y, df_transition
        # end def _partition_raw_dataframe
# end class AmesHousingReportGenerator
