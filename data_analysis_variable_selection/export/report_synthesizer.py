import json
import logging
import os
import typing as ty
import pandas as pd

from ..database.manager import DuckDBStorageManager

logger = logging.getLogger(__name__)


class ReportSynthesizer:
    """Synthesizes comprehensive analytical reports into multi-sheet Excel workbooks and Markdown documents.
    """

    def generate_excel_workbook(
        self,
        db_manager: DuckDBStorageManager,
        path_output_excel: str
    ) -> str:
        """Exports analytical warehouse tables into a multi-sheet Excel workbook.

        Args:
            db_manager: Connected DuckDBStorageManager.
            path_output_excel: Filepath for the generated .xlsx workbook.

        Returns:
            Path to created Excel file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_excel)), exist_ok=True)

        df_sel = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_selection ORDER BY weight DESC")
        df_clust = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_clustering ORDER BY id_cluster, score_related DESC")
        df_corr = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_correlation ORDER BY correlation_score DESC")
        df_proto = db_manager.fetch_records_sql("SELECT * FROM analysis_representative_samples ORDER BY type_subspace, distance_score ASC")

        with pd.ExcelWriter(path_output_excel, engine="openpyxl") as writer:
            df_sel.to_excel(writer, sheet_name="Anchor Variables", index=False)
            df_clust.to_excel(writer, sheet_name="Cluster Themes", index=False)
            df_corr.to_excel(writer, sheet_name="Pairwise Correlations", index=False)
            df_proto.to_excel(writer, sheet_name="Representative Exemplars", index=False)
        # end with

        logger.info(f"Excel workbook generated at: {path_output_excel}")
        return path_output_excel
        # end def generate_excel_workbook

    def generate_markdown_report(
        self,
        db_manager: DuckDBStorageManager,
        path_output_markdown: str,
        title_report: str = "Two-Sample Variable Selection Analysis Report",
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None,
    ) -> str:
        """Synthesizes a structured Markdown executive report with tables and embedded visualizations.

        Args:
            db_manager: Connected DuckDBStorageManager.
            path_output_markdown: Destination filepath for report.md.
            title_report: Display title for the report.
            dict_artifacts: Optional mapping of visual artifact paths.

        Returns:
            Path to generated Markdown report.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_markdown)), exist_ok=True)

        df_sel = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_selection ORDER BY weight DESC")
        df_clust = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_clustering ORDER BY id_cluster, score_related DESC")
        df_proto = db_manager.fetch_records_sql("SELECT * FROM analysis_representative_samples ORDER BY type_subspace, distance_score ASC")

        lines = [
            f"# {title_report}",
            "",
            "> **Executive Summary**: This report summarizes the statistical discrepancy drivers discovered between sample distribution $X$ and distribution $Y$. Anchor variables ($\\hat{S}$) isolate the intrinsic coordinates of change, while clustering ($S_\\text{tilde}$) reveals broader thematic patterns.",
            "",
            "---",
            "",
            "## 1. Discovered Anchor Variables ($\\hat{S}$)",
            "",
            "Anchor variables represent the core intrinsic dimensions exhibiting maximum discrepancy between distributions:",
            "",
            "| Variable ID | Variable Name | Discrepancy Weight |",
            "| :--- | :--- | :--- |",
        ]

        for _, row in df_sel.iterrows():
            lines.append(f"| {int(row['id_variable'])} | **{row['name_variable']}** | {float(row['weight']):.4f} |")
        # end for

        lines.extend([
            "",
            "---",
            "",
            "## 2. Cluster Themes & Augmented Feature Sets ($S_\\text{tilde}$)",
            "",
            "Features correlated with anchor variables are grouped into thematic clusters to provide business/domain interpretability:",
            "",
            "| Cluster ID | Feature Name | Relatedness Score |",
            "| :--- | :--- | :--- |",
        ])

        for _, row in df_clust.head(25).iterrows():
            score_str = f"{float(row['score_related']):.4f}" if pd.notna(row['score_related']) else "N/A"
            lines.append(
                f"| Cluster {int(row['id_cluster'])} | **{row['name_variable']}** | {score_str} |"
            )
        # end for

        if len(df_clust) > 25:
            lines.append(f"| ... | *({len(df_clust) - 25} more records in Excel report)* | ... |")
        # end if

        lines.extend([
            "",
            "---",
            "",
            "## 3. Representative Prototype Exemplars",
            "",
            "Prototypical samples representing the central density of each distribution in the discrepancy subspace:",
            "",
            "| Sample ID | True Label | Prototype Role | Subspace | Discrepancy / Distance Score |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for _, row in df_proto.head(10).iterrows():
            lines.append(
                f"| #{int(row['id_sample'])} | {row['label_class']} | {row['is_prototype_for']} | {row['type_subspace']} | {float(row['distance_score']):.4f} |"
            )
        # end for

        if dict_artifacts:
            lines.extend([
                "",
                "---",
                "",
                "## 4. Visualizations Gallery",
                "",
            ])
            if "constellation_network" in dict_artifacts:
                rel_path = os.path.basename(dict_artifacts["constellation_network"])
                lines.extend([
                    "### Constellation Network Graph",
                    f"![Constellation Network]({rel_path})",
                    "",
                ])
            # end if

            if "tornado_charts" in dict_artifacts:
                lines.append("### Thematic Cluster Tornado Charts")
                for path_tornado in dict_artifacts["tornado_charts"]:
                    rel_path = os.path.basename(path_tornado)
                    lines.append(f"![Tornado Chart]({rel_path})")
                    lines.append("")
                # end for
            # end if

            if "persona_radar_charts" in dict_artifacts:
                lines.append("### Persona Comparison Radar Charts")
                for path_radar in dict_artifacts["persona_radar_charts"]:
                    rel_path = os.path.basename(path_radar)
                    lines.append(f"![Radar Chart]({rel_path})")
                    lines.append("")
                # end for
            # end if
        # end if

        content_md = "\n".join(lines) + "\n"

        with open(path_output_markdown, "w", encoding="utf-8") as f_out:
            f_out.write(content_md)
        # end with

        logger.info(f"Markdown report generated at: {path_output_markdown}")
        return path_output_markdown
        # end def generate_markdown_report
# end class ReportSynthesizer
