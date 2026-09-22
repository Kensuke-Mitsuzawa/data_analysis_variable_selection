from datetime import datetime, timezone
import json
import logging
import os
import re
import subprocess
import typing as ty
import jinja2
import pandas as pd

from ..database.manager import DuckDBStorageManager

logger = logging.getLogger(__name__)

DEFAULT_BASE_REPORT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "template_report",
    "base_report.md"
)


class ReportSynthesizer:
    """Synthesizes comprehensive analytical reports into multi-sheet Excel workbooks and Markdown documents via templates.
    """

    def __init__(
        self,
        path_template_markdown: ty.Optional[str] = None
    ):
        """Initializes the report synthesizer.

        Args:
            path_template_markdown: Optional path to template Markdown file with placeholders.
        """
        self.path_template_markdown = path_template_markdown or DEFAULT_BASE_REPORT_TEMPLATE_PATH
        # end def __init__

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
        path_dataset_report: ty.Optional[str] = None,
        path_template_markdown: ty.Optional[str] = None,
    ) -> str:
        """Synthesizes a structured Markdown executive report by substituting placeholders into the template.

        Args:
            db_manager: Connected DuckDBStorageManager.
            path_output_markdown: Destination filepath for report.md.
            title_report: Display title for the report.
            dict_artifacts: Optional mapping of visual artifact paths.
            path_dataset_report: Optional file path or relative link to dataset-specific report.
            path_template_markdown: Optional custom template Markdown file path.

        Returns:
            Path to generated Markdown report.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_markdown)), exist_ok=True)

        df_sel = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_selection ORDER BY weight DESC")
        df_clust = db_manager.fetch_records_sql("SELECT * FROM analysis_variable_clustering ORDER BY id_cluster, score_related DESC")
        df_proto = db_manager.fetch_records_sql("SELECT * FROM analysis_representative_samples ORDER BY type_subspace, distance_score ASC")

        # Extract git commit and timestamp
        report_generation_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        git_commit_id = "Unknown"
        try:
            git_commit_id = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            pass
        # end try

        # Extract dataset-specific distribution label descriptions
        label_x_description = "Distribution X"
        label_y_description = "Distribution Y"
        try:
            df_meta = db_manager.fetch_records_sql("SELECT * FROM dataset_metadata")
            if not df_meta.empty:
                row_json = df_meta[df_meta["key"] == "metadata_json"]
                if not row_json.empty:
                    meta_dict = json.loads(row_json["value"].values[0])
                    label_x_description = meta_dict.get("label_x_description", label_x_description)
                    label_y_description = meta_dict.get("label_y_description", label_y_description)
                # end if
            # end if
        except Exception as e:
            logger.warning(f"Could not load label descriptions from dataset_metadata: {e}")
        # end try

        # 1. Prepare component content strings
        summary_note = self.render_summary_note(path_dataset_report=path_dataset_report)
        table_anchors = self.render_table_anchor_variables(df_sel=df_sel)
        table_clust = self.render_table_cluster_themes(df_clust=df_clust, df_sel=df_sel)
        table_proto = self.render_table_representative_prototypes(df_proto=df_proto)
        section_marginal = self.render_section_marginal_univariate_distributions(dict_artifacts=dict_artifacts)
        section_corr = self.render_section_variable_correlation(dict_artifacts=dict_artifacts)
        section_constellation = self.render_section_constellation_network(dict_artifacts=dict_artifacts)
        section_tornado = self.render_section_tornado_charts(dict_artifacts=dict_artifacts)
        section_radar = self.render_section_persona_radar_charts(dict_artifacts=dict_artifacts)
        section_vis = self.render_section_visualizations(dict_artifacts=dict_artifacts)

        dict_context = {
            "title_report": title_report,
            "summary_note": summary_note,
            "report_generation_timestamp": report_generation_timestamp,
            "git_commit_id": git_commit_id,
            "label_x_description": label_x_description,
            "label_y_description": label_y_description,
            "table_anchor_variables": table_anchors,
            "section_marginal_univariate_distributions": section_marginal,
            "section_variable_correlation": section_corr,
            "section_constellation_network": section_constellation,
            "table_cluster_themes": table_clust,
            "section_tornado_charts": section_tornado,
            "table_representative_prototypes": table_proto,
            "section_persona_radar_charts": section_radar,
            "section_visualizations": section_vis,
        }

        # 2. Load template
        target_template_path = path_template_markdown or self.path_template_markdown
        content_template = self._load_template_content(path_template=target_template_path)

        # 3. Render template with Jinja2 (fallback to replace)
        content_rendered = self._render_template_with_context(
            content_template=content_template,
            dict_context=dict_context
        )

        with open(path_output_markdown, "w", encoding="utf-8") as f_out:
            f_out.write(content_rendered)
        # end with

        logger.info(f"Markdown report generated from template at: {path_output_markdown}")
        return path_output_markdown
        # end def generate_markdown_report

    def render_summary_note(
        self,
        path_dataset_report: ty.Optional[str] = None
    ) -> str:
        """Renders the executive summary block with optional companion dataset link.

        Args:
            path_dataset_report: Optional file path or link to dataset report.

        Returns:
            Markdown quote block string.
        """
        summary_note = (
            "> **Executive Summary**: This report summarizes the statistical discrepancy drivers discovered "
            "between sample distribution $X$ and distribution $Y$. Anchor variables ($\\hat{S}$) isolate the "
            "intrinsic coordinates of change, while clustering ($S_\\text{tilde}$) reveals broader thematic patterns."
        )
        if path_dataset_report:
            ref_link = os.path.basename(path_dataset_report)
            summary_note += (
                f"\n>\n> *Note: For shallow exploratory statistics, domain-specific metrics, and baseline distributions, "
                f"refer to the companion [{ref_link}]({ref_link}).*"
            )
        # end if
        return summary_note
        # end def render_summary_note

    def render_table_anchor_variables(
        self,
        df_sel: pd.DataFrame
    ) -> str:
        """Formats discovered anchor variables into a Markdown table.

        Args:
            df_sel: DataFrame containing selection records.

        Returns:
            Markdown table string.
        """
        if df_sel.empty:
            return "*No anchor variables discovered.*"
        # end if

        lines = [
            "| Variable ID | Variable Name | Discrepancy Weight |",
            "| :--- | :--- | :--- |",
        ]
        for _, row in df_sel.iterrows():
            lines.append(f"| {int(row['id_variable'])} | **{row['name_variable']}** | {float(row['weight']):.4f} |")
        # end for
        return "\n".join(lines)
        # end def render_table_anchor_variables

    def render_table_cluster_themes(
        self,
        df_clust: pd.DataFrame,
        df_sel: ty.Optional[pd.DataFrame] = None,
        top_k_per_cluster: int = 5
    ) -> str:
        """Formats clustered features into Markdown tables separated per cluster ID.

        Filters out records where the Relatedness Score is NA / None.
        Identifies whether each feature was discovered by Variable Selection (anchor)
        or Variable Augmentation.

        Args:
            df_clust: DataFrame containing clustering records.
            df_sel: Optional DataFrame containing variable selection records.
            top_k_per_cluster: Maximum top augmented records shown per cluster.

        Returns:
            Markdown formatted tables separated by cluster.
        """
        if df_clust.empty:
            return "*No cluster records available.*"
        # end if

        # Filter out rows where Relatedness Score is NA
        df_valid = df_clust[df_clust["score_related"].notna()].copy()
        if df_valid.empty:
            return "*No augmented features with valid relatedness scores available.*"
        # end if

        set_anchor_names = set(df_sel["name_variable"].tolist()) if df_sel is not None and not df_sel.empty else set()

        cluster_blocks: ty.List[str] = []
        clusters = sorted(df_valid["id_cluster"].unique())

        for cid in clusters:
            df_c = df_valid[df_valid["id_cluster"] == cid]

            # Separate anchors and augmented features within this cluster
            df_anchors = df_c[df_c["name_variable"].isin(set_anchor_names)].sort_values("score_related", ascending=False)
            df_augmented = df_c[~df_c["name_variable"].isin(set_anchor_names)].sort_values("score_related", ascending=False)

            lines = [
                f"### Cluster {int(cid)}",
                "",
                "| Feature Name | Detection Source | Relatedness Score |",
                "| :--- | :--- | :--- |",
            ]

            # Render anchor features
            for _, row in df_anchors.iterrows():
                lines.append(
                    f"| **{row['name_variable']}** | Variable Selection | {float(row['score_related']):.4f} |"
                )
            # end for

            # Render top-k augmented features
            df_aug_top = df_augmented.head(top_k_per_cluster)
            for _, row in df_aug_top.iterrows():
                lines.append(
                    f"| {row['name_variable']} | Variable Augmentation | {float(row['score_related']):.4f} |"
                )
            # end for

            cluster_blocks.append("\n".join(lines))
        # end for cid

        return "\n\n".join(cluster_blocks)
        # end def render_table_cluster_themes

    def render_section_marginal_univariate_distributions(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Formats the marginal univariate distributions section for detected anchor variables.

        Args:
            dict_artifacts: Mapping of artifact paths.

        Returns:
            Markdown snippet string with embedded univariate distribution plots.
        """
        if not dict_artifacts or "marginal_distribution_plots" not in dict_artifacts:
            return "*Marginal univariate distribution plots not available.*"
        # end if

        list_plots = dict_artifacts["marginal_distribution_plots"]
        if not list_plots:
            return "*No marginal distribution plots generated.*"
        # end if

        lines = [
            "Comparative univariate distributions of unscaled feature values between Distribution $X$ and Distribution $Y$ for all detected anchor variables ($\\hat{S}$):",
            "",
        ]

        for path_img in list_plots:
            rel_path = os.path.basename(path_img)
            name_part = os.path.splitext(rel_path)[0].replace("univariate_distribution_", "")
            lines.append(f"### Marginal Distribution: `{name_part}`")
            lines.append(f"![Marginal Distribution - {name_part}]({rel_path})")
            lines.append("")
        # end for

        return "\n".join(lines).strip()
        # end def render_section_marginal_univariate_distributions

    def render_section_variable_correlation(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Formats the variable correlation section with embedded heatmap.

        Args:
            dict_artifacts: Mapping of artifact paths.

        Returns:
            Markdown snippet string with embedded heatmap.
        """
        if not dict_artifacts or "correlation_heatmap" not in dict_artifacts:
            return "*Variable correlation heatmap not available.*"
        # end if

        path_heatmap = dict_artifacts["correlation_heatmap"]
        rel_path = os.path.basename(path_heatmap)

        lines = [
            "Pairwise relationship matrix derived from Graphical Lasso precision analysis across key anchor variables and their augmented thematic clusters ($S_\\text{tilde}$). Red indicates positive correlation / strong connection, while Blue indicates negative or lower association:",
            "",
            f"![Variable Correlation Matrix Heatmap]({rel_path})",
        ]
        return "\n".join(lines)
        # end def render_section_variable_correlation

    def render_table_representative_prototypes(
        self,
        df_proto: pd.DataFrame,
        max_rows_per_label: int = 5
    ) -> str:
        """Formats representative prototype samples into Markdown tables separated by True Label (X and Y).

        Args:
            df_proto: DataFrame containing exemplar records.
            max_rows_per_label: Maximum rows to display per distribution label.

        Returns:
            Markdown tables separated by True Label.
        """
        if df_proto.empty:
            return "*No prototype exemplars available.*"
        # end if

        label_blocks: ty.List[str] = []
        labels = sorted(df_proto["label_class"].unique())

        for label in labels:
            df_label = df_proto[df_proto["label_class"] == label].head(max_rows_per_label)
            lines = [
                f"### Distribution ${label}$ Prototypes",
                "",
                "| Sample ID | True Label | Prototype Role | Subspace | Discrepancy / Distance Score |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for _, row in df_label.iterrows():
                lines.append(
                    f"| #{int(row['id_sample'])} | {row['label_class']} | {row['is_prototype_for']} | {row['type_subspace']} | {float(row['distance_score']):.4f} |"
                )
            # end for
            label_blocks.append("\n".join(lines))
        # end for label

        return "\n\n".join(label_blocks)
        # end def render_table_representative_prototypes

    def render_section_constellation_network(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Formats the constellation network graph markdown snippet.

        Args:
            dict_artifacts: Mapping of artifact paths.

        Returns:
            Markdown image snippet or notice.
        """
        if not dict_artifacts or "constellation_network" not in dict_artifacts:
            return "*Constellation network graph not available.*"
        # end if
        rel_path = os.path.basename(dict_artifacts["constellation_network"])
        return f"![Constellation Network]({rel_path})"
        # end def render_section_constellation_network

    def render_section_tornado_charts(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Formats the thematic cluster tornado charts markdown snippet.

        Args:
            dict_artifacts: Mapping of artifact paths.

        Returns:
            Markdown images snippet or notice.
        """
        if not dict_artifacts or "tornado_charts" not in dict_artifacts:
            return "*Thematic cluster tornado charts not available.*"
        # end if

        def _get_cluster_num(path_str: str) -> int:
            match = re.search(r"cluster_(\d+)", path_str)
            return int(match.group(1)) if match else 999999
        # end def

        sorted_paths = sorted(dict_artifacts["tornado_charts"], key=_get_cluster_num)
        lines = []
        for path_tornado in sorted_paths:
            rel_path = os.path.basename(path_tornado)
            lines.append(f"![Tornado Chart]({rel_path})\n")
        # end for
        return "\n".join(lines).strip()
        # end def render_section_tornado_charts

    def render_section_persona_radar_charts(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Formats the persona comparison radar charts markdown snippet.

        Args:
            dict_artifacts: Mapping of artifact paths.

        Returns:
            Markdown images snippet or notice.
        """
        if not dict_artifacts or "persona_radar_charts" not in dict_artifacts:
            return "*Persona comparison radar charts not available.*"
        # end if

        def _get_cluster_num(path_str: str) -> int:
            match = re.search(r"cluster_(\d+)", path_str)
            return int(match.group(1)) if match else 999999
        # end def

        sorted_paths = sorted(dict_artifacts["persona_radar_charts"], key=_get_cluster_num)
        lines = []
        for path_radar in sorted_paths:
            rel_path = os.path.basename(path_radar)
            lines.append(f"![Radar Chart]({rel_path})\n")
        # end for
        return "\n".join(lines).strip()
        # end def render_section_persona_radar_charts

    def render_section_visualizations(
        self,
        dict_artifacts: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> str:
        """Legacy compatibility method for visualization gallery.

        Returns empty string as visualizations are now integrated into contextual template sections.
        """
        return ""
        # end def render_section_visualizations

    def _load_template_content(
        self,
        path_template: str
    ) -> str:
        """Loads template text from disk or falls back to built-in default if missing."""
        if os.path.exists(path_template):
            with open(path_template, "r", encoding="utf-8") as f_tpl:
                content = f_tpl.read()
                if content.strip():
                    return content
                # end if
            # end with
        # end if

        # Fallback default template
        logger.warning(f"Template not found or empty at: {path_template}. Using built-in default.")
        return (
            "# {{ title_report }}\n\n"
            "{{ summary_note }}\n\n"
            "---\n\n"
            "## 1. Discovered Anchor Variables ($\\hat{S}$)\n\n"
            "Anchor variables represent the core intrinsic dimensions exhibiting maximum discrepancy between distributions:\n\n"
            "{{ table_anchor_variables }}\n\n"
            "---\n\n"
            "## 2. Cluster Themes & Augmented Feature Sets ($S_\\text{tilde}$)\n\n"
            "Features correlated with anchor variables are grouped into thematic clusters to provide business/domain interpretability:\n\n"
            "{{ table_cluster_themes }}\n\n"
            "---\n\n"
            "## 3. Representative Prototype Exemplars\n\n"
            "Prototypical samples representing the central density of each distribution in the discrepancy subspace:\n\n"
            "{{ table_representative_prototypes }}\n\n"
            "{{ section_visualizations }}\n"
        )
        # end def _load_template_content

    def _render_template_with_context(
        self,
        content_template: str,
        dict_context: ty.Dict[str, ty.Any]
    ) -> str:
        """Renders the template with the provided context variables using Jinja2."""
        try:
            template = jinja2.Template(content_template)
            return template.render(**dict_context)
        except Exception as err:
            logger.warning(f"Jinja2 rendering failed ({err}); falling back to standard string substitution.")
            res = content_template
            for key, val in dict_context.items():
                res = res.replace(f"{{{{ {key} }}}}", str(val)).replace(f"{{{{{key}}}}}", str(val))
            # end for
            return res
        # end try
        # end def _render_template_with_context
# end class ReportSynthesizer
