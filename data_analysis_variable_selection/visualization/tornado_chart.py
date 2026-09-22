import os
import typing as ty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.models import (
    VariableClusteringResult,
    VariableSelectionResult,
)


class TornadoChartPlotter:
    """Generates thematic horizontal 'Tornado' bar charts for clusters containing MMD anchor variables.
    """

    def __init__(self, limit_top_variables: int = 10):
        """Initializes the tornado chart plotter.

        Args:
            limit_top_variables: Maximum number of correlated variables displayed per chart.
        """
        self.limit_top_variables = limit_top_variables
        # end def __init__

    def plot_thematic_tornado_charts(
        self,
        clustering_result: VariableClusteringResult,
        selection_result: VariableSelectionResult,
        directory_output: str
    ) -> ty.List[str]:
        """Generates one tornado bar chart for each cluster containing at least one MMD anchor variable.

        Args:
            clustering_result: VariableClusteringResult holding cluster assignments and scores.
            selection_result: VariableSelectionResult holding anchor variables.
            directory_output: Directory where generated image files are written.

        Returns:
            List of generated chart file paths.
        """
        os.makedirs(directory_output, exist_ok=True)
        list_generated_paths: ty.List[str] = []

        set_anchor_indices = set(selection_result.indices_selected)
        dict_anchor_names = dict(zip(selection_result.indices_selected, selection_result.names_selected))

        # Group memberships by cluster
        dict_cluster_memberships: ty.Dict[int, ty.List] = {}
        for m in clustering_result.list_memberships:
            dict_cluster_memberships.setdefault(m.id_cluster, []).append(m)
        # end for m

        for cluster_id in sorted(dict_cluster_memberships.keys()):
            list_m = dict_cluster_memberships[cluster_id]
            anchors_in_cluster = [m for m in list_m if m.id_variable in set_anchor_indices]

            # Only plot clusters containing at least one anchor variable
            if not anchors_in_cluster:
                continue
            # end if

            # Format title
            anchor_names = [dict_anchor_names.get(m.id_variable, m.name_variable) for m in anchors_in_cluster]
            if len(anchor_names) == 1:
                title_text = f"Theme: {anchor_names[0]}"
            else:
                title_text = f"Theme: {' & '.join(anchor_names)}"
            # end if

            # Filter and sort variables by score_related descending
            valid_vars = [m for m in list_m if m.score_related is not None]
            valid_vars.sort(key=lambda m: m.score_related, reverse=True)
            top_vars = valid_vars[: self.limit_top_variables]

            if not top_vars:
                continue
            # end if

            # Plot horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")

            var_names = [m.name_variable for m in reversed(top_vars)]
            scores = [m.score_related for m in reversed(top_vars)]
            is_anchor_bar = [m.id_variable in set_anchor_indices for m in reversed(top_vars)]

            colors = ["#38bdf8" if not is_a else "#f59e0b" for is_a in is_anchor_bar]

            y_pos = np.arange(len(var_names))
            bars = ax.barh(y_pos, scores, color=colors, height=0.65, edgecolor="#0f172a")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(var_names, fontsize=10, color="#f8fafc")
            ax.tick_params(axis="x", colors="#94a3b8")
            ax.set_xlabel("Relevance Score (max |corr(v, s)| with anchor)", fontsize=11, color="#cbd5e1", labelpad=10)

            # Display values at end of bars
            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}",
                    va="center",
                    ha="left",
                    color="#f8fafc",
                    fontsize=9,
                    fontweight="bold"
                )
            # end for

            ax.set_title(title_text, fontsize=14, color="#f8fafc", pad=15, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#475569")
            ax.spines["left"].set_color("#475569")
            ax.grid(axis="x", linestyle="--", alpha=0.2, color="#94a3b8")

            max_score = max(scores) if scores else 1.0
            ax.set_xlim(0, max(1.0, max_score * 1.15))

            path_chart = os.path.join(directory_output, f"tornado_cluster_{cluster_id}.png")
            plt.tight_layout()
            plt.savefig(path_chart, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            list_generated_paths.append(os.path.abspath(path_chart))
        # end for cluster_id

        return list_generated_paths
        # end def plot_thematic_tornado_charts
# end class TornadoChartPlotter
