import os
import typing as ty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.models import (
    PrototypeSampleResult,
    VariableClusteringResult,
    VariableSelectionResult,
)


class PersonaRadarChartPlotter:
    """Renders Persona Radar (Spider) charts comparing Top-1 prototypical samples across distributions.
    """

    def __init__(self, limit_axes_max: int = 8):
        """Initializes the radar chart plotter.

        Args:
            limit_axes_max: Hard limit on the number of features displayed per chart (strictly <= 8).
        """
        assert limit_axes_max <= 8, "Radar charts are cognitively capped at a maximum of 8 axes."
        self.limit_axes_max = limit_axes_max
        # end def __init__

    def plot_persona_radar_charts(
        self,
        prototype_result: PrototypeSampleResult,
        clustering_result: VariableClusteringResult,
        selection_result: VariableSelectionResult,
        directory_output: str
    ) -> ty.List[str]:
        """Generates one radar chart per anchor cluster comparing Top-1 prototype of X vs Top-1 prototype of Y.

        Args:
            prototype_result: PrototypeSampleResult containing exemplar records.
            clustering_result: VariableClusteringResult containing cluster assignments.
            selection_result: VariableSelectionResult containing anchor variables.
            directory_output: Directory where generated image files are written.

        Returns:
            List of generated radar chart image file paths.
        """
        os.makedirs(directory_output, exist_ok=True)
        list_generated_paths: ty.List[str] = []

        # Find Top-1 prototype for X and Top-1 for Y
        prototypes_x = [p for p in prototype_result.list_prototype_records if p.label_class == "X"]
        prototypes_y = [p for p in prototype_result.list_prototype_records if p.label_class == "Y"]

        if not prototypes_x or not prototypes_y:
            return []
        # end if

        prototype_top_x = prototypes_x[0]
        prototype_top_y = prototypes_y[0]

        set_anchor_indices = set(selection_result.indices_selected)

        # Group memberships by cluster
        dict_cluster_memberships: ty.Dict[int, ty.List] = {}
        for m in clustering_result.list_memberships:
            dict_cluster_memberships.setdefault(m.id_cluster, []).append(m)
        # end for m

        for cluster_id, list_m in dict_cluster_memberships.items():
            anchors_in_cluster = [m for m in list_m if m.id_variable in set_anchor_indices]

            if not anchors_in_cluster:
                continue
            # end if

            # Rule 2: Anchors first, then highest score_related up to limit_axes_max
            selected_vars = list(anchors_in_cluster)
            non_anchors = [m for m in list_m if m.id_variable not in set_anchor_indices and m.score_related is not None]
            non_anchors.sort(key=lambda m: m.score_related, reverse=True)

            slots_remaining = self.limit_axes_max - len(selected_vars)
            if slots_remaining > 0:
                selected_vars.extend(non_anchors[:slots_remaining])
            else:
                selected_vars = selected_vars[: self.limit_axes_max]
            # end if

            # We need at least 3 features to make a polygon radar chart
            if len(selected_vars) < 3:
                continue
            # end if

            feature_names = [m.name_variable for m in selected_vars]

            # Extract raw values for top X and top Y prototypes
            vals_x = [prototype_top_x.dict_feature_values.get(f_name, 0.0) for f_name in feature_names]
            vals_y = [prototype_top_y.dict_feature_values.get(f_name, 0.0) for f_name in feature_names]

            # Rule 3: Normalize to [0, 1] scale across the two samples
            vals_x_norm = []
            vals_y_norm = []
            for vx, vy in zip(vals_x, vals_y):
                min_v = min(vx, vy)
                max_v = max(vx, vy)
                range_v = max_v - min_v
                if range_v < 1e-8:
                    vals_x_norm.append(0.5)
                    vals_y_norm.append(0.5)
                else:
                    vals_x_norm.append((vx - min_v) / range_v)
                    vals_y_norm.append((vy - min_v) / range_v)
                # end if
            # end for

            # Setup radar angles
            num_vars = len(feature_names)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            # Close polygon
            angles += angles[:1]
            vals_x_norm += vals_x_norm[:1]
            vals_y_norm += vals_y_norm[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")

            # Draw axes and labels
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_names, fontsize=9, color="#f8fafc")

            ax.set_rlabel_position(0)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#94a3b8", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.grid(color="#475569", linestyle="--", alpha=0.5)

            # Plot distribution X
            ax.plot(angles, vals_x_norm, color="#38bdf8", linewidth=2.5, label="Prototype X (Pre-Crash / Matched)")
            ax.fill(angles, vals_x_norm, color="#38bdf8", alpha=0.25)

            # Plot distribution Y
            ax.plot(angles, vals_y_norm, color="#f43f5e", linewidth=2.5, label="Prototype Y (Post-Crash / Unmatched)")
            ax.fill(angles, vals_y_norm, color="#f43f5e", alpha=0.25)

            ax.set_title(
                f"Persona Comparison: Cluster {cluster_id} Prototype Contrast",
                fontsize=13,
                color="#f8fafc",
                pad=22,
                fontweight="bold"
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

            path_chart = os.path.join(directory_output, f"persona_radar_cluster_{cluster_id}.png")
            plt.tight_layout()
            plt.savefig(path_chart, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            list_generated_paths.append(os.path.abspath(path_chart))
        # end for cluster_id

        return list_generated_paths
        # end def plot_persona_radar_charts
# end class PersonaRadarChartPlotter
