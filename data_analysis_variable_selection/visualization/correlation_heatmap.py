import os
import typing as ty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.models import (
    CorrelationResult,
    VariableClusteringResult,
    VariableSelectionResult,
)


class CorrelationHeatmapPlotter:
    """Generates heatmaps of the variable correlation/precision matrix with Red-to-Blue colormaps.
    """

    def __init__(
        self,
        colormap_name: str = "coolwarm",
        max_variables_display: int = 35
    ):
        """Initializes the correlation heatmap plotter.

        Args:
            colormap_name: Matplotlib colormap where Red is higher and Blue is lower.
            max_variables_display: Maximum number of variables displayed to preserve legibility.
        """
        self.colormap_name = colormap_name
        self.max_variables_display = max_variables_display
        # end def __init__

    def plot_correlation_heatmap(
        self,
        correlation_result: CorrelationResult,
        directory_output: str,
        clustering_result: ty.Optional[VariableClusteringResult] = None,
        selection_result: ty.Optional[VariableSelectionResult] = None,
        path_output_image: ty.Optional[str] = None
    ) -> str:
        """Plots the variable relationship heatmap and saves it to disk.

        Args:
            correlation_result: CorrelationResult containing the correlation/precision matrix.
            directory_output: Target folder for saving the image.
            clustering_result: Optional VariableClusteringResult with augmented variable set S_tilde.
            selection_result: Optional VariableSelectionResult with anchor variables hat_S.
            path_output_image: Optional custom output file path.

        Returns:
            Path to the saved correlation heatmap PNG.
        """
        os.makedirs(directory_output, exist_ok=True)
        path_image = path_output_image or os.path.join(directory_output, "correlation_heatmap.png")

        matrix_corr = np.array(correlation_result.matrix_correlation, dtype=np.float64)
        names_all = list(correlation_result.names_variables)
        total_features = len(names_all)

        # 1. Determine which variables to include
        selected_indices = self._select_display_indices(
            total_features=total_features,
            matrix_corr=matrix_corr,
            clustering_result=clustering_result,
            selection_result=selection_result
        )

        sub_matrix = matrix_corr[np.ix_(selected_indices, selected_indices)]
        sub_names = [names_all[i] for i in selected_indices]

        # 2. Render Heatmap
        n_vars = len(sub_names)
        fig_size = max(8.0, min(16.0, n_vars * 0.45 + 3.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)

        # Determine scale: if precision or absolute correlation
        val_min = float(np.nanmin(sub_matrix))
        val_max = float(np.nanmax(sub_matrix))
        if val_min >= 0.0:
            vmin_plot = 0.0
            vmax_plot = max(1.0, val_max)
        else:
            limit = max(abs(val_min), abs(val_max))
            vmin_plot = -limit
            vmax_plot = limit
        # end if

        im = ax.imshow(
            sub_matrix,
            cmap=self.colormap_name,
            vmin=vmin_plot,
            vmax=vmax_plot,
            interpolation="nearest"
        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation / Relationship Magnitude", fontsize=10, labelpad=10)

        # Axis ticks and labels
        ax.set_xticks(np.arange(n_vars))
        ax.set_yticks(np.arange(n_vars))
        ax.set_xticklabels(sub_names, rotation=90, ha="right", fontsize=8)
        ax.set_yticklabels(sub_names, fontsize=8)

        # Highlight anchor variables in bold if known
        if selection_result:
            set_anchors = set(selection_result.names_selected)
            for idx, name in enumerate(sub_names):
                if name in set_anchors:
                    ax.get_xticklabels()[idx].set_fontweight("bold")
                    ax.get_xticklabels()[idx].set_color("#b2182b")
                    ax.get_yticklabels()[idx].set_fontweight("bold")
                    ax.get_yticklabels()[idx].set_color("#b2182b")
                # end if
            # end for
        # end if

        subtitle = f"({n_vars} Key Variables: Anchors & Augmented Thematic Neighborhoods)" if n_vars < total_features else f"({n_vars} Variables)"
        ax.set_title(
            f"Variable Relationship Matrix Heatmap\n{subtitle}",
            fontsize=12,
            fontweight="bold",
            pad=15
        )

        plt.tight_layout()
        plt.savefig(path_image, dpi=150)
        plt.close(fig)

        return path_image
        # end def plot_correlation_heatmap

    def _select_display_indices(
        self,
        total_features: int,
        matrix_corr: np.ndarray,
        clustering_result: ty.Optional[VariableClusteringResult],
        selection_result: ty.Optional[VariableSelectionResult]
    ) -> ty.List[int]:
        """Selects the most informative subset of variable indices to keep the heatmap readable."""
        if total_features <= self.max_variables_display:
            return list(range(total_features))
        # end if

        chosen: ty.List[int] = []
        if selection_result and selection_result.indices_selected:
            chosen.extend(selection_result.indices_selected)
        # end if

        if clustering_result and clustering_result.indices_augmented_s_tilde:
            for idx in clustering_result.indices_augmented_s_tilde:
                if idx not in chosen and idx < total_features:
                    chosen.append(idx)
                # end if
            # end for
        # end if

        if len(chosen) > self.max_variables_display:
            # Prioritize anchors, then highest average correlation with anchors
            anchor_indices = [idx for idx in (selection_result.indices_selected if selection_result else []) if idx < total_features]
            if anchor_indices:
                other_indices = [i for i in chosen if i not in anchor_indices]
                other_indices.sort(
                    key=lambda idx: float(np.nanmax(np.abs(matrix_corr[idx, anchor_indices]))),
                    reverse=True
                )
                slots = self.max_variables_display - len(anchor_indices)
                chosen = anchor_indices + other_indices[:max(0, slots)]
            else:
                chosen = chosen[:self.max_variables_display]
            # end if
        elif len(chosen) < 10:
            # Fallback to top-K most connected variables
            avg_conn = np.nanmean(np.abs(matrix_corr), axis=1)
            sorted_by_conn = list(np.argsort(-avg_conn))
            for idx in sorted_by_conn:
                if idx not in chosen:
                    chosen.append(int(idx))
                # end if
                if len(chosen) >= self.max_variables_display:
                    break
                # end if
            # end for
        # end if

        chosen.sort()
        return chosen
        # end def _select_display_indices
# end class CorrelationHeatmapPlotter
