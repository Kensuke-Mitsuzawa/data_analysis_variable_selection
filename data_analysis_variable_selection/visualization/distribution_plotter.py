import os
import re
import typing as ty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.models import (
    TwoSampleDataContainer,
    VariableSelectionResult,
)
from .palette import ReportVisualPalette


class MarginalDistributionPlotter:
    """Generates comparative univariate marginal distribution plots for detected anchor variables.
    """

    def __init__(
        self,
        color_distribution_x: str = ReportVisualPalette.COLOR_DISTRIBUTION_X,
        color_distribution_y: str = ReportVisualPalette.COLOR_DISTRIBUTION_Y,
        alpha_fill: float = 0.4
    ):
        """Initializes the marginal distribution plotter.

        Args:
            color_distribution_x: Color code for distribution X (canonically Red).
            color_distribution_y: Color code for distribution Y (canonically Blue).
            alpha_fill: Opacity for density area fill.
        """
        self.color_distribution_x = color_distribution_x
        self.color_distribution_y = color_distribution_y
        self.alpha_fill = alpha_fill
        # end def __init__

    def plot_marginal_distributions(
        self,
        container: TwoSampleDataContainer,
        selection_result: VariableSelectionResult,
        directory_output: str
    ) -> ty.List[str]:
        """Plots the marginal univariate distribution of unscaled features for each selected anchor variable.

        Args:
            container: TwoSampleDataContainer containing unscaled matrices and feature names.
            selection_result: VariableSelectionResult containing selected anchor variables.
            directory_output: Output directory where image files are saved.

        Returns:
            List of generated image file paths.
        """
        os.makedirs(directory_output, exist_ok=True)
        list_generated_paths: ty.List[str] = []

        label_x = "Distribution X"
        label_y = "Distribution Y"
        if container.metadata_dataset:
            if "label_x_description" in container.metadata_dataset:
                label_x = str(container.metadata_dataset["label_x_description"])
            # end if
            if "label_y_description" in container.metadata_dataset:
                label_y = str(container.metadata_dataset["label_y_description"])
            # end if
        # end if

        name_to_index = {name: idx for idx, name in enumerate(container.name_features)}

        for idx_anchor, name_anchor in zip(
            selection_result.indices_selected,
            selection_result.names_selected
        ):
            # Resolve column index
            col_idx = idx_anchor if idx_anchor < container.sample_matrix_x.shape[1] else name_to_index.get(name_anchor)
            if col_idx is None:
                continue
            # end if

            vals_x = container.sample_matrix_x[:, col_idx]
            vals_y = container.sample_matrix_y[:, col_idx]

            vals_x = vals_x[np.isfinite(vals_x)]
            vals_y = vals_y[np.isfinite(vals_y)]

            if len(vals_x) == 0 or len(vals_y) == 0:
                continue
            # end if

            sanitized_name = re.sub(r"[^\w\-]", "_", str(name_anchor))
            filename = f"univariate_distribution_{idx_anchor}_{sanitized_name}.png"
            path_image = os.path.join(directory_output, filename)

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

            unique_vals = np.unique(np.concatenate([vals_x, vals_y]))
            is_discrete = len(unique_vals) <= 5

            if is_discrete:
                self._plot_discrete_bars(
                    ax=ax,
                    vals_x=vals_x,
                    vals_y=vals_y,
                    unique_vals=unique_vals,
                    label_x=label_x,
                    label_y=label_y
                )
            else:
                self._plot_continuous_density(
                    ax=ax,
                    vals_x=vals_x,
                    vals_y=vals_y,
                    label_x=label_x,
                    label_y=label_y
                )
            # end if

            ax.set_title(
                f"Marginal Distribution: {name_anchor}",
                fontsize=12,
                fontweight="bold",
                pad=12
            )
            ax.set_xlabel(f"{name_anchor} (Original Unscaled Value)", fontsize=10, labelpad=8)
            ax.set_ylabel("Frequency / Density", fontsize=10, labelpad=8)
            legend = ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9)
            if legend:
                for text_entry, color_entry in zip(
                    legend.get_texts(),
                    [self.color_distribution_x, self.color_distribution_y]
                ):
                    text_entry.set_color(color_entry)
                    text_entry.set_fontweight("bold")
                # end for
            # end if
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.savefig(path_image, dpi=150)
            plt.close(fig)

            list_generated_paths.append(path_image)
        # end for

        return list_generated_paths
        # end def plot_marginal_distributions

    def _plot_discrete_bars(
        self,
        ax: plt.Axes,
        vals_x: np.ndarray,
        vals_y: np.ndarray,
        unique_vals: np.ndarray,
        label_x: str,
        label_y: str
    ) -> None:
        """Plots comparative grouped bar charts for discrete/binary features."""
        x_indices = np.arange(len(unique_vals))
        width = 0.35

        prop_x = np.array([np.mean(vals_x == val) for val in unique_vals])
        prop_y = np.array([np.mean(vals_y == val) for val in unique_vals])

        ax.bar(
            x_indices - width / 2,
            prop_x,
            width=width,
            color=self.color_distribution_x,
            alpha=0.85,
            label=f"{label_x} (N={len(vals_x)})",
            edgecolor="#ffffff"
        )
        ax.bar(
            x_indices + width / 2,
            prop_y,
            width=width,
            color=self.color_distribution_y,
            alpha=0.85,
            label=f"{label_y} (N={len(vals_y)})",
            edgecolor="#ffffff"
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in unique_vals])
        # end def _plot_discrete_bars

    def _plot_continuous_density(
        self,
        ax: plt.Axes,
        vals_x: np.ndarray,
        vals_y: np.ndarray,
        label_x: str,
        label_y: str
    ) -> None:
        """Plots comparative continuous density / histograms for numeric features."""
        val_min = min(float(np.min(vals_x)), float(np.min(vals_y)))
        val_max = max(float(np.max(vals_x)), float(np.max(vals_y)))

        if val_max == val_min:
            bins = np.linspace(val_min - 1.0, val_max + 1.0, 10)
        else:
            bins = np.linspace(val_min, val_max, 25)
        # end if

        ax.hist(
            vals_x,
            bins=bins,
            density=True,
            color=self.color_distribution_x,
            alpha=self.alpha_fill,
            label=f"{label_x} (mean={np.mean(vals_x):.2f})",
            edgecolor=self.color_distribution_x,
            linewidth=1.2
        )
        ax.hist(
            vals_y,
            bins=bins,
            density=True,
            color=self.color_distribution_y,
            alpha=self.alpha_fill,
            label=f"{label_y} (mean={np.mean(vals_y):.2f})",
            edgecolor=self.color_distribution_y,
            linewidth=1.2
        )
        # end def _plot_continuous_density
# end class MarginalDistributionPlotter
