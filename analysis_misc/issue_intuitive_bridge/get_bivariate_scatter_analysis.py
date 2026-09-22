#!/usr/bin/env python3
"""Standalone script to perform bivariate scatter analysis linking MMD anchor variables to intuitive metrics.

Usage:
    python get_bivariate_scatter_analysis.py --config plans/config_ames_housing.toml
    python get_bivariate_scatter_analysis.py --x-vars YearBuilt --y-vars SalePrice --anchor-vars SaleCondition_Normal
"""
import argparse
import itertools
import os
import tomllib
import typing as ty
from pathlib import Path
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import pearsonr


class BivariateScatterConfig(BaseModel):
    """Configuration container for bivariate scatter analysis."""
    path_config_toml: str = Field(description="Path to pipeline TOML configuration file.")
    list_x_variables: ty.List[str] = Field(description="Features for X-axis (e.g. YearBuilt, GrLivArea).")
    list_y_variables: ty.List[str] = Field(description="Features for Y-axis (e.g. SalePrice, GrLivArea).")
    list_anchor_variables: ty.List[str] = Field(description="MMD anchor variables for color coding (e.g. SaleCondition_Normal).")
    directory_output: str = Field(description="Output directory for generated scatter plots.")
# end class BivariateScatterConfig


class DataWarehouseLoader:
    """Loads preprocessed feature tables and raw supplements from DuckDB and raw storage."""

    def __init__(self, path_toml_config: str):
        """Initializes loader with path to TOML configuration.

        Args:
            path_toml_config: Path to TOML configuration file.
        """
        self.path_toml_config = path_toml_config
        self.dict_toml = self._parse_toml_file(path_toml_config)
        # end def __init__

    def load_dataset_combined(self, list_required_columns: ty.List[str]) -> pd.DataFrame:
        """Loads dataset from DuckDB and merges any missing columns from raw CSV.

        Args:
            list_required_columns: List of column names requested for analysis.

        Returns:
            DataFrame containing sample_id, distribution_label, and requested columns.
        """
        path_db = self._extract_database_path()
        if not os.path.exists(path_db):
            raise FileNotFoundError(f"DuckDB database not found at {path_db}")
        # end if

        con = duckdb.connect(path_db, read_only=True)
        try:
            df_preprocessed = con.execute("SELECT * FROM dataset_features_preprocessed").df()
        finally:
            con.close()
        # end try

        cols_missing = [c for c in list_required_columns if c not in df_preprocessed.columns]
        if cols_missing:
            df_raw_supplement = self._load_raw_columns(cols_missing)
            if df_raw_supplement is not None:
                for col in cols_missing:
                    if col in df_raw_supplement.columns:
                        df_preprocessed[col] = df_raw_supplement[col].values
                    # end if
                # end for
            # end if
        # end if

        return df_preprocessed
        # end def load_dataset_combined

    def extract_distribution_labels(self) -> ty.Dict[str, str]:
        """Extracts human-readable distribution labels from metadata table.

        Returns:
            Dictionary mapping 'X' and 'Y' to descriptive strings.
        """
        dict_labels = {"X": "Pre-Crash (2006-2007)", "Y": "Post-Crash (2009-2010)"}
        path_db = self._extract_database_path()
        if not os.path.exists(path_db):
            return dict_labels
        # end if

        con = duckdb.connect(path_db, read_only=True)
        try:
            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
            if "dataset_metadata" in tables:
                records = con.execute("SELECT key, value FROM dataset_metadata").fetchall()
                meta_dict = dict(records)
                import json
                if "metadata_json" in meta_dict:
                    parsed_json = json.loads(meta_dict["metadata_json"])
                    if "label_x_description" in parsed_json:
                        dict_labels["X"] = str(parsed_json["label_x_description"])
                    # end if
                    if "label_y_description" in parsed_json:
                        dict_labels["Y"] = str(parsed_json["label_y_description"])
                    # end if
                # end if
            # end if
        finally:
            con.close()
        # end try
        return dict_labels
        # end def extract_distribution_labels

    def _parse_toml_file(self, path_toml: str) -> ty.Dict[str, ty.Any]:
        """Parses TOML file into dictionary."""
        with open(path_toml, "rb") as f_in:
            return tomllib.load(f_in)
        # end with
        # end def _parse_toml_file

    def _extract_database_path(self) -> str:
        """Resolves absolute path to DuckDB file from TOML config."""
        project_cfg = self.dict_toml.get("project", {})
        output_dir = project_cfg.get("output_directory", "")
        db_file = project_cfg.get("database_file", "analysis_warehouse.duckdb")
        return str(Path(output_dir) / db_file)
        # end def _extract_database_path

    def _load_raw_columns(self, columns_requested: ty.List[str]) -> ty.Optional[pd.DataFrame]:
        """Loads requested raw columns from raw CSV aligned by temporal split."""
        dataset_cfg = self.dict_toml.get("dataset", {}).get("ames_housing", {})
        raw_path = dataset_cfg.get("raw_data_path")
        if not raw_path or not os.path.exists(raw_path):
            return None
        # end if

        df_raw = pd.read_csv(raw_path)
        if "YrSold" not in df_raw.columns:
            return None
        # end if

        years_x = [2006, 2007]
        years_y = [2009, 2010]
        df_x = df_raw[df_raw["YrSold"].isin(years_x)].reset_index(drop=True)
        df_y = df_raw[df_raw["YrSold"].isin(years_y)].reset_index(drop=True)
        df_aligned = pd.concat([df_x, df_y], axis=0, ignore_index=True)

        cols_available = [c for c in columns_requested if c in df_aligned.columns]
        if not cols_available:
            return None
        # end if
        return df_aligned[cols_available]
        # end def _load_raw_columns
# end class DataWarehouseLoader


class BivariateScatterPlotter:
    """Generates comparative 2-panel scatter plots colored by MMD anchor variable states."""

    def __init__(
        self,
        color_anchor_active: str = "#E63946",
        color_anchor_inactive: str = "#457B9D"
    ):
        """Initializes plotter with color scheme for anchor states.

        Args:
            color_anchor_active: Color when anchor == 1.
            color_anchor_inactive: Color when anchor == 0.
        """
        self.color_active = color_anchor_active
        self.color_inactive = color_anchor_inactive
        # end def __init__

    def plot_bivariate_scatter(
        self,
        df_data: pd.DataFrame,
        name_x: str,
        name_y: str,
        name_anchor: str,
        labels_distribution: ty.Dict[str, str],
        directory_output: str
    ) -> str:
        """Plots two-panel canvas comparing (X vs Y) across Pre-Crash and Post-Crash markets.

        Args:
            df_data: DataFrame with features and distribution_label.
            name_x: Name of X-axis variable.
            name_y: Name of Y-axis variable.
            name_anchor: Name of MMD anchor variable used for color-coding.
            labels_distribution: Human-readable names for distribution labels 'X' and 'Y'.
            directory_output: Target directory where image is saved.

        Returns:
            Path of saved PNG plot.
        """
        os.makedirs(directory_output, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150, sharex=True, sharey=True)
        ax_dist_x, ax_dist_y = axes[0], axes[1]

        df_valid = df_data.dropna(subset=[name_x, name_y, name_anchor])
        df_dist_x = df_valid[df_valid["distribution_label"] == "X"]
        df_dist_y = df_valid[df_valid["distribution_label"] == "Y"]

        # Render Left: Pre-Crash (X)
        self._render_scatter_subplot(
            ax=ax_dist_x,
            df_era=df_dist_x,
            name_x=name_x,
            name_y=name_y,
            name_anchor=name_anchor,
            title_era=f"Distribution X: {labels_distribution.get('X', 'Pre-Crash')}"
        )

        # Render Right: Post-Crash (Y)
        self._render_scatter_subplot(
            ax=ax_dist_y,
            df_era=df_dist_y,
            name_x=name_x,
            name_y=name_y,
            name_anchor=name_anchor,
            title_era=f"Distribution Y: {labels_distribution.get('Y', 'Post-Crash')}"
        )

        fig.suptitle(
            f"Bivariate Scatter: {name_x} vs {name_y} | Colored by Anchor: {name_anchor}",
            fontsize=13,
            fontweight="bold",
            y=0.98
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        filename = f"scatter_{name_anchor}_{name_x}_vs_{name_y}.png"
        path_output = os.path.join(directory_output, filename)
        plt.savefig(path_output, dpi=150)
        plt.close(fig)

        return path_output
        # end def plot_bivariate_scatter

    def _render_scatter_subplot(
        self,
        ax: plt.Axes,
        df_era: pd.DataFrame,
        name_x: str,
        name_y: str,
        name_anchor: str,
        title_era: str
    ) -> None:
        """Renders scatter points and linear trend lines for active and inactive anchor subsets."""
        df_active = df_era[df_era[name_anchor] == 1.0]
        df_inactive = df_era[df_era[name_anchor] == 0.0]

        # Inactive anchor points (anchor == 0)
        if len(df_inactive) > 0:
            x_vals = df_inactive[name_x].to_numpy()
            y_vals = df_inactive[name_y].to_numpy()
            ax.scatter(
                x_vals, y_vals,
                color=self.color_inactive,
                alpha=0.45,
                s=28,
                label=f"{name_anchor}=0 (N={len(df_inactive)})",
                edgecolor="none"
            )
            self._render_trendline(ax, x_vals, y_vals, self.color_inactive, linestyle=":")
        # end if

        # Active anchor points (anchor == 1)
        if len(df_active) > 0:
            x_vals = df_active[name_x].to_numpy()
            y_vals = df_active[name_y].to_numpy()
            ax.scatter(
                x_vals, y_vals,
                color=self.color_active,
                alpha=0.75,
                s=36,
                label=f"{name_anchor}=1 (N={len(df_active)})",
                edgecolor="black",
                linewidth=0.5
            )
            self._render_trendline(ax, x_vals, y_vals, self.color_active, linestyle="-")
        # end if

        # Add stats text box
        text_stats = self._calculate_stats_summary(df_active, df_inactive, name_x, name_y, name_anchor)
        ax.text(
            0.04, 0.96, text_stats,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#ced4da", alpha=0.9)
        )

        ax.set_title(title_era, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(name_x, fontsize=10)
        ax.set_ylabel(name_y, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
        # end def _render_scatter_subplot

    def _render_trendline(
        self,
        ax: plt.Axes,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        color: str,
        linestyle: str = "-"
    ) -> None:
        """Fits and draws an OLS trendline over given coordinates."""
        if len(x_vals) > 2 and np.std(x_vals) > 1e-5:
            poly = np.polyfit(x_vals, y_vals, deg=1)
            x_trend = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            y_trend = np.polyval(poly, x_trend)
            ax.plot(x_trend, y_trend, color=color, linestyle=linestyle, lw=2.0)
        # end if
        # end def _render_trendline

    def _calculate_stats_summary(
        self,
        df_active: pd.DataFrame,
        df_inactive: pd.DataFrame,
        name_x: str,
        name_y: str,
        name_anchor: str
    ) -> str:
        """Calculates Pearson correlation for active and inactive groups."""
        lines = []
        if len(df_active) > 2 and np.std(df_active[name_x]) > 1e-5:
            r_act, _ = pearsonr(df_active[name_x], df_active[name_y])
            lines.append(f"{name_anchor}=1: r = {r_act:+.2f} (N={len(df_active)})")
        else:
            lines.append(f"{name_anchor}=1: N={len(df_active)}")
        # end if

        if len(df_inactive) > 2 and np.std(df_inactive[name_x]) > 1e-5:
            r_inact, _ = pearsonr(df_inactive[name_x], df_inactive[name_y])
            lines.append(f"{name_anchor}=0: r = {r_inact:+.2f} (N={len(df_inactive)})")
        else:
            lines.append(f"{name_anchor}=0: N={len(df_inactive)}")
        # end if

        return "\n".join(lines)
        # end def _calculate_stats_summary
# end class BivariateScatterPlotter


def execute_bivariate_analysis(config: BivariateScatterConfig) -> ty.List[str]:
    """Coordinates data loading, looping over variable combinations, and figure generation.

    Args:
        config: Validated BivariateScatterConfig instance.

    Returns:
        List of generated figure file paths.
    """
    loader = DataWarehouseLoader(path_toml_config=config.path_config_toml)
    all_needed_cols = list(set(config.list_x_variables + config.list_y_variables + config.list_anchor_variables))
    df_data = loader.load_dataset_combined(list_required_columns=all_needed_cols)
    dict_labels = loader.extract_distribution_labels()

    plotter = BivariateScatterPlotter()
    list_saved_figures: ty.List[str] = []

    print("\n" + "=" * 70)
    print(" Running Bivariate Scatter Analysis")
    print(f" X Variables:      {config.list_x_variables}")
    print(f" Y Variables:      {config.list_y_variables}")
    print(f" Anchor Variables: {config.list_anchor_variables}")
    print(f" Output Directory: {config.directory_output}")
    print("=" * 70)

    for x_var, y_var, anchor_var in itertools.product(
        config.list_x_variables,
        config.list_y_variables,
        config.list_anchor_variables
    ):
        if x_var == y_var:
            continue
        # end if
        if x_var not in df_data.columns:
            print(f" [SKIP] X variable '{x_var}' not found in dataset columns.")
            continue
        # end if
        if y_var not in df_data.columns:
            print(f" [SKIP] Y variable '{y_var}' not found in dataset columns.")
            continue
        # end if
        if anchor_var not in df_data.columns:
            print(f" [SKIP] Anchor variable '{anchor_var}' not found in dataset columns.")
            continue
        # end if

        path_fig = plotter.plot_bivariate_scatter(
            df_data=df_data,
            name_x=x_var,
            name_y=y_var,
            name_anchor=anchor_var,
            labels_distribution=dict_labels,
            directory_output=config.directory_output
        )
        list_saved_figures.append(path_fig)
        print(f" [GENERATED] {Path(path_fig).name}")
    # end for

    print(f"\nCompleted! Generated {len(list_saved_figures)} scatter plots in {config.directory_output}\n")
    return list_saved_figures
    # end def execute_bivariate_analysis


def parse_arguments_cli() -> BivariateScatterConfig:
    """Parses command-line arguments for bivariate scatter analysis.

    Returns:
        Validated BivariateScatterConfig instance.
    """
    parser = argparse.ArgumentParser(
        description="Plot comparative bivariate scatter plots colored by MMD anchor states across market eras."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/root/data_analysis_variable_selection/plans/config_ames_housing.toml",
        help="Path to pipeline TOML config file (default: plans/config_ames_housing.toml)."
    )
    parser.add_argument(
        "--x-vars",
        nargs="+",
        default=["YearBuilt", "GrLivArea"],
        help="List of X-axis variables (e.g. YearBuilt GrLivArea)."
    )
    parser.add_argument(
        "--y-vars",
        nargs="+",
        default=["SalePrice", "GrLivArea"],
        help="List of Y-axis variables (e.g. SalePrice GrLivArea)."
    )
    parser.add_argument(
        "--anchor-vars",
        nargs="+",
        default=["SaleCondition_Normal", "SaleType_New"],
        help="List of MMD anchor features to color-code (e.g. SaleCondition_Normal SaleType_New)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/data_analysis_variable_selection/analysis_misc/issue_intuitive_bridge/outputs/bivariate_scatter",
        help="Directory to save generated scatter plots."
    )

    args = parser.parse_args()

    return BivariateScatterConfig(
        path_config_toml=str(Path(args.config).resolve()),
        list_x_variables=list(args.x_vars),
        list_y_variables=list(args.y_vars),
        list_anchor_variables=list(args.anchor_vars),
        directory_output=str(Path(args.output_dir).resolve())
    )
    # end def parse_arguments_cli


def main() -> None:
    """Entry point for standalone CUI execution."""
    config = parse_arguments_cli()
    execute_bivariate_analysis(config)
    # end def main


if __name__ == "__main__":
    main()
# end if
