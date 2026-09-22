#!/usr/bin/env python3
"""Standalone script to analyze conditional univariate distributions between MMD anchor variables and intuitive target variables.

Usage:
    python get_univariate_distribution_conditioned.py --config plans/config_ames_housing.toml
    python get_univariate_distribution_conditioned.py --conditioning-vars SaleType_New --target-vars GrLivArea
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
from scipy.stats import gaussian_kde


class ConditionedDistributionConfig(BaseModel):
    """Configuration container for conditioned univariate distribution analysis."""
    path_config_toml: str = Field(description="Path to pipeline TOML configuration file.")
    list_conditioning_variables: ty.List[str] = Field(description="Conditioning features (e.g. SaleType_New).")
    list_target_variables: ty.List[str] = Field(description="Target features to analyze (e.g. GrLivArea).")
    directory_output: str = Field(description="Output directory for generated plots.")
# end class ConditionedDistributionConfig


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

        # Filter pre-crash and post-crash rows aligned with preprocessing
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


class ConditionedDistributionPlotter:
    """Generates comparative 2-panel canvas plots of target variables conditional on binary anchor states."""

    def __init__(
        self,
        color_x: str = "#E63946",
        color_y: str = "#1D3557"
    ):
        """Initializes plotter with color scheme.

        Args:
            color_x: Hex color for Pre-Crash distribution X.
            color_y: Hex color for Post-Crash distribution Y.
        """
        self.color_x = color_x
        self.color_y = color_y
        # end def __init__

    def plot_variable_pair_conditioned(
        self,
        df_data: pd.DataFrame,
        name_conditioning_variable: str,
        name_target_variable: str,
        labels_distribution: ty.Dict[str, str],
        directory_output: str
    ) -> str:
        """Plots two-panel canvas comparing target variable under condition=1 vs condition=0.

        Args:
            df_data: DataFrame with features and distribution_label.
            name_conditioning_variable: Name of conditioning column (e.g. SaleType_New).
            name_target_variable: Name of target column (e.g. GrLivArea).
            labels_distribution: Human-readable names for distribution labels 'X' and 'Y'.
            directory_output: Target directory where image is saved.

        Returns:
            Path of saved PNG plot.
        """
        os.makedirs(directory_output, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
        ax_cond1, ax_cond0 = axes[0], axes[1]

        # Condition 1: conditioning_variable == 1
        self._render_density_subplot(
            ax=ax_cond1,
            df_subset=df_data[df_data[name_conditioning_variable] == 1.0],
            name_target=name_target_variable,
            title_condition=f"{name_conditioning_variable} = 1 (Active Anchor State)",
            labels_distribution=labels_distribution
        )

        # Condition 0: conditioning_variable == 0
        self._render_density_subplot(
            ax=ax_cond0,
            df_subset=df_data[df_data[name_conditioning_variable] == 0.0],
            name_target=name_target_variable,
            title_condition=f"{name_conditioning_variable} = 0 (Inactive Anchor State)",
            labels_distribution=labels_distribution
        )

        fig.suptitle(
            f"Conditional Distribution: {name_target_variable} | Conditioned by {name_conditioning_variable}",
            fontsize=13,
            fontweight="bold",
            y=0.98
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.88])

        filename = f"conditioned_{name_conditioning_variable}_vs_{name_target_variable}.png"
        path_output = os.path.join(directory_output, filename)
        plt.savefig(path_output, dpi=150)
        plt.close(fig)

        return path_output
        # end def plot_variable_pair_conditioned

    def _render_density_subplot(
        self,
        ax: plt.Axes,
        df_subset: pd.DataFrame,
        name_target: str,
        title_condition: str,
        labels_distribution: ty.Dict[str, str]
    ) -> None:
        """Renders comparative histogram and KDE curves for X and Y in the provided axes."""
        vals_x = df_subset[df_subset["distribution_label"] == "X"][name_target].dropna().to_numpy()
        vals_y = df_subset[df_subset["distribution_label"] == "Y"][name_target].dropna().to_numpy()

        n_x, n_y = len(vals_x), len(vals_y)
        label_x_full = f"{labels_distribution.get('X', 'Distribution X')} (N={n_x})"
        label_y_full = f"{labels_distribution.get('Y', 'Distribution Y')} (N={n_y})"

        if n_x > 0 and n_y > 0:
            mean_x, mean_y = float(np.mean(vals_x)), float(np.mean(vals_y))
            med_x, med_y = float(np.median(vals_x)), float(np.median(vals_y))
            pct_shift = ((mean_y - mean_x) / mean_x) * 100.0 if mean_x != 0 else 0.0

            # Determine bins
            combined = np.concatenate([vals_x, vals_y])
            bins = np.linspace(np.min(combined), np.max(combined), 25)

            # Histograms
            ax.hist(vals_x, bins=bins, density=True, alpha=0.35, color=self.color_x, edgecolor=self.color_x)
            ax.hist(vals_y, bins=bins, density=True, alpha=0.35, color=self.color_y, edgecolor=self.color_y)

            # KDE curves
            if len(np.unique(vals_x)) > 3:
                kde_x = gaussian_kde(vals_x)
                x_grid = np.linspace(np.min(combined), np.max(combined), 200)
                ax.plot(x_grid, kde_x(x_grid), color=self.color_x, lw=2.2, label=label_x_full)
            else:
                ax.plot([], [], color=self.color_x, lw=2.2, label=label_x_full)
            # end if

            if len(np.unique(vals_y)) > 3:
                kde_y = gaussian_kde(vals_y)
                x_grid = np.linspace(np.min(combined), np.max(combined), 200)
                ax.plot(x_grid, kde_y(x_grid), color=self.color_y, lw=2.2, label=label_y_full)
            else:
                ax.plot([], [], color=self.color_y, lw=2.2, label=label_y_full)
            # end if

            # Vertical lines for medians
            ax.axvline(med_x, color=self.color_x, linestyle="--", alpha=0.8, lw=1.5, label=f"Median X: {med_x:,.1f}")
            ax.axvline(med_y, color=self.color_y, linestyle="--", alpha=0.8, lw=1.5, label=f"Median Y: {med_y:,.1f}")

            # Stats annotation box
            text_stats = (
                f"Shift in Mean: {pct_shift:+.2f}%\n"
                f"Mean X: {mean_x:,.1f} | Mean Y: {mean_y:,.1f}\n"
                f"Med X:  {med_x:,.1f} | Med Y:  {med_y:,.1f}"
            )
            ax.text(
                0.97, 0.95, text_stats,
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
            )
        elif n_x > 0:
            ax.hist(vals_x, bins=20, density=True, alpha=0.5, color=self.color_x, label=label_x_full)
            ax.text(0.5, 0.5, "Post-Crash (Y) N=0", transform=ax.transAxes, ha="center")
        elif n_y > 0:
            ax.hist(vals_y, bins=20, density=True, alpha=0.5, color=self.color_y, label=label_y_full)
            ax.text(0.5, 0.5, "Pre-Crash (X) N=0", transform=ax.transAxes, ha="center")
        else:
            ax.text(0.5, 0.5, "No samples in this condition", transform=ax.transAxes, ha="center")
        # end if

        # Title positioned with pad to avoid overlap with legend box above the plot box
        ax.set_title(title_condition, fontsize=11, fontweight="bold", pad=42)
        ax.set_xlabel(name_target, fontsize=9.5)
        ax.set_ylabel("Probability Density", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        # Position legend box on/above the plot box with 2 columns
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc"
        )
        # end def _render_density_subplot
# end class ConditionedDistributionPlotter


def execute_conditioned_analysis(config: ConditionedDistributionConfig) -> ty.List[str]:
    """Coordinates data loading, looping over variable combinations, and figure generation.

    Args:
        config: Validated ConditionedDistributionConfig instance.

    Returns:
        List of generated figure file paths.
    """
    loader = DataWarehouseLoader(path_toml_config=config.path_config_toml)
    all_needed_cols = list(set(config.list_conditioning_variables + config.list_target_variables))
    df_data = loader.load_dataset_combined(list_required_columns=all_needed_cols)
    dict_labels = loader.extract_distribution_labels()

    plotter = ConditionedDistributionPlotter()
    list_saved_figures: ty.List[str] = []

    print("\n" + "=" * 70)
    print(" Running Conditioned Univariate Distribution Analysis")
    print(f" Conditioning Variables: {config.list_conditioning_variables}")
    print(f" Target Variables:       {config.list_target_variables}")
    print(f" Output Directory:       {config.directory_output}")
    print("=" * 70)

    for cond_var, target_var in itertools.product(
        config.list_conditioning_variables,
        config.list_target_variables
    ):
        if cond_var not in df_data.columns:
            print(f" [SKIP] Conditioning variable '{cond_var}' not found in dataset columns.")
            continue
        # end if
        if target_var not in df_data.columns:
            print(f" [SKIP] Target variable '{target_var}' not found in dataset columns.")
            continue
        # end if

        path_fig = plotter.plot_variable_pair_conditioned(
            df_data=df_data,
            name_conditioning_variable=cond_var,
            name_target_variable=target_var,
            labels_distribution=dict_labels,
            directory_output=config.directory_output
        )
        list_saved_figures.append(path_fig)
        print(f" [GENERATED] {Path(path_fig).name}")
    # end for

    print(f"\nCompleted! Generated {len(list_saved_figures)} plots in {config.directory_output}\n")
    return list_saved_figures
    # end def execute_conditioned_analysis


def parse_arguments_cli() -> ConditionedDistributionConfig:
    """Parses command-line arguments for conditioned univariate distribution analysis.

    Returns:
        Validated ConditionedDistributionConfig instance.
    """
    parser = argparse.ArgumentParser(
        description="Plot comparative univariate distributions of target variables conditioned on MMD anchor features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/root/data_analysis_variable_selection/plans/config_ames_housing.toml",
        help="Path to pipeline TOML config file (default: plans/config_ames_housing.toml)."
    )
    parser.add_argument(
        "--conditioning-vars",
        nargs="+",
        default=["SaleType_New", "SaleCondition_Normal", "SaleCondition_Partial"],
        help="List of conditioning anchor variables (e.g. SaleType_New SaleCondition_Normal)."
    )
    parser.add_argument(
        "--target-vars",
        nargs="+",
        default=["SalePrice", "GrLivArea", "LotArea", "YearBuilt", "TotalBsmtSF"],
        help="List of target continuous/intuitive variables (e.g. GrLivArea LotArea)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/data_analysis_variable_selection/analysis_misc/issue_intuitive_bridge/outputs/univariate_conditioned",
        help="Directory to save generated comparison plots."
    )

    args = parser.parse_args()

    return ConditionedDistributionConfig(
        path_config_toml=str(Path(args.config).resolve()),
        list_conditioning_variables=list(args.conditioning_vars),
        list_target_variables=list(args.target_vars),
        directory_output=str(Path(args.output_dir).resolve())
    )
    # end def parse_arguments_cli


def main() -> None:
    """Entry point for standalone CUI execution."""
    config = parse_arguments_cli()
    execute_conditioned_analysis(config)
    # end def main


if __name__ == "__main__":
    main()
# end if
