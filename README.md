# Data Analysis Variable Selection

A statistical analytics engine and pipeline for two-sample variable selection, relationship analysis, and reporting across tabular datasets (e.g. Ames Housing). It detects key distributional shift drivers using **MMD (Maximum Mean Discrepancy)** or **1D-Wasserstein distance**, clusters correlated features into themes, extracts exemplar prototypes, and exports executive reports.

---

## 1. Installation

The project uses `uv` for dependency and virtual environment management.

### Clone and Install

```bash
# Clone the repository
git clone <repository_url>
cd project-data-analysis-variable-selection

# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Alternatively, with standard `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 2. Running the CUI Pipeline

The pipeline is configured via a TOML file (e.g. `config_ames_housing.toml`) and operated through a Typer-powered Command-Line Interface (CUI).

You can run commands using `python cli.py <command>` or as a module `python -m data_analysis_variable_selection.cli <command>`.

### Run Full Pipeline End-to-End

To execute all 5 stages in sequence:

```bash
python cli.py run-all --config config_ames_housing.toml
```

### Run Pipeline Stages Step-by-Step

You can also run each stage independently:

```bash
# 1. Setup raw dataset (download and extract)
python cli.py setup --config config_ames_housing.toml

# 2. Clean features, split X vs Y, save to DuckDB and cache arrays
python cli.py preprocess --config config_ames_housing.toml

# 3. Detect anchor variables using MMD or 1D-Wasserstein
python cli.py variable-detection --config config_ames_housing.toml

# 4. Analyze correlation/precision matrix, cluster features, and extract prototypes
python cli.py variable-analysis --config config_ames_housing.toml

# 5. Generate Excel workbook (.xlsx), Markdown report (report.md), and visual plots
python cli.py generate-report --config config_ames_housing.toml
```

To see all available commands and flags:

```bash
python cli.py --help
```

---

## 3. Deliverables & Outputs

When the pipeline finishes, the following artifacts are generated in the configured `output_directory`:

- `report.md`: Executive Markdown summary linking all findings and embedded charts.
- `analysis_report.xlsx`: Multi-sheet Excel workbook containing Anchor Variables, Cluster Themes, Pairwise Correlations, and Exemplar Prototypes.
- `constellation_network.png`: Force-directed graph showing relationships among variables.
- `tornado_cluster_*.png`: Horizontal bar charts ranking top features per cluster.
- `persona_radar_cluster_*.png`: Radar charts contrasting central prototypes of $X$ vs. $Y$.
- `analysis_warehouse.duckdb`: DuckDB database storing normalized analysis tables and preprocessed features.
- `preprocessed_features.npz`: NumPy archive containing preprocessed distributions $(X, Y)$ and feature names.
