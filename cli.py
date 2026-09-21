#!/usr/bin/env python3
"""Top-level command-line execution script for the variable selection pipeline.
Usage:
    python cli.py --help
    python cli.py setup --config config.toml
    python cli.py preprocess --config config.toml
    python cli.py variable-detection --config config.toml
    python cli.py variable-analysis --config config.toml
    python cli.py generate-report --config config.toml
    python cli.py run-all --config config.toml
"""
from data_analysis_variable_selection.cli.main import app

if __name__ == "__main__":
    app()
