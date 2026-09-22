"""Static color palette definitions for two-sample variable selection visualizations.
Provides unified color constants for distributions X (Red) and Y (Blue) across all plots and legends.
"""
import typing as ty


class ReportVisualPalette:
    """Static repository of canonical visualization colors ensuring visual and cognitive consistency across reports.
    """
    # Distribution X is canonically Red across all report plots
    COLOR_DISTRIBUTION_X: str = "#dc2626"       # Tailwind Red-600 / Vivid Crimson for light background figures
    COLOR_DISTRIBUTION_X_DARK: str = "#f43f5e"  # Rose-Red 500 for dark background figures (e.g. radar charts)

    # Distribution Y is canonically Blue across all report plots
    COLOR_DISTRIBUTION_Y: str = "#2563eb"       # Tailwind Blue-600 / Vivid Royal Blue for light background figures
    COLOR_DISTRIBUTION_Y_DARK: str = "#38bdf8"  # Sky-Blue 400 for dark background figures (e.g. radar charts)

    # Accent and auxiliary styling constants
    COLOR_ACCENT_AMBER: str = "#f59e0b"         # Amber for anchor variable highlights
    COLOR_DOT_BORDER: str = "#ffffff"           # High-contrast white border for polygon vertex markers
# end class ReportVisualPalette


# Static module-level constants for direct import
COLOR_DISTRIBUTION_X: str = ReportVisualPalette.COLOR_DISTRIBUTION_X
COLOR_DISTRIBUTION_Y: str = ReportVisualPalette.COLOR_DISTRIBUTION_Y
COLOR_DISTRIBUTION_X_DARK: str = ReportVisualPalette.COLOR_DISTRIBUTION_X_DARK
COLOR_DISTRIBUTION_Y_DARK: str = ReportVisualPalette.COLOR_DISTRIBUTION_Y_DARK
