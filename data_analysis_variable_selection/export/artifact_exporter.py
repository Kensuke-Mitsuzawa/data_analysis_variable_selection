import os
import typing as ty
from ..database.manager import DuckDBStorageManager
from ..common.models import (
    CorrelationResult,
    VariableClusteringResult,
    VariableSelectionResult,
    PrototypeSampleResult,
)
from .tabular_exporter import TabularArtifactExporter
from ..visualization.constellation_graph import ConstellationGraphPlotter
from ..visualization.tornado_chart import TornadoChartPlotter
from ..visualization.radar_chart import PersonaRadarChartPlotter


class ArtifactExporter:
    """Master orchestrator generating and saving all tabular and visual deliverables to the output directory.
    """

    def __init__(self):
        """Initializes the artifact exporter with tabular and visual sub-exporters.
        """
        self.tabular_exporter = TabularArtifactExporter()
        self.constellation_plotter = ConstellationGraphPlotter()
        self.tornado_plotter = TornadoChartPlotter()
        self.radar_plotter = PersonaRadarChartPlotter()
        # end def __init__

    def export_all_artifacts(
        self,
        db_manager: DuckDBStorageManager,
        correlation_result: CorrelationResult,
        clustering_result: VariableClusteringResult,
        selection_result: VariableSelectionResult,
        prototype_result: PrototypeSampleResult,
        directory_output: str
    ) -> ty.Dict[str, ty.Any]:
        """Generates all CSV files and charts and writes them to the specified directory.

        Args:
            db_manager: Connected DuckDBStorageManager containing populated tables.
            correlation_result: Correlation result for network graph.
            clustering_result: Clustering result for tornado and radar charts.
            selection_result: Variable selection result with anchor variables.
            prototype_result: Extracted prototype samples for radar charts.
            directory_output: Output folder for deliverables.

        Returns:
            Dictionary mapping artifact names to their file paths.
        """
        os.makedirs(directory_output, exist_ok=True)
        dict_artifacts: ty.Dict[str, ty.Any] = {}

        # 1. Tabular CSV Artifacts
        path_cluster_csv = os.path.join(directory_output, "cluster_analysis.csv")
        self.tabular_exporter.export_cluster_analysis_csv(db_manager, path_cluster_csv)
        dict_artifacts["cluster_analysis_csv"] = path_cluster_csv

        path_samples_csv = os.path.join(directory_output, "representative_samples.csv")
        self.tabular_exporter.export_representative_samples_csv(db_manager, path_samples_csv)
        dict_artifacts["representative_samples_csv"] = path_samples_csv

        # 2. Constellation Network Graph
        path_constellation = os.path.join(directory_output, "constellation_network.png")
        self.constellation_plotter.plot_constellation_graph(
            correlation_result=correlation_result,
            clustering_result=clustering_result,
            selection_result=selection_result,
            path_output_image=path_constellation,
        )
        dict_artifacts["constellation_network"] = path_constellation

        # 3. Thematic Tornado Bar Charts
        list_tornado = self.tornado_plotter.plot_thematic_tornado_charts(
            clustering_result=clustering_result,
            selection_result=selection_result,
            directory_output=directory_output,
        )
        dict_artifacts["tornado_charts"] = list_tornado

        # 4. Persona Radar Charts
        list_radar = self.radar_plotter.plot_persona_radar_charts(
            prototype_result=prototype_result,
            clustering_result=clustering_result,
            selection_result=selection_result,
            directory_output=directory_output,
        )
        dict_artifacts["persona_radar_charts"] = list_radar

        return dict_artifacts
        # end def export_all_artifacts
# end class ArtifactExporter
