import os
import tempfile
import numpy as np
import pandas as pd

from data_analysis_variable_selection.common.models import (
    CorrelationEdge,
    CorrelationResult,
    ClusterMembership,
    VariableClusteringResult,
    VariableSelectionResult,
    PrototypeSampleRecord,
    PrototypeSampleResult,
)
from data_analysis_variable_selection.visualization.constellation_graph import ConstellationGraphPlotter
from data_analysis_variable_selection.visualization.tornado_chart import TornadoChartPlotter
from data_analysis_variable_selection.visualization.radar_chart import PersonaRadarChartPlotter
from data_analysis_variable_selection.database.manager import DuckDBStorageManager
from data_analysis_variable_selection.export.tabular_exporter import TabularArtifactExporter


def test_visualizations_and_exports():
    with tempfile.TemporaryDirectory() as temp_dir:
        names = [f"feat_{i}" for i in range(12)]
        corr_matrix = np.eye(12)
        edges = [
            CorrelationEdge(id_variable_1=0, id_variable_2=1, correlation_score=0.7),
            CorrelationEdge(id_variable_1=0, id_variable_2=2, correlation_score=0.6),
            CorrelationEdge(id_variable_1=1, id_variable_2=3, correlation_score=0.5),
        ]
        corr_res = CorrelationResult(
            matrix_correlation=corr_matrix,
            list_edges=edges,
            names_variables=names
        )

        sel_res = VariableSelectionResult(
            indices_selected=[0],
            names_selected=["feat_0"],
            weights_selected=[1.5]
        )

        memberships = []
        for i in range(12):
            cid = 0 if i < 6 else 1
            score = 1.0 - 0.1 * i if cid == 0 else None
            memberships.append(
                ClusterMembership(id_variable=i, name_variable=names[i], id_cluster=cid, score_related=score)
            )
        # end for

        clust_res = VariableClusteringResult(
            list_memberships=memberships,
            dict_cluster_to_variables={0: list(range(6)), 1: list(range(6, 12))},
            indices_augmented_s_tilde=list(range(6)),
            names_augmented_s_tilde=names[:6],
            dict_anchor_to_cluster={0: 0}
        )

        proto_res = PrototypeSampleResult(
            list_prototype_records=[
                PrototypeSampleRecord(
                    id_sample=1,
                    label_class="X",
                    is_prototype_for="X",
                    distance_score=2.0,
                    type_subspace="hat_S",
                    dict_feature_values={n: float(i * 2) for i, n in enumerate(names)}
                ),
                PrototypeSampleRecord(
                    id_sample=2,
                    label_class="Y",
                    is_prototype_for="Y",
                    distance_score=1.8,
                    type_subspace="hat_S",
                    dict_feature_values={n: float(i * 1.5) for i, n in enumerate(names)}
                )
            ]
        )

        # 1. Constellation Graph
        constellation = ConstellationGraphPlotter()
        path_const = os.path.join(temp_dir, "constellation.png")
        out_const = constellation.plot_constellation_graph(corr_res, clust_res, sel_res, path_const)
        assert os.path.exists(out_const)
        assert os.path.getsize(out_const) > 0

        # 2. Tornado Chart
        tornado = TornadoChartPlotter(limit_top_variables=5)
        out_tornados = tornado.plot_thematic_tornado_charts(clust_res, sel_res, temp_dir)
        assert len(out_tornados) == 1
        assert os.path.exists(out_tornados[0])

        # 3. Radar Chart (assert strict <= 8 axes limit)
        radar = PersonaRadarChartPlotter(limit_axes_max=8)
        out_radars = radar.plot_persona_radar_charts(proto_res, clust_res, sel_res, temp_dir)
        assert len(out_radars) == 1
        assert os.path.exists(out_radars[0])

        # 4. Tabular CSV export
        db = DuckDBStorageManager(":memory:")
        db.initialize_database_schema()
        db.insert_records_clustering(clust_res)
        db.insert_records_prototypes(proto_res)

        tab_exporter = TabularArtifactExporter()
        path_c_csv = os.path.join(temp_dir, "cluster.csv")
        tab_exporter.export_cluster_analysis_csv(db, path_c_csv)
        assert os.path.exists(path_c_csv)
        df_c = pd.read_csv(path_c_csv)
        assert len(df_c) == 12

        path_s_csv = os.path.join(temp_dir, "samples.csv")
        tab_exporter.export_representative_samples_csv(db, path_s_csv)
        assert os.path.exists(path_s_csv)
        df_s = pd.read_csv(path_s_csv)
        assert len(df_s) == 2

        db.close_connection_database()
