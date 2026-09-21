from data_analysis_variable_selection.database.manager import DuckDBStorageManager
from data_analysis_variable_selection.common.models import (
    VariableSelectionResult,
    CorrelationEdge,
    CorrelationResult,
    ClusterMembership,
    VariableClusteringResult,
    PrototypeSampleRecord,
    PrototypeSampleResult,
)
import numpy as np


def test_duckdb_storage_manager():
    db = DuckDBStorageManager(":memory:")
    db.initialize_database_schema()

    # 1. Insert Selection
    sel_res = VariableSelectionResult(
        indices_selected=[0, 2],
        names_selected=["var_a", "var_c"],
        weights_selected=[1.25, 0.85]
    )
    db.insert_records_selection(sel_res)
    df_sel = db.fetch_records_sql("SELECT * FROM analysis_variable_selection")
    assert len(df_sel) == 2
    assert set(df_sel["name_variable"]) == {"var_a", "var_c"}

    # 2. Insert Correlation
    corr_res = CorrelationResult(
        matrix_correlation=np.eye(3),
        list_edges=[
            CorrelationEdge(id_variable_1=0, id_variable_2=1, correlation_score=0.45),
            CorrelationEdge(id_variable_1=0, id_variable_2=2, correlation_score=0.65),
        ],
        names_variables=["var_a", "var_b", "var_c"]
    )
    db.insert_records_correlation(corr_res)
    df_corr = db.fetch_records_sql("SELECT * FROM analysis_variable_correlation")
    assert len(df_corr) == 2

    # 3. Insert Clustering
    clust_res = VariableClusteringResult(
        list_memberships=[
            ClusterMembership(id_variable=0, name_variable="var_a", id_cluster=0, score_related=1.0),
            ClusterMembership(id_variable=1, name_variable="var_b", id_cluster=0, score_related=0.45),
            ClusterMembership(id_variable=2, name_variable="var_c", id_cluster=1, score_related=1.0),
        ],
        dict_cluster_to_variables={0: [0, 1], 1: [2]},
        indices_augmented_s_tilde=[0, 1, 2],
        names_augmented_s_tilde=["var_a", "var_b", "var_c"],
        dict_anchor_to_cluster={0: 0, 2: 1}
    )
    db.insert_records_clustering(clust_res)
    df_clust = db.fetch_records_sql("SELECT * FROM analysis_variable_clustering")
    assert len(df_clust) == 3

    # 4. Insert Prototypes
    proto_res = PrototypeSampleResult(
        list_prototype_records=[
            PrototypeSampleRecord(
                id_sample=10,
                label_class="X",
                is_prototype_for="X",
                distance_score=2.5,
                type_subspace="hat_S",
                dict_feature_values={"var_a": 1.0, "var_c": 0.5}
            )
        ]
    )
    db.insert_records_prototypes(proto_res)
    df_proto = db.fetch_records_sql("SELECT * FROM analysis_representative_samples")
    assert len(df_proto) == 1
    assert df_proto.iloc[0]["label_class"] == "X"

    db.close_connection_database()
