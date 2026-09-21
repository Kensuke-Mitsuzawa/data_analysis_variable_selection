import numpy as np
from data_analysis_variable_selection.correlation.analyzer import CorrelationAnalyzer
from data_analysis_variable_selection.clustering.clusterer import VariableClusterer


def test_correlation_and_clustering():
    np.random.seed(42)
    # Generate data with two distinct correlated blocks
    n = 100
    z = np.random.randn(n, 6)
    # Block 1: vars 0, 1, 2
    z[:, 1] = z[:, 0] + 0.1 * np.random.randn(n)
    z[:, 2] = z[:, 0] + 0.1 * np.random.randn(n)
    # Block 2: vars 3, 4, 5
    z[:, 4] = z[:, 3] + 0.1 * np.random.randn(n)
    z[:, 5] = z[:, 3] + 0.1 * np.random.randn(n)

    names = [f"var_{i}" for i in range(6)]

    # 1. Correlation Analyzer
    analyzer = CorrelationAnalyzer(threshold_edge_default=0.4)
    corr_res = analyzer.compute_matrix_correlation(z, names, method="pearson")

    assert corr_res.matrix_correlation.shape == (6, 6)
    assert len(corr_res.list_edges) > 0

    # 2. Variable Clusterer with anchor at var_0
    clusterer = VariableClusterer(num_clusters_default=2)
    clust_res = clusterer.cluster_variables_relationship(
        matrix_rel=corr_res.matrix_correlation,
        list_selected_anchors=[0],
        names_variables=names,
        num_clusters=2
    )

    # Augmented S_tilde should contain var_0's entire cluster
    assert 0 in clust_res.indices_augmented_s_tilde
    assert len(clust_res.indices_augmented_s_tilde) >= 3

    # Check score_related is computed for cluster containing anchor
    memberships = {m.name_variable: m for m in clust_res.list_memberships}
    assert memberships["var_0"].score_related is not None
    assert memberships["var_1"].score_related is not None
