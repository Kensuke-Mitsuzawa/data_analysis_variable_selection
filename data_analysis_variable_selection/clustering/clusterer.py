import typing as ty
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..common.models import ClusterMembership, VariableClusteringResult


class VariableClusterer:
    """Partitions variables into semantic clusters based on relationship matrices and expands anchor sets.
    """

    def __init__(self, num_clusters_default: int = 5):
        """Initializes the variable clusterer.

        Args:
            num_clusters_default: Default number of clusters when not specified.
        """
        self.num_clusters_default = num_clusters_default
        # end def __init__

    def cluster_variables_relationship(
        self,
        matrix_rel: np.ndarray,
        list_selected_anchors: ty.List[int],
        names_variables: ty.List[str],
        num_clusters: ty.Optional[int] = None
    ) -> VariableClusteringResult:
        """Partitions variables into clusters and identifies augmented variable sets and relatedness scores.

        Args:
            matrix_rel: Relationship/correlation matrix Sigma (d x d).
            list_selected_anchors: List of selected anchor variable indices (hat_S).
            names_variables: List of variable names.
            num_clusters: Number of clusters to form.

        Returns:
            VariableClusteringResult holding memberships, augmented variables S_tilde, and cluster mappings.
        """
        num_variables = matrix_rel.shape[0]

        if num_variables <= 1:
            cluster_assignments = np.zeros(num_variables, dtype=int)
        else:
            k = num_clusters if num_clusters is not None else min(self.num_clusters_default, num_variables)
            k = max(1, min(k, num_variables))

            # Convert correlation/relationship into affinity / distance
            matrix_dist = 1.0 - np.clip(np.abs(matrix_rel), 0.0, 1.0)
            np.fill_diagonal(matrix_dist, 0.0)

            clustering_model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average"
            )
            cluster_assignments = clustering_model.fit_predict(matrix_dist)
        # end if

        dict_cluster_to_variables: ty.Dict[int, ty.List[int]] = {}
        for idx_var, id_clust in enumerate(cluster_assignments):
            dict_cluster_to_variables.setdefault(int(id_clust), []).append(idx_var)
        # end for idx_var

        set_anchor_indices = set(list_selected_anchors)
        dict_anchor_to_cluster: ty.Dict[int, int] = {}
        set_anchor_clusters: ty.Set[int] = set()

        for anchor_idx in list_selected_anchors:
            if 0 <= anchor_idx < num_variables:
                cluster_id = int(cluster_assignments[anchor_idx])
                dict_anchor_to_cluster[anchor_idx] = cluster_id
                set_anchor_clusters.add(cluster_id)
            # end if
        # end for anchor_idx

        # Augmented variables S_tilde: Union of clusters containing at least one anchor
        indices_augmented_s_tilde: ty.List[int] = []
        for cluster_id in sorted(set_anchor_clusters):
            indices_augmented_s_tilde.extend(dict_cluster_to_variables.get(cluster_id, []))
        # end for cluster_id

        # If no anchors were provided or matched, fallback to all variables or empty
        if not indices_augmented_s_tilde and list_selected_anchors:
            indices_augmented_s_tilde = list(list_selected_anchors)
        # end if
        indices_augmented_s_tilde = sorted(list(set(indices_augmented_s_tilde)))
        names_augmented_s_tilde = [names_variables[idx] for idx in indices_augmented_s_tilde]

        # Compute membership and score_related
        list_memberships: ty.List[ClusterMembership] = []
        for idx_var, name_var in enumerate(names_variables):
            cluster_id = int(cluster_assignments[idx_var])
            cluster_vars = dict_cluster_to_variables.get(cluster_id, [])
            anchors_in_cluster = [a for a in cluster_vars if a in set_anchor_indices]

            score_related: ty.Optional[float] = None
            if anchors_in_cluster:
                # Max absolute correlation with an anchor in this cluster
                scores = [abs(float(matrix_rel[idx_var, a])) for a in anchors_in_cluster]
                score_related = float(max(scores))
            # end if

            list_memberships.append(
                ClusterMembership(
                    id_variable=idx_var,
                    name_variable=name_var,
                    id_cluster=cluster_id,
                    score_related=score_related,
                )
            )
        # end for idx_var

        return VariableClusteringResult(
            list_memberships=list_memberships,
            dict_cluster_to_variables=dict_cluster_to_variables,
            indices_augmented_s_tilde=indices_augmented_s_tilde,
            names_augmented_s_tilde=names_augmented_s_tilde,
            dict_anchor_to_cluster=dict_anchor_to_cluster,
        )
        # end def cluster_variables_relationship
# end class VariableClusterer
