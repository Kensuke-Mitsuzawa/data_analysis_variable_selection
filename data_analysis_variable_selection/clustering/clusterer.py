import abc
import typing as ty
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..common.models import ClusterMembership, VariableClusteringResult


class BaseVariableClusterer(abc.ABC):
    """Abstract base executor for partitioning variables into semantic clusters based on relationship matrices.
    """

    @abc.abstractmethod
    def cluster_variables_relationship(
        self,
        matrix_rel: np.ndarray,
        list_selected_anchors: ty.List[int],
        names_variables: ty.List[str],
        num_clusters: ty.Optional[int] = None,
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
        pass
        # end def cluster_variables_relationship
# end class BaseVariableClusterer


class AnchorGuidedThematicClusterer(BaseVariableClusterer):
    r"""Partitions variables into semantic clusters guided by anchor features with collinearity reduction.

    Algorithm: Anchor-Guided Thematic Clustering with Collinearity Reduction
    ========================================================================

    Mathematical Formulation:
    -------------------------
    1. Notation & Inputs:
       - Let $V = \{1, 2, \dots, d\}$ be the full set of variables.
       - Let $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$ be the relationship matrix (empirical correlation
         or Graphical Lasso precision matrix), where $|\Sigma_{ij}| \in [0, 1]$.
       - Let $\hat{S} \subset V$ be the set of $m = |\hat{S}|$ anchor variables discovered by the MMD
         variable selection algorithm.
       - Let $\tau_{\text{collinear}} \in [0, 1]$ be the collinearity threshold (default: 0.5).
       - Let $\tau_{\text{augment}} \in [0, 1]$ be the augmentation threshold (default: 0.15).

    2. Step 1: Anchor Collinearity Graph & Theme Partitioning:
       Anchor variables measuring identical or highly collinear concepts are grouped together.
       An undirected anchor graph $G_{\hat{S}} = (\hat{S}, E_{\hat{S}})$ is constructed with edge set:

       $$E_{\hat{S}} = \{ (s_a, s_b) \in \hat{S} \times \hat{S} \mid a \neq b, |\Sigma_{s_a, s_b}| \ge \tau_{\text{collinear}} \}$$

       The connected components of $G_{\hat{S}}$ define $K$ disjoint anchor themes:

       $$\hat{S} = \bigcup_{k=1}^K \hat{S}_k, \quad \text{where } \hat{S}_k \cap \hat{S}_l = \emptyset \; (\forall k \neq l)$$

    3. Step 2: Contextual Variable Affinity:
       For every non-anchor variable $v \in V \setminus \hat{S}$, its affinity to each anchor theme $k$ is
       evaluated as the maximum absolute relationship to any anchor in that theme:

       $$\rho(v, k) = \max_{s \in \hat{S}_k} |\Sigma_{v, s}|, \quad \forall k \in \{1, \dots, K\}$$

       The best theme candidate is:

       $$k^*(v) = \arg\max_{k \in \{1, \dots, K\}} \rho(v, k), \quad \text{with score } \rho^*(v) = \rho(v, k^*(v))$$

    4. Step 3: Thresholded Membership Assignment:
       Variables with sufficient affinity to at least one theme are assigned to $k^*(v)$.
       Uncorrelated ambient variables are directed into background Cluster 0:

       $$C(v) = \begin{cases}
       k^*(v), & \text{if } v \notin \hat{S} \text{ and } \rho^*(v) \ge \tau_{\text{augment}} \\
       k, & \text{if } v \in \hat{S}_k \\
       0, & \text{if } v \notin \hat{S} \text{ and } \rho^*(v) < \tau_{\text{augment}} \quad (\text{Background})
       \end{cases}$$

    5. Step 4: Relatedness Score & Augmented Feature Set ($S_\text{tilde}$):
       The relatedness score is defined as:

       $$\text{score\_related}(v) = \begin{cases}
       1.0, & \text{if } v \in \hat{S} \\
       \rho^*(v), & \text{if } C(v) \in \{1, \dots, K\} \\
       \text{None (NA)}, & \text{if } C(v) = 0
       \end{cases}$$

       The augmented feature set $\tilde{S}$ is the union of all active anchor theme clusters:

       $$\tilde{S} = \bigcup_{k=1}^K \{ v \in V \mid C(v) = k \} = \hat{S} \cup \{ v \in V \setminus \hat{S} \mid \rho^*(v) \ge \tau_{\text{augment}} \}$$

    Scientific References & Theoretical Grounding:
    ---------------------------------------------
    1. Seeded / Semi-Supervised Clustering:
       - Basu, S., Banerjee, A., & Mooney, R. (2002). "Semi-supervised clustering by seeded KMeans."
         In Proceedings of the 19th International Conference on Machine Learning (ICML), pp. 19-26.
       - Wagstaff, K., Cardie, C., Rogers, S., & Schroedl, S. (2001). "Constrained K-means clustering
         with background knowledge." In ICML, pp. 577-584.

    2. Correlated Feature Grouping in High-Dimensional Statistics:
       - Bühlmann, P., Rütimann, P., van de Geer, S., & Zhang, C.-H. (2013). "Correlated variables in regression: Clustering and sparse estimation." Journal of Statistical Planning and Inference, 143(11), 1835-1858.
       - Toloşi, L., & Lengauer, T. (2011). "Classification with correlated features: unreliability of feature ranking and solutions." Bioinformatics, 27(14), 1986-1994.

    3. Hub / Driver-Based Module Assignment (WGCNA):
       - Zhang, B., & Horvath, S. (2005). "A general framework for weighted gene co-expression network
         analysis." Statistical Applications in Genetics and Molecular Biology, 4(1), Article 17.

    4. Exemplar / Medoid Clustering:
       - Kaufman, L., & Rousseeuw, P. J. (1990). "Finding Groups in Data: An Introduction to Cluster
         Analysis." John Wiley & Sons.
    """

    def __init__(
        self,
        num_clusters_default: int = 5,
        threshold_collinear: float = 0.5,
        threshold_augment: float = 0.15,
    ):
        """Initializes the anchor-guided thematic clusterer.

        Args:
            num_clusters_default: Default number of clusters when not specified.
            threshold_collinear: Absolute correlation threshold to group collinear anchors into the same theme.
            threshold_augment: Absolute correlation threshold to include an ambient variable in an anchor theme.
        """
        self.num_clusters_default = num_clusters_default
        self.threshold_collinear = threshold_collinear
        self.threshold_augment = threshold_augment
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
        valid_anchors = [a for a in list_selected_anchors if 0 <= a < num_variables]

        if num_variables <= 1 or not valid_anchors:
            # Fallback when no valid anchors are present
            if num_variables <= 1:
                cluster_assignments = np.zeros(num_variables, dtype=int)
            else:
                k = num_clusters if num_clusters is not None else min(self.num_clusters_default, num_variables)
                k = max(1, min(k, num_variables))
                matrix_dist = 1.0 - np.clip(np.abs(matrix_rel), 0.0, 1.0)
                np.fill_diagonal(matrix_dist, 0.0)
                clustering_model = AgglomerativeClustering(
                    n_clusters=k,
                    metric="precomputed",
                    linkage="complete"
                )
                cluster_assignments = clustering_model.fit_predict(matrix_dist)
            # end if

            dict_cluster_to_variables: ty.Dict[int, ty.List[int]] = {}
            for idx_var, id_clust in enumerate(cluster_assignments):
                dict_cluster_to_variables.setdefault(int(id_clust), []).append(idx_var)
            # end for idx_var

            indices_augmented_s_tilde = list(range(num_variables)) if not list_selected_anchors else list(valid_anchors)
            names_augmented_s_tilde = [names_variables[idx] for idx in indices_augmented_s_tilde]

            list_memberships = [
                ClusterMembership(
                    id_variable=idx_var,
                    name_variable=name_var,
                    id_cluster=int(cluster_assignments[idx_var]),
                    score_related=None,
                )
                for idx_var, name_var in enumerate(names_variables)
            ]

            return VariableClusteringResult(
                list_memberships=list_memberships,
                dict_cluster_to_variables=dict_cluster_to_variables,
                indices_augmented_s_tilde=indices_augmented_s_tilde,
                names_augmented_s_tilde=names_augmented_s_tilde,
                dict_anchor_to_cluster={},
            )
        # end if

        # 1. Group anchors into collinear themes
        anchor_groups: ty.List[ty.List[int]] = []
        for a in valid_anchors:
            assigned = False
            for group in anchor_groups:
                if any(abs(float(matrix_rel[a, g_a])) >= self.threshold_collinear for g_a in group):
                    group.append(a)
                    assigned = True
                    break
                # end if
            # end for group
            if not assigned:
                anchor_groups.append([a])
            # end if
        # end for a

        # Theme clusters are indexed 1, 2, ..., len(anchor_groups)
        # Cluster 0 is reserved for unanchored background variables
        dict_cluster_to_variables = {}
        dict_anchor_to_cluster = {}
        cluster_assignments = np.zeros(num_variables, dtype=int)
        scores_related: ty.List[ty.Optional[float]] = [None] * num_variables

        set_anchors = set(valid_anchors)
        for theme_idx, group in enumerate(anchor_groups, start=1):
            for a in group:
                dict_anchor_to_cluster[a] = theme_idx
                cluster_assignments[a] = theme_idx
                scores_related[a] = 1.0
            # end for a
        # end for theme_idx

        # 2. Assign remaining non-anchor variables to best anchor theme if correlation >= threshold
        for i in range(num_variables):
            if i in set_anchors:
                continue
            # end if
            best_theme = 0
            best_score = 0.0
            for theme_idx, group in enumerate(anchor_groups, start=1):
                score_theme = max(abs(float(matrix_rel[i, a])) for a in group)
                if score_theme > best_score:
                    best_score = score_theme
                    best_theme = theme_idx
                # end if
            # end for theme_idx

            if best_score >= self.threshold_augment:
                cluster_assignments[i] = best_theme
                scores_related[i] = float(best_score)
            else:
                cluster_assignments[i] = 0
                scores_related[i] = None
            # end if
        # end for i

        for idx_var, id_clust in enumerate(cluster_assignments):
            dict_cluster_to_variables.setdefault(int(id_clust), []).append(idx_var)
        # end for idx_var

        # Augmented variables S_tilde: Union of clusters containing anchors (themes 1..K)
        indices_augmented_s_tilde = []
        for theme_idx in range(1, len(anchor_groups) + 1):
            indices_augmented_s_tilde.extend(dict_cluster_to_variables.get(theme_idx, []))
        # end for theme_idx
        indices_augmented_s_tilde = sorted(list(set(indices_augmented_s_tilde)))
        names_augmented_s_tilde = [names_variables[idx] for idx in indices_augmented_s_tilde]

        list_memberships = [
            ClusterMembership(
                id_variable=idx_var,
                name_variable=name_var,
                id_cluster=int(cluster_assignments[idx_var]),
                score_related=scores_related[idx_var],
            )
            for idx_var, name_var in enumerate(names_variables)
        ]

        return VariableClusteringResult(
            list_memberships=list_memberships,
            dict_cluster_to_variables=dict_cluster_to_variables,
            indices_augmented_s_tilde=indices_augmented_s_tilde,
            names_augmented_s_tilde=names_augmented_s_tilde,
            dict_anchor_to_cluster=dict_anchor_to_cluster,
        )
        # end def cluster_variables_relationship
# end class AnchorGuidedThematicClusterer


class VariableClusterer(AnchorGuidedThematicClusterer):
    """Default variable clusterer implementing anchor-guided thematic clustering with collinearity reduction.
    """
    pass
    # end class VariableClusterer
