import logging
import typing as ty
import numpy as np
from sklearn.covariance import GraphicalLassoCV, empirical_covariance
from ..common.models import CorrelationEdge, CorrelationResult

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes pairwise relationships and sparse graph representations across variables.
    """

    def __init__(self, threshold_edge_default: float = 0.3):
        """Initializes the analyzer.

        Args:
            threshold_edge_default: Default threshold for filtering edges in the graph.
        """
        self.threshold_edge_default = threshold_edge_default
        # end def __init__

    def compute_matrix_correlation(
        self,
        matrix_pooled: np.ndarray,
        names_variables: ty.List[str],
        method: str = "graphical_lasso",
        threshold_edge: ty.Optional[float] = None
    ) -> CorrelationResult:
        """Computes pairwise relationship matrix Sigma on pooled data Z = X union Y.

        Args:
            matrix_pooled: Pooled feature matrix (shape: N, d).
            names_variables: Feature names corresponding to columns.
            method: 'graphical_lasso' or 'pearson'.
            threshold_edge: Cutoff threshold for edge generation (defaults to instance setting).

        Returns:
            CorrelationResult with relationship matrix and edge list.
        """
        cutoff = threshold_edge if threshold_edge is not None else self.threshold_edge_default
        num_features = matrix_pooled.shape[1]

        matrix_rel = np.zeros((num_features, num_features), dtype=np.float64)

        if method == "graphical_lasso" and num_features > 1:
            try:
                # Graphical Lasso on covariance
                model_glasso = GraphicalLassoCV(max_iter=200)
                model_glasso.fit(matrix_pooled)
                # Precision matrix Theta
                matrix_precision = np.abs(model_glasso.precision_)
                # Normalize diagonal to 1.0 for comparability
                diag_sqrt = np.sqrt(np.diag(matrix_precision))
                diag_outer = np.outer(diag_sqrt, diag_sqrt)
                diag_outer[diag_outer == 0] = 1.0
                matrix_rel = matrix_precision / diag_outer
            except Exception as exc:
                logger.warning(
                    f"GraphicalLassoCV failed ({exc}). Falling back to empirical absolute correlation."
                )
                matrix_rel = np.abs(np.corrcoef(matrix_pooled, rowvar=False))
            # end try
        else:
            matrix_rel = np.abs(np.corrcoef(matrix_pooled, rowvar=False))
        # end if

        # Handle any NaN entries from constant columns
        matrix_rel = np.nan_to_num(matrix_rel, nan=0.0)
        np.fill_diagonal(matrix_rel, 1.0)

        list_edges = self.filter_edges_threshold(matrix_rel=matrix_rel, threshold=cutoff)

        return CorrelationResult(
            matrix_correlation=matrix_rel,
            list_edges=list_edges,
            names_variables=names_variables,
        )
        # end def compute_matrix_correlation

    def filter_edges_threshold(
        self,
        matrix_rel: np.ndarray,
        threshold: float
    ) -> ty.List[CorrelationEdge]:
        """Filters pairwise relationships exceeding the given threshold into an undirected edge list.

        Args:
            matrix_rel: Symmetric relationship matrix (d x d).
            threshold: Minimum score threshold.

        Returns:
            List of CorrelationEdge instances for pairs (i < j).
        """
        list_edges: ty.List[CorrelationEdge] = []
        num_features = matrix_rel.shape[0]

        for i in range(num_features):
            for j in range(i + 1, num_features):
                score = float(matrix_rel[i, j])
                if score >= threshold:
                    list_edges.append(
                        CorrelationEdge(
                            id_variable_1=i,
                            id_variable_2=j,
                            correlation_score=score,
                        )
                    )
                # end if
            # end for j
        # end for i

        return list_edges
        # end def filter_edges_threshold
# end class CorrelationAnalyzer
