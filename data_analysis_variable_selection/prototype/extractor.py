import typing as ty
import numpy as np
from ..common.models import PrototypeSampleRecord, PrototypeSampleResult


class PrototypeExtractor:
    """Extracts prototypical exemplar records from distributions X and Y within specified subspaces.
    """

    def __init__(self, metric_mode: str = "discriminative", epsilon_cov: float = 1e-6):
        """Initializes the prototype extractor.

        Args:
            metric_mode: 'discriminative' (high discrepancy vs opposite class) or 'centroid' (closest to own centroid).
            epsilon_cov: Regularization added to covariance diagonal for stable inversion.
        """
        self.metric_mode = metric_mode
        self.epsilon_cov = epsilon_cov
        # end def __init__

    def extract_samples_prototype(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        names_features: ty.List[str],
        list_subspace_indices: ty.List[int],
        type_subspace: str,
        top_n: int = 5
    ) -> PrototypeSampleResult:
        """Extracts top-N prototypical records for distributions X and Y within the given subspace.

        Args:
            sample_x: Feature matrix for distribution X (n_X, d).
            sample_y: Feature matrix for distribution Y (n_Y, d).
            names_features: All feature names.
            list_subspace_indices: Selected feature indices defining the subspace (e.g. hat_S or S_tilde).
            type_subspace: Label for subspace ('hat_S' or 'hat_S_augmented').
            top_n: Number of exemplar prototypes to select for each class.

        Returns:
            PrototypeSampleResult containing list of PrototypeSampleRecord objects.
        """
        if not list_subspace_indices:
            list_subspace_indices = list(range(sample_x.shape[1]))
        # end if

        # Project into subspace
        x_sub = sample_x[:, list_subspace_indices]
        y_sub = sample_y[:, list_subspace_indices]

        # Calculate distances/scores
        scores_x = self._compute_prototypicality_scores(
            target_samples=x_sub,
            own_distribution=x_sub,
            opposite_distribution=y_sub
        )
        scores_y = self._compute_prototypicality_scores(
            target_samples=y_sub,
            own_distribution=y_sub,
            opposite_distribution=x_sub
        )

        n_pick_x = min(top_n, len(sample_x))
        n_pick_y = min(top_n, len(sample_y))

        # Higher score = more prototypical/exemplary
        top_indices_x = np.argsort(-scores_x)[:n_pick_x]
        top_indices_y = np.argsort(-scores_y)[:n_pick_y]

        list_records: ty.List[PrototypeSampleRecord] = []

        # Prototype records for X
        for idx in top_indices_x:
            feat_dict = {
                names_features[f_idx]: float(sample_x[idx, f_idx])
                for f_idx in list_subspace_indices
            }
            list_records.append(
                PrototypeSampleRecord(
                    id_sample=int(idx),
                    label_class="X",
                    is_prototype_for="X",
                    distance_score=float(scores_x[idx]),
                    type_subspace=type_subspace,
                    dict_feature_values=feat_dict,
                )
            )
        # end for idx

        # Prototype records for Y
        for idx in top_indices_y:
            feat_dict = {
                names_features[f_idx]: float(sample_y[idx, f_idx])
                for f_idx in list_subspace_indices
            }
            list_records.append(
                PrototypeSampleRecord(
                    id_sample=int(idx),
                    label_class="Y",
                    is_prototype_for="Y",
                    distance_score=float(scores_y[idx]),
                    type_subspace=type_subspace,
                    dict_feature_values=feat_dict,
                )
            )
        # end for idx

        return PrototypeSampleResult(list_prototype_records=list_records)
        # end def extract_samples_prototype

    def compute_distance_mahalanobis(
        self,
        samples: np.ndarray,
        reference_distribution: np.ndarray
    ) -> np.ndarray:
        """Computes Mahalanobis distance of samples relative to a reference distribution's centroid.

        Args:
            samples: Query samples (N, k).
            reference_distribution: Reference samples (M, k).

        Returns:
            1D array of Mahalanobis distances (length N).
        """
        centroid = np.mean(reference_distribution, axis=0)
        cov = np.cov(reference_distribution, rowvar=False)

        if cov.ndim == 0 or reference_distribution.shape[1] == 1:
            variance = float(cov) if cov.ndim == 0 else cov[0, 0]
            var_inv = 1.0 / max(variance + self.epsilon_cov, self.epsilon_cov)
            diff = samples - centroid
            dists = np.sqrt(diff[:, 0] ** 2 * var_inv)
            return dists
        # end if

        cov_reg = cov + np.eye(cov.shape[0]) * self.epsilon_cov
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except Exception:
            cov_inv = np.eye(cov.shape[0])
        # end try

        diff = samples - centroid
        left = np.dot(diff, cov_inv)
        dist_sq = np.sum(left * diff, axis=1)
        dists = np.sqrt(np.maximum(dist_sq, 0.0))
        return dists
        # end def compute_distance_mahalanobis

    def _compute_prototypicality_scores(
        self,
        target_samples: np.ndarray,
        own_distribution: np.ndarray,
        opposite_distribution: np.ndarray
    ) -> np.ndarray:
        """Computes prototype ranking scores based on configured metric mode.
        """
        dist_own = self.compute_distance_mahalanobis(target_samples, own_distribution)
        dist_opp = self.compute_distance_mahalanobis(target_samples, opposite_distribution)

        if self.metric_mode == "discriminative":
            # High distance from opponent and low distance from own centroid
            scores = dist_opp - dist_own
        else:
            # Closest to own centroid
            scores = -dist_own
        # end if
        return scores
        # end def _compute_prototypicality_scores
# end class PrototypeExtractor
