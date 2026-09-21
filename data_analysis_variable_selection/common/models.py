import typing as ty
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class TwoSampleDataContainer(BaseModel):
    """Container holding two-sample dataset matrices and associated feature metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_matrix_x: np.ndarray = Field(description="Sample matrix for distribution X (shape: n_X, d)")
    sample_matrix_y: np.ndarray = Field(description="Sample matrix for distribution Y (shape: n_Y, d)")
    name_features: ty.List[str] = Field(description="List of feature names matching the column order")
    metadata_dataset: ty.Dict[str, ty.Any] = Field(default_factory=dict, description="Arbitrary dataset metadata")

    def save_to_npz(self, filepath: str) -> None:
        """Saves matrices, feature names, and metadata into a compressed .npz archive."""
        import json
        np.savez_compressed(
            filepath,
            sample_matrix_x=self.sample_matrix_x,
            sample_matrix_y=self.sample_matrix_y,
            name_features=np.array(self.name_features, dtype=object),
            metadata_dataset=json.dumps(self.metadata_dataset),
        )
        # end def save_to_npz

    @classmethod
    def load_from_npz(cls, filepath: str) -> "TwoSampleDataContainer":
        """Reconstructs TwoSampleDataContainer from a .npz archive."""
        import json
        data = np.load(filepath, allow_pickle=True)
        meta_str = str(data["metadata_dataset"])
        try:
            metadata = json.loads(meta_str)
        except Exception:
            metadata = {}
        # end try
        return cls(
            sample_matrix_x=data["sample_matrix_x"],
            sample_matrix_y=data["sample_matrix_y"],
            name_features=list(data["name_features"]),
            metadata_dataset=metadata,
        )
        # end def load_from_npz


class ScaledDataContainer(BaseModel):
    """Container holding standardized two-sample matrices and scaling parameters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_matrix_x_scaled: np.ndarray = Field(description="Z-score standardized matrix for distribution X")
    sample_matrix_y_scaled: np.ndarray = Field(description="Z-score standardized matrix for distribution Y")
    vector_mean: np.ndarray = Field(description="Feature-wise mean computed from pooled dataset Z = X union Y")
    vector_std: np.ndarray = Field(description="Feature-wise standard deviation from pooled dataset Z")
    name_features: ty.List[str] = Field(description="List of feature names")


class VariableSelectionResult(BaseModel):
    """Result of the MMD variable selection algorithm.
    """
    indices_selected: ty.List[int] = Field(description="Indices of selected anchor variables (hat_S)")
    names_selected: ty.List[str] = Field(description="Names of selected anchor variables")
    weights_selected: ty.List[float] = Field(description="Weights assigned to selected anchor variables")
    p_value: ty.Optional[float] = Field(default=None, description="Hypothesis test p-value if evaluated")
    metadata_selection: ty.Dict[str, ty.Any] = Field(default_factory=dict, description="Additional optimization metadata")


class CorrelationEdge(BaseModel):
    """Representation of an edge between two variables in the correlation graph.
    """
    id_variable_1: int = Field(description="Zero-based index of variable 1")
    id_variable_2: int = Field(description="Zero-based index of variable 2")
    correlation_score: float = Field(description="Magnitude or score of relationship between variable 1 and 2")


class CorrelationResult(BaseModel):
    """Result of pairwise variable relationship analysis.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    matrix_correlation: np.ndarray = Field(description="Pairwise correlation or precision matrix Sigma (d x d)")
    list_edges: ty.List[CorrelationEdge] = Field(description="List of significant edges above threshold")
    names_variables: ty.List[str] = Field(description="Variable names corresponding to matrix indices")


class ClusterMembership(BaseModel):
    """Cluster assignment and relatedness score for a single variable.
    """
    id_variable: int = Field(description="Zero-based index of the variable")
    name_variable: str = Field(description="Name of the variable")
    id_cluster: int = Field(description="Cluster ID to which the variable belongs")
    score_related: ty.Optional[float] = Field(
        default=None,
        description="Max absolute correlation with an anchor in the cluster, None if no anchor in cluster"
    )


class VariableClusteringResult(BaseModel):
    """Result of clustering variables and identifying augmented variable sets.
    """
    list_memberships: ty.List[ClusterMembership] = Field(description="List of cluster memberships for all variables")
    dict_cluster_to_variables: ty.Dict[int, ty.List[int]] = Field(description="Mapping of cluster ID to variable indices")
    indices_augmented_s_tilde: ty.List[int] = Field(description="Indices of the globally augmented feature set S_tilde")
    names_augmented_s_tilde: ty.List[str] = Field(description="Names of the globally augmented feature set S_tilde")
    dict_anchor_to_cluster: ty.Dict[int, int] = Field(description="Mapping of anchor variable index to its cluster ID")


class PrototypeSampleRecord(BaseModel):
    """Record describing a single extracted prototypical exemplar sample.
    """
    id_sample: int = Field(description="Sample identifier / row index in distribution")
    label_class: str = Field(description="Distribution label: 'X' or 'Y'")
    is_prototype_for: str = Field(description="Class this prototype represents ('X' or 'Y')")
    distance_score: float = Field(description="Distance or discrepancy ranking metric")
    type_subspace: str = Field(description="Subspace used for prototype selection: 'hat_S' or 'hat_S_augmented'")
    dict_feature_values: ty.Dict[str, float] = Field(description="Feature values for the prototype sample")


class PrototypeSampleResult(BaseModel):
    """Container holding all extracted prototype records.
    """
    list_prototype_records: ty.List[PrototypeSampleRecord] = Field(description="List of extracted prototype records")
