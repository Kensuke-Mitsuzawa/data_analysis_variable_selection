import json
import typing as ty
import duckdb
import pandas as pd
from .schema import DDL_TABLES
from ..common.models import (
    TwoSampleDataContainer,
    VariableSelectionResult,
    CorrelationResult,
    VariableClusteringResult,
    PrototypeSampleResult,
)


class DuckDBStorageManager:
    """Manages the DuckDB analytical warehouse, schema creation, record persistence, and SQL querying.
    """

    def __init__(self, path_database: str = ":memory:"):
        """Initializes the database manager.

        Args:
            path_database: File path to DuckDB file or ':memory:' for in-memory DB.
        """
        self.path_database = path_database
        self.connection_db = duckdb.connect(database=self.path_database)
        # end def __init__

    def initialize_database_schema(self) -> None:
        """Executes table creation DDL for all analytical output tables.
        """
        for name_table, ddl_query in DDL_TABLES.items():
            self.connection_db.execute(ddl_query)
        # end for name_table
        # end def initialize_database_schema

    def insert_records_selection(self, result: VariableSelectionResult) -> None:
        """Persists MMD variable selection results into analysis_variable_selection.

        Args:
            result: VariableSelectionResult holding selected indices, names, and weights.
        """
        self.connection_db.execute("DELETE FROM analysis_variable_selection")
        for idx_var, name_var, weight in zip(
            result.indices_selected,
            result.names_selected,
            result.weights_selected
        ):
            self.connection_db.execute(
                """
                INSERT INTO analysis_variable_selection (id_variable, name_variable, weight)
                VALUES (?, ?, ?)
                """,
                [int(idx_var), str(name_var), float(weight)]
            )
        # end for
        # end def insert_records_selection

    def insert_records_correlation(self, result: CorrelationResult) -> None:
        """Persists pairwise correlation edge list into analysis_variable_correlation.

        Args:
            result: CorrelationResult holding edges above threshold.
        """
        self.connection_db.execute("DELETE FROM analysis_variable_correlation")
        for edge in result.list_edges:
            self.connection_db.execute(
                """
                INSERT INTO analysis_variable_correlation (id_variable_1, id_variable_2, correlation_score)
                VALUES (?, ?, ?)
                """,
                [int(edge.id_variable_1), int(edge.id_variable_2), float(edge.correlation_score)]
            )
        # end for edge
        # end def insert_records_correlation

    def insert_records_clustering(self, result: VariableClusteringResult) -> None:
        """Persists variable clustering and relatedness scores into analysis_variable_clustering.

        Args:
            result: VariableClusteringResult holding memberships.
        """
        self.connection_db.execute("DELETE FROM analysis_variable_clustering")
        for membership in result.list_memberships:
            self.connection_db.execute(
                """
                INSERT INTO analysis_variable_clustering (id_variable, name_variable, id_cluster, score_related)
                VALUES (?, ?, ?, ?)
                """,
                [
                    int(membership.id_variable),
                    str(membership.name_variable),
                    int(membership.id_cluster),
                    membership.score_related if membership.score_related is not None else None,
                ]
            )
        # end for membership
        # end def insert_records_clustering

    def insert_records_prototypes(self, result: PrototypeSampleResult) -> None:
        """Persists prototypical exemplar samples into analysis_representative_samples.

        Args:
            result: PrototypeSampleResult holding exemplar records.
        """
        self.connection_db.execute("DELETE FROM analysis_representative_samples")
        for prototype in result.list_prototype_records:
            features_json_str = json.dumps(prototype.dict_feature_values)
            self.connection_db.execute(
                """
                INSERT INTO analysis_representative_samples
                (id_sample, label_class, is_prototype_for, distance_score, type_subspace, features_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    int(prototype.id_sample),
                    str(prototype.label_class),
                    str(prototype.is_prototype_for),
                    float(prototype.distance_score),
                    str(prototype.type_subspace),
                    features_json_str,
                ]
            )
        # end for prototype
        # end def insert_records_prototypes

    def insert_preprocessed_features(self, container: TwoSampleDataContainer) -> None:
        """Persists preprocessed feature records as a human-readable table and saves dataset metadata.

        Args:
            container: TwoSampleDataContainer holding sample matrices X and Y, and feature names.
        """
        import numpy as np

        n_x = container.sample_matrix_x.shape[0]
        n_y = container.sample_matrix_y.shape[0]

        ids_x = [f"X_{i:06d}" for i in range(n_x)]
        ids_y = [f"Y_{j:06d}" for j in range(n_y)]
        all_ids = ids_x + ids_y
        all_labels = ["X"] * n_x + ["Y"] * n_y

        matrix_combined = np.vstack([container.sample_matrix_x, container.sample_matrix_y])
        df_combined = pd.DataFrame(matrix_combined, columns=container.name_features)
        df_combined.insert(0, "distribution_label", all_labels)
        df_combined.insert(0, "sample_id", all_ids)

        self.connection_db.register("df_features_temp", df_combined)
        self.connection_db.execute("CREATE OR REPLACE TABLE dataset_features_preprocessed AS SELECT * FROM df_features_temp")
        self.connection_db.unregister("df_features_temp")

        # Save dataset metadata
        self.connection_db.execute("CREATE OR REPLACE TABLE dataset_metadata (key VARCHAR PRIMARY KEY, value VARCHAR)")
        meta_items = [
            ("dataset_name", str(container.metadata_dataset.get("dataset_name", "Unknown"))),
            ("n_samples_x", str(n_x)),
            ("n_samples_y", str(n_y)),
            ("num_features", str(len(container.name_features))),
            ("feature_names_json", json.dumps(container.name_features)),
            ("metadata_json", json.dumps(container.metadata_dataset)),
        ]
        for k, v in meta_items:
            self.connection_db.execute(
                "INSERT INTO dataset_metadata (key, value) VALUES (?, ?)",
                [k, v]
            )
        # end for k, v
        # end def insert_preprocessed_features

    def fetch_preprocessed_features(self) -> TwoSampleDataContainer:
        """Loads TwoSampleDataContainer back from the dataset_features_preprocessed table.

        Returns:
            TwoSampleDataContainer holding sample matrices X and Y, feature names, and metadata.
        """
        import numpy as np

        df_all = self.connection_db.execute("SELECT * FROM dataset_features_preprocessed").df()
        feature_cols = [c for c in df_all.columns if c not in ("sample_id", "distribution_label")]

        df_x = df_all[df_all["distribution_label"] == "X"][feature_cols]
        df_y = df_all[df_all["distribution_label"] == "Y"][feature_cols]

        matrix_x = df_x.to_numpy(dtype=np.float64)
        matrix_y = df_y.to_numpy(dtype=np.float64)

        metadata: ty.Dict[str, ty.Any] = {}
        try:
            df_meta = self.connection_db.execute("SELECT key, value FROM dataset_metadata").df()
            dict_raw = dict(zip(df_meta["key"], df_meta["value"]))
            if "metadata_json" in dict_raw:
                metadata = json.loads(dict_raw["metadata_json"])
            # end if
        except Exception:
            pass
        # end try

        return TwoSampleDataContainer(
            sample_matrix_x=matrix_x,
            sample_matrix_y=matrix_y,
            name_features=feature_cols,
            metadata_dataset=metadata,
        )
        # end def fetch_preprocessed_features

    def fetch_records_sql(self, query_sql: str) -> pd.DataFrame:
        """Executes a SQL query against the analytical DuckDB database and returns a Pandas DataFrame.

        Args:
            query_sql: SQL select query string.

        Returns:
            Pandas DataFrame containing query results.
        """
        df_result = self.connection_db.execute(query_sql).df()
        return df_result
        # end def fetch_records_sql

    def close_connection_database(self) -> None:
        """Closes the DuckDB connection.
        """
        if self.connection_db is not None:
            self.connection_db.close()
        # end if
        # end def close_connection_database
# end class DuckDBStorageManager
