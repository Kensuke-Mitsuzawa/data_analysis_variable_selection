"""DDL schema definitions for internal analytical DuckDB warehouse.
"""
import typing as ty

DDL_TABLES: ty.Dict[str, str] = {
    "analysis_variable_selection": """
        CREATE TABLE IF NOT EXISTS analysis_variable_selection (
            id_variable INTEGER PRIMARY KEY,
            name_variable VARCHAR NOT NULL,
            weight DOUBLE NOT NULL
        );
    """,
    "analysis_variable_correlation": """
        CREATE TABLE IF NOT EXISTS analysis_variable_correlation (
            id_variable_1 INTEGER NOT NULL,
            id_variable_2 INTEGER NOT NULL,
            correlation_score DOUBLE NOT NULL
        );
    """,
    "analysis_variable_clustering": """
        CREATE TABLE IF NOT EXISTS analysis_variable_clustering (
            id_variable INTEGER NOT NULL,
            name_variable VARCHAR NOT NULL,
            id_cluster INTEGER NOT NULL,
            score_related DOUBLE
        );
    """,
    "analysis_representative_samples": """
        CREATE TABLE IF NOT EXISTS analysis_representative_samples (
            id_sample INTEGER NOT NULL,
            label_class VARCHAR NOT NULL,
            is_prototype_for VARCHAR NOT NULL,
            distance_score DOUBLE NOT NULL,
            type_subspace VARCHAR NOT NULL,
            features_json VARCHAR
        );
    """,
}
