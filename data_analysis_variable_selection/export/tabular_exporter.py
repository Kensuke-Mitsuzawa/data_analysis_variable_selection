import json
import os
import typing as ty
import pandas as pd
from ..database.manager import DuckDBStorageManager


class TabularArtifactExporter:
    """Exports normalized database records to standard CSV deliverables for analysts.
    """

    def export_cluster_analysis_csv(
        self,
        db_manager: DuckDBStorageManager,
        path_output_csv: str
    ) -> str:
        """Exports cluster_analysis.csv containing id_variable, name_variable, id_cluster, score_related.

        Args:
            db_manager: DuckDBStorageManager connected to populated database.
            path_output_csv: Target path for the CSV deliverable.

        Returns:
            Absolute path to the created CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_csv)), exist_ok=True)
        query = """
            SELECT id_variable, name_variable, id_cluster, score_related
            FROM analysis_variable_clustering
            ORDER BY id_cluster ASC, score_related DESC NULLS LAST
        """
        df_clusters = db_manager.fetch_records_sql(query)
        df_clusters.to_csv(path_output_csv, index=False)
        return os.path.abspath(path_output_csv)
        # end def export_cluster_analysis_csv

    def export_representative_samples_csv(
        self,
        db_manager: DuckDBStorageManager,
        path_output_csv: str
    ) -> str:
        """Exports representative_samples.csv joining exemplar metadata with original raw features.

        Args:
            db_manager: DuckDBStorageManager connected to populated database.
            path_output_csv: Target path for the CSV deliverable.

        Returns:
            Absolute path to the created CSV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_output_csv)), exist_ok=True)
        query = """
            SELECT id_sample, label_class, is_prototype_for, distance_score, type_subspace, features_json
            FROM analysis_representative_samples
            ORDER BY type_subspace, label_class, distance_score DESC
        """
        df_samples = db_manager.fetch_records_sql(query)

        # Unpack JSON features into tabular columns
        list_rows: ty.List[ty.Dict[str, ty.Any]] = []
        for _, row in df_samples.iterrows():
            dict_row = {
                "id_sample": row["id_sample"],
                "label_class": row["label_class"],
                "is_prototype_for": row["is_prototype_for"],
                "distance_score": row["distance_score"],
                "type_subspace": row["type_subspace"],
            }
            if pd.notna(row["features_json"]):
                try:
                    feat_dict = json.loads(row["features_json"])
                    dict_row.update(feat_dict)
                except Exception:
                    pass
                # end try
            # end if
            list_rows.append(dict_row)
        # end for

        df_expanded = pd.DataFrame(list_rows)
        df_expanded.to_csv(path_output_csv, index=False)
        return os.path.abspath(path_output_csv)
        # end def export_representative_samples_csv
# end class TabularArtifactExporter
