import datetime
import logging
import os
import subprocess
import typing as ty
from pathlib import Path
from pydantic import BaseModel, Field

from ..cli.cli_config import PipelineCliConfig
from ..common.models import TwoSampleDataContainer
from ..common.base_preprocessor import BaseDatasetPreprocessor

logger = logging.getLogger(__name__)


class FeatureItemData(BaseModel):
    """Specification of a processed feature, its source original feature(s), and its data type.
    """
    feature_processed: str = Field(description="Name of the processed feature (or 'removed').")
    feature_original: str = Field(description="Original source feature name or list expression if multiple combined.")
    type_feature: str = Field(description="Data type of the processed feature (e.g. str, int, category, float, removed).")
# end class FeatureItemData


class FeatureOperationRecorder:
    """Records feature transformations, one-hot dummy expansions, and removed raw columns.
    """

    def __init__(self) -> None:
        self._items: ty.List[FeatureItemData] = []
        self._processed_names: ty.Set[str] = set()
    # end def __init__

    def clear(self) -> None:
        """Resets the recorded feature items."""
        self._items.clear()
        self._processed_names.clear()
    # end def clear

    def record_feature(
        self,
        name_processed: str,
        source_original: ty.Union[str, ty.List[str]],
        type_feature: str
    ) -> None:
        """Records a processed feature and its source original column(s).

        Args:
            name_processed: Processed feature column name.
            source_original: Single raw column or list of raw columns.
            type_feature: Data type of the processed feature (e.g. float, int, category).
        """
        if isinstance(source_original, list):
            if len(source_original) > 1:
                orig_str = f"[{', '.join(repr(c) for c in source_original)}]"
            elif len(source_original) == 1:
                orig_str = source_original[0]
            else:
                orig_str = "None"
            # end if
        else:
            orig_str = str(source_original)
        # end if

        self._items.append(
            FeatureItemData(
                feature_processed=name_processed,
                feature_original=orig_str,
                type_feature=type_feature
            )
        )
        self._processed_names.add(name_processed)
    # end def record_feature

    def record_onehot_expansion(
        self,
        source_column: str,
        generated_columns: ty.List[str]
    ) -> None:
        """Records dummy indicator columns created from a single categorical feature.

        Args:
            source_column: Original categorical column name.
            generated_columns: List of binary indicator column names generated.
        """
        for col in generated_columns:
            self.record_feature(
                name_processed=col,
                source_original=source_column,
                type_feature="category"
            )
        # end for col
    # end def record_onehot_expansion

    def record_removed(
        self,
        source_original: str
    ) -> None:
        """Records a raw column that was dropped or removed during preprocessing.

        Args:
            source_original: Original column name that was dropped/removed.
        """
        self._items.append(
            FeatureItemData(
                feature_processed="removed",
                feature_original=source_original,
                type_feature="removed"
            )
        )
    # end def record_removed

    def get_feature_items(self, include_removed: bool = True) -> ty.List[FeatureItemData]:
        """Returns the list of recorded feature items.

        Args:
            include_removed: Whether to include removed raw features.

        Returns:
            List of FeatureItemData specifications.
        """
        if include_removed:
            return list(self._items)
        # end if
        return [item for item in self._items if item.type_feature != "removed"]
    # end def get_feature_items
# end class FeatureOperationRecorder


class FeatureOperationTracker:
    """Tracks feature transformations and exports feature lineage markdown specifications.
    """

    def track_features_dataset(
        self,
        preprocessor: BaseDatasetPreprocessor,
        config: PipelineCliConfig,
        container: ty.Optional[TwoSampleDataContainer] = None,
        include_removed: bool = True
    ) -> ty.List[FeatureItemData]:
        """Tracks all feature operations for the dataset and returns their specifications.

        Args:
            preprocessor: Dataset preprocessor instance.
            config: Pipeline configuration.
            container: Optional preprocessed TwoSampleDataContainer.
            include_removed: Whether to include raw features removed during preprocessing.

        Returns:
            List of FeatureItemData specifications.
        """
        if hasattr(preprocessor, "track_feature_operations"):
            list_features = preprocessor.track_feature_operations(
                container=container,
                include_removed=include_removed
            )
        else:
            dataset_name = config.project.dataset_name.lower().strip()
            if dataset_name == "ames_housing":
                list_features = self._extract_features_ames_housing(config, include_removed=include_removed)
            elif dataset_name == "speed_dating":
                list_features = self._extract_features_speed_dating(config, include_removed=include_removed)
            else:
                raise ValueError(f"Unsupported dataset '{dataset_name}' for feature tracking.")
            # end if
        # end if

        return list_features
        # end def track_features_dataset

    def export_feature_list_markdown(
        self,
        list_features: ty.List[FeatureItemData],
        config: PipelineCliConfig,
        path_output_markdown: ty.Optional[str] = None
    ) -> str:
        """Exports markdown document with 3-column table of processed features to output directory.

        Args:
            list_features: List of FeatureItemData specifications.
            config: PipelineCliConfig instance.
            path_output_markdown: Optional custom target path for the markdown file.

        Returns:
            Path to the written markdown file.
        """
        dataset_name = config.project.dataset_name.lower().strip()
        if dataset_name == "ames_housing":
            name_display = "Ames Housing Dataset"
            url_source = config.dataset.ames_housing.url_download or "https://www.openml.org/d/42165"
        elif dataset_name == "speed_dating":
            name_display = "Columbia Speed Dating Experiment"
            url_source = config.dataset.speed_dating.url_download or "https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment"
        else:
            name_display = dataset_name
            url_source = "Unknown"
        # end if

        md_content = self._format_markdown_document(
            name_dataset=name_display,
            url_source=url_source,
            list_features=list_features
        )

        out_dir = Path(config.project.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_file = Path(path_output_markdown) if path_output_markdown else (out_dir / "features.md")
        target_file.parent.mkdir(parents=True, exist_ok=True)

        with open(target_file, "w", encoding="utf-8") as f_out:
            f_out.write(md_content)
        # end with

        # Also write a copy to feature_list.md if target was features.md
        if not path_output_markdown or Path(path_output_markdown).name == "features.md":
            alt_file = out_dir / "feature_list.md"
            with open(alt_file, "w", encoding="utf-8") as f_alt:
                f_alt.write(md_content)
            # end with
        # end if

        logger.info(f"Exported feature list markdown ({len(list_features)} features) to: {target_file}")
        return str(target_file.resolve())
        # end def export_feature_list_markdown

    def _extract_features_speed_dating(
        self,
        config: PipelineCliConfig,
        include_removed: bool = True
    ) -> ty.List[FeatureItemData]:
        """Extracts lineage and data type for Speed Dating processed features via preprocessor."""
        from .speed_dating.config import SpeedDatingPreprocessingConfig
        from .speed_dating.preprocessor import SpeedDatingPreprocessor

        sd_cfg = SpeedDatingPreprocessingConfig(
            path_data_file=config.dataset.speed_dating.raw_data_path
        )
        preprocessor = SpeedDatingPreprocessor(config=sd_cfg)
        return preprocessor.track_feature_operations(include_removed=include_removed)
        # end def _extract_features_speed_dating

    def _extract_features_ames_housing(
        self,
        config: PipelineCliConfig,
        include_removed: bool = True
    ) -> ty.List[FeatureItemData]:
        """Extracts lineage and data type for Ames Housing processed features via preprocessor."""
        from .ames_housing.config import AmesPreprocessingConfig
        from .ames_housing.preprocessor import AmesHousingPreprocessor

        ames_cfg = AmesPreprocessingConfig(
            path_data_file=config.dataset.ames_housing.raw_data_path
        )
        preprocessor = AmesHousingPreprocessor(config=ames_cfg)
        return preprocessor.track_feature_operations(include_removed=include_removed)
        # end def _extract_features_ames_housing

    def _format_markdown_document(
        self,
        name_dataset: str,
        url_source: str,
        list_features: ty.List[FeatureItemData]
    ) -> str:
        """Formats the feature specifications as a markdown document containing a 3-column table.

        Args:
            name_dataset: Descriptive human-readable dataset name.
            url_source: Source URL for the dataset.
            list_features: List of FeatureItemData specifications.

        Returns:
            Formatted markdown content.
        """
        try:
            commit_id = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            commit_id = "Unknown"
        # end try

        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        active_features = [f for f in list_features if f.type_feature != "removed"]
        removed_features = [f for f in list_features if f.type_feature == "removed"]

        # Sort active features by processed feature name, removed features by original feature name
        active_features.sort(key=lambda item: item.feature_processed)
        removed_features.sort(key=lambda item: item.feature_original)

        lines: ty.List[str] = [
            f"# Feature Specification: {name_dataset}",
            "",
            "| Metadata | Value |",
            "| :--- | :--- |",
            f"| Generation Date | {timestamp_utc} |",
            f"| Git Commit ID | `{commit_id}` |",
            f"| Source URL | {url_source} |",
            f"| Active Processed Features | {len(active_features)} |",
            f"| Removed Features | {len(removed_features)} |",
            f"| Total Features Tracked | {len(list_features)} |",
            "",
            "## Processed Features",
            "",
            "| processed feature | original feature | type of the processed feature |",
            "| :--- | :--- | :--- |",
        ]

        for feat in active_features:
            lines.append(f"| {feat.feature_processed} | {feat.feature_original} | {feat.type_feature} |")
        # end for feat

        for feat in removed_features:
            lines.append(f"| {feat.feature_processed} | {feat.feature_original} | {feat.type_feature} |")
        # end for feat

        lines.append("")
        return "\n".join(lines)
        # end def _format_markdown_document
# end class FeatureOperationTracker
