import logging
import shutil
import tarfile
import urllib.request
import zipfile
import typing as ty
from pathlib import Path

from ..cli.cli_config import PipelineCliConfig
from .ames_housing.loader import AmesHousingDataLoader

logger = logging.getLogger(__name__)


class DatasetSetupHandler:
    """Handles downloading, uncompressing, and verifying datasets for the analysis pipeline.
    """

    def setup_dataset(self, config: PipelineCliConfig) -> str:
        """Executes setup actions for the target dataset specified in the config.

        Args:
            config: PipelineCliConfig instance.

        Returns:
            Path to verified raw dataset file.
        """
        dataset_name = config.project.dataset_name.lower().strip()

        if dataset_name == "ames_housing":
            return self._setup_ames_housing(config)
        elif dataset_name == "speed_dating":
            return self._setup_speed_dating(config)
        else:
            raise ValueError(f"Unsupported dataset name '{dataset_name}' in configuration.")
        # end if
        # end def setup_dataset

    def _setup_ames_housing(self, config: PipelineCliConfig) -> str:
        """Downloads or validates Ames Housing dataset file using pathlib.Path."""
        ames_cfg = config.dataset.ames_housing
        raw_path = ames_cfg.raw_data_path

        if raw_path:
            path_raw = Path(raw_path)
            if path_raw.exists():
                logger.info(f"Ames Housing dataset found at existing path: {path_raw}")
                return str(path_raw.resolve())
            # end if
        # end if

        # Determine target download path
        target_dir = Path(config.project.output_directory) / "raw_data"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_csv = Path(raw_path) if raw_path else (target_dir / "ames_housing_raw.csv")

        if target_csv.exists():
            logger.info(f"Target raw file already exists at: {target_csv}")
            return str(target_csv.resolve())
        # end if

        # Attempt download if URL specified
        if ames_cfg.url_download:
            try:
                logger.info(f"Downloading Ames Housing dataset from: {ames_cfg.url_download}")
                temp_download = target_dir / "download_temp"
                urllib.request.urlretrieve(ames_cfg.url_download, str(temp_download))

                # Check if archive
                if zipfile.is_zipfile(temp_download):
                    with zipfile.ZipFile(temp_download, "r") as zip_ref:
                        zip_ref.extractall(target_dir)
                    # end with
                    temp_download.unlink(missing_ok=True)
                elif tarfile.is_tarfile(temp_download):
                    with tarfile.open(temp_download, "r:*") as tar_ref:
                        tar_ref.extractall(target_dir)
                    # end with
                    temp_download.unlink(missing_ok=True)
                else:
                    target_csv.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp_download), str(target_csv))
                # end if
            except Exception as exc:
                logger.warning(f"Download from URL failed ({exc}). Falling back to OpenML / internal loader.")
            # end try
        # end if

        if not target_csv.exists():
            logger.info("Using AmesHousingDataLoader to fetch and cache raw data...")
            loader = AmesHousingDataLoader()
            df = loader.load_data_raw()
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target_csv, index=False)
        # end if

        logger.info(f"Dataset successfully setup at: {target_csv}")
        return str(target_csv.resolve())
        # end def _setup_ames_housing

    def _setup_speed_dating(self, config: PipelineCliConfig) -> str:
        """Sets up Speed Dating dataset using pathlib.Path."""
        sd_cfg = config.dataset.speed_dating
        raw_path = sd_cfg.raw_data_path

        if raw_path:
            path_raw = Path(raw_path)
            if path_raw.exists():
                logger.info(f"Speed Dating dataset found at: {path_raw}")
                return str(path_raw.resolve())
            # end if
        # end if

        target_dir = Path(config.project.output_directory) / "raw_data"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_csv = Path(raw_path) if raw_path else (target_dir / "speed_dating_raw.csv")

        if target_csv.exists():
            return str(target_csv.resolve())
        # end if

        if sd_cfg.url_download:
            logger.info(f"Downloading Speed Dating dataset from: {sd_cfg.url_download}")
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(sd_cfg.url_download, str(target_csv))
        else:
            logger.info(f"Raw speed dating file not found at {target_csv}. Please provide dataset path in TOML.")
        # end if

        return str(target_csv.resolve())
        # end def _setup_speed_dating
# end class DatasetSetupHandler
