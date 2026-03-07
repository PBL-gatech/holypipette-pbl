"""Build pipette coordinate records from saved rig-recorder frames."""

from __future__ import annotations


import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patcherbot.deepLearning.pipetteDetector import PipetteDetector1, PipetteDetector2
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser


from experiments.DatasetBuilder2 import DatasetBuilder2, _read_csv_with_fallback


@dataclass
class FrameRecord:
    """Represents a single recorded fram with its timestamp and pipette position"""
    timestamp: float
    pi_x: float
    pi_y: float
    pi_z: float

class _DatasetFilterHelper(DatasetBuilder2):
    """Lightweight DatasetBuilder2 adapter to reuse demo filtering utilities."""

    def __init__(self, rig_data_root: Path) -> None:
        """
        Initialize the preparer with the rig data directory.

        Args:
            rig_data_root (Path): Path to directory containing raw rig data.
        """
        self._rig_data_root = Path(rig_data_root)
        self._data_root = self._rig_data_root.parent
        self._log_data_root = self._data_root / "log_data"
        super().__init__(
            dataset_name="ImageDatasetPreparer_filter.hdf5",
            val_ratio=0.0,
            omit_stage_movement=False,
            random_seed=0,
        )

    def _write_metadata_files(self) -> None:  # pragma: no cover - metadata not needed
        """Skip DatasetBuilder2 metadata emission for filtering adapter."""
        self._cached_metadata = self._collect_metadata()

    def load_graph_values(self, folder: str) -> Optional[pd.DataFrame]:
        """Load graph_recording.csv from a folder if it exists.
        
        Args:
            folder (str): Subdirectory under the rig data root.

        Returns:
            DataFrame of graph values if the file exists and loads successfully, otherwise None.
        """
        graph_path = self._rig_data_root / folder / "graph_recording.csv"
        if not graph_path.exists():
            return None
        try:
            return pd.read_csv(graph_path, sep=";")
        except Exception:
            return None

    def load_log_values(self, folder: str) -> Optional[pd.DataFrame]:
        """Load log values for a recording based on its data.
        
        Args:
            folder (str): Recording foler name containing a date prefix.

        Returns:
            DataFrame of log values if the corresponding log file loads successfully,
                otherwise None.
        """
        day_token = folder[:10]
        log_path = self._log_data_root / f"logs_{day_token}.csv"
        if not log_path.exists():
            return None
        try:
            return _read_csv_with_fallback(log_path, on_bad_lines="skip")
        except Exception:
            return None

    def compute_attempt_windows(
        self,
        folder: str,
        log_values: pd.DataFrame,
        graph_values: pd.DataFrame,
    ) -> List[Tuple[float, float]]:
        """
        Compute merged time windows of successful state attempts during an experiment.

        Args:
            folder (str): Recording folder identifier.
            log_values(DataFrame): DataFrame containing experiment log entries.
            graph_values (DataFrame): DataFrame containing graph recording timestamps.

        Returns:
            List[Tuple[float, float]]: Merged (start, end) timestamp windows for successful attempts.
        """
        if graph_values.empty:
            return []
        timestamps = graph_values.iloc[:, 0].to_numpy(dtype=float)
        experiment_first_timestamp = float(timestamps[0] - 1)
        experiment_last_timestamp = float(timestamps[-1] + 1)
        recording_ranges = self.get_timestamps_for_all_experiment_recordings(
            log_values,
            experiment_first_timestamp,
            experiment_last_timestamp,
        )
        state_attempts = self.get_timestamps_for_all_successful_state_attempts(
            folder,
            log_values,
            recording_ranges,
        )
        all_windows: List[Tuple[float, float]] = []
        for ranges in state_attempts.values():
            all_windows.extend(ranges)
        if not all_windows:
            return []
        return self._merge_windows(all_windows)

    @staticmethod
    def _merge_windows(windows: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping or adjacent time windows.

        Args:
            windows (Iterable[Tuple[float, float]]): Iterable of (start, end) timestamp pairs
        
        Returns:
            List[Tuple[float, float]]: Merged (start, end) windows sorted by start time.
        """
        ordered = sorted((float(start), float(end)) for start, end in windows if start < end)
        if not ordered:
            return []
        merged: List[Tuple[float, float]] = []
        cur_start, cur_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged




class ImageDatasetPreparer:
    """
    Prepare image datasets from rig recorder data, optionally filtering frames
    and applying pipette detection and focusing.
    """
    def __init__(
        self,
        rig_data_root: Path,
        *,
        use_detector1: bool = True,
        filter_images: bool = True,
    ) -> None:
        """
        Initialize an ImageDatasetPreparer.

        Args:
            rig_data_root (Path): Root folder for rig recorder data.
            use_detector1 (bool): Toggle which pipette detector to use (default True).
            filter_images (bool): Whether to apply frame filtering (default True).

        Raises:
            FileNotFoundError: If 'rig_data_root' does not exist.
        """
        self.rig_data_root = Path(rig_data_root)
        if not self.rig_data_root.exists():
            raise FileNotFoundError(f"Rig-recorder root not found: {self.rig_data_root}")

        self.filter_images = filter_images
        self._filter_helper: Optional[_DatasetFilterHelper]
        if self.filter_images:
            try:
                self._filter_helper = _DatasetFilterHelper(self.rig_data_root)
            except Exception as exc:
                logging.warning("Failed to initialise DatasetBuilder2 filter helper: %s", exc)
                self._filter_helper = None
                self.filter_images = False
            else:
                logging.info("Frame filtering enabled for ImageDatasetPreparer")
        else:
            self._filter_helper = None

        self.detector = PipetteDetector1() #if use_detector1 else PipetteDetector2()
        self.focuser = PipetteFocuser()

    def build_csv(
        self,
        demo_folder: str,
        *,
        output_name: str = "cv_movement_recording.csv",
        camera_subdir: str = "camera_frames",
    ) -> Path:
        """
        Build a merged CSV of stage and pipette positions for a demo
        
        Collects camera frames and movement data from the demo folder, optionally filters
        frames, infers pipette coordinates, merges with stage movement timestamps, and
        writes a semicolon-delimited csv containing: timestamp, st_x, st_y, st_z, pi_x, pi_y,
        pi_z.

        Args:
            demo_folder (str): Name of the demo folder.
            output_name (str, optional): Name for the output CSV.
                Defaults to "cv_movement_recording.csv".
            camera_subdir (str, optional): Subdirectory containing camera frames.
                Defaults to "camera_frames".
        
                Returns:
                    Path: Path to generated CSV file.
        """
        demo_path = self._resolve_demo_path(demo_folder)
        movement_path = demo_path / "movement_recording.csv"
        if not movement_path.exists():
            raise FileNotFoundError(f"movement_recording.csv not found: {movement_path}")

        camera_dir = demo_path / camera_subdir
        if not camera_dir.exists():
            raise FileNotFoundError(f"Camera frame folder not found: {camera_dir}")


        frame_paths = self._collect_frame_paths(camera_dir)
        if not frame_paths:
            raise RuntimeError(f"No image frames found in {camera_dir}")

        if self.filter_images:
            frame_paths = self._apply_frame_filter(frame_paths, demo_path)
            if not frame_paths:
                raise RuntimeError("No image frames remained after applying demonstration filter")

        logging.info("Loaded %d frames from %s", len(frame_paths), camera_dir)

        frame_records = self._infer_pipette_coordinates(frame_paths)
        if not frame_records:
            raise RuntimeError("No valid frames remained after inference preprocessing")
        logging.info("Inference complete for %d frames", len(frame_records))

        movement_df = pd.read_csv(movement_path, sep=";")
        required_columns = {"timestamp", "st_x", "st_y", "st_z"}
        missing_columns = required_columns - set(movement_df.columns)
        if missing_columns:
            raise ValueError(f"movement_recording.csv missing columns: {sorted(missing_columns)}")
        movement_df = movement_df.sort_values("timestamp").reset_index(drop=True)

        frame_df = pd.DataFrame([record.__dict__ for record in frame_records])
        frame_df = frame_df.sort_values("timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            frame_df,
            movement_df[["timestamp", "st_x", "st_y", "st_z"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        missing_stage = merged[["st_x", "st_y", "st_z"]].isna().any(axis=1).sum()
        if missing_stage:
            logging.warning("Stage data missing for %d frames", missing_stage)

        merged = merged[["timestamp", "st_x", "st_y", "st_z", "pi_x", "pi_y", "pi_z"]]
        output_path = demo_path / output_name
        merged.to_csv(output_path, sep=";", index=False, float_format="%.6f")
        logging.info("Wrote %s", output_path)
        return output_path

    def _resolve_demo_path(self, demo_folder: str) -> Path:
        """
        Resolve the full path to a demo folder.
        
        Checks if 'demo_folder' exists as given or relative to the rig data root.

        Args:
            demo_folder (str): Name or path of the demo folder.

        Returns:
            Path: Resolved Path object pointing to the demo folder.
        
        Raises:
            FileNotFoundError: If the folder cannot be found.
        """
        candidate = Path(demo_folder)
        if candidate.is_dir():
            return candidate
        resolved = self.rig_data_root / demo_folder
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Rig-recorder demo folder not found: {demo_folder}")

    @staticmethod
    def _collect_frame_paths(camera_dir: Path) -> List[Path]:
        """
        Collect all image file paths in a folder.

        Args:
            camera_dir (Path): Directory containing camera frames.

        Returns:
            List[Path]: Sorted listed of image file paths with common image extensions.
        """
        frame_paths = [
            path
            for path in sorted(camera_dir.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        return frame_paths

    def _apply_frame_filter(self, frame_paths: Sequence[Path], demo_path: Path) -> List[Path]:
        """
        Filter image frames to include only those within demonstration windows.

        Args:
            frame_paths (Sequence[Path]): Paths of all camera frames.
            demo_path (Path): Path to the demo folder.
        
        Returns:
            List[Path]: Filtered list of frame paths within detected demo windows.
        """
        if not self.filter_images or self._filter_helper is None:
            return list(frame_paths)

        helper = self._filter_helper
        graph_df = helper.load_graph_values(demo_path.name)
        if graph_df is None:
            logging.warning("Graph recording missing or unreadable for %s; skipping frame filter", demo_path)
            return list(frame_paths)

        log_df = helper.load_log_values(demo_path.name)
        if log_df is None or log_df.empty:
            logging.warning("Log data missing for %s; skipping frame filter", demo_path)
            return list(frame_paths)

        windows = helper.compute_attempt_windows(demo_path.name, log_df, graph_df)
        if not windows:
            logging.info("No demonstration windows detected for %s; using all frames", demo_path)
            return list(frame_paths)

        filtered: List[Path] = []
        removed = 0
        for img_path in frame_paths:
            timestamp = self._parse_timestamp(img_path.name)
            if timestamp is None:
                removed += 1
                continue
            if any(start <= timestamp <= end for start, end in windows):
                filtered.append(img_path)
            else:
                removed += 1

        if not filtered:
            logging.warning("Frame filter removed all frames for %s; fallback to full set", demo_path)
            return list(frame_paths)

        if removed:
            logging.info("Filtered out %d frames outside demonstration windows for %s", removed, demo_path)
        return filtered


    def _infer_pipette_coordinates(self, frame_paths: Sequence[Path]) -> List[FrameRecord]:
        """
        Infer pipette (x, y, z) coordinates for each frame.

        Args:
            frame_paths (Sequence[Path]): Paths to image frames.

        Returns:
            List[FrameRecord]: List of frame records with timestamp and pipette coordinates.
        """
        records: List[FrameRecord] = []
        skipped_without_timestamp = 0
        failed_to_load = 0
        failed_detection = 0

        total_frames = len(frame_paths) if hasattr(frame_paths, "__len__") else None
        iterable = (
            tqdm(frame_paths, desc="Inferring pipette coordinates", unit="frame", total=total_frames)
            if tqdm is not None
            else frame_paths
        )

        try:
            for img_path in iterable:
                timestamp = self._parse_timestamp(img_path.name)
                if timestamp is None:
                    skipped_without_timestamp += 1
                    logging.debug("Skipping frame without timestamp: %s", img_path.name)
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    failed_to_load += 1
                    logging.debug("Failed to load image: %s", img_path)
                    continue

                xy = self.detector.detect_pipette(img)
                if xy is None:
                    failed_detection += 1
                    pi_x = np.nan
                    pi_y = np.nan
                else:
                    pi_x, pi_y = float(xy[0]), float(xy[1])

                pi_z = float(self.focuser.get_pipette_focus_value(img))
                records.append(FrameRecord(timestamp=timestamp, pi_x=pi_x, pi_y=pi_y, pi_z=pi_z))
        finally:
            if tqdm is not None and hasattr(iterable, "close"):
                iterable.close()

        if skipped_without_timestamp:
            logging.warning("Skipped %d frames without valid timestamps", skipped_without_timestamp)
        if failed_to_load:
            logging.warning("Failed to load %d frames", failed_to_load)
        if failed_detection:
            logging.info("Detector returned no result for %d frames", failed_detection)
        return records

    @staticmethod
    def _parse_timestamp(filename: str) -> Optional[float]:
        """
        Extract a numeric timestamp from an image filename.

        Args:
            filename (str): Name of the file, expected format includes an underscore
                before the timestamp.
            
        Returns:
            Optional[float]: Parsed timestamp as a float, or None if parsing fails.
        """
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        timestamp_str = "_".join(parts[1:])
        try:
            return float(timestamp_str)
        except ValueError:
            return None


def _configure_logging(verbose: bool) -> None:
    """
    Configure the root logger with a simple format and verbosity.

    Args:
        verbose (bool): If True, set logging level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def run_preparer(
    rig_data_root: Path,
    rig_recorder_data_folder_set: Sequence[str | Path],
    *,
    output_name: str = "cv_movement_recording.csv",
    use_detector1: bool = False,
    filter_images: bool = False,
    verbose: bool = False,
) -> None:
    """
    Process a set of rig-recorder demo folders to generate movement CSVs.

    Args:
        rig_data_root (Path): Root directory containing rig-recorder data.
        rig_recorder_data_folder_set (Sequence[str | Path]): List of demo folder names or paths.
        output_name (str, optional): Name for the generated CSV file. Defaults to "cv_movement_recording.csv".
        use_detector1 (bool, optional): If True, use PipetteDetector1; otherwise alternate detector. Defaults to False.
        filter_images (bool, optional): If True, filter frames using demonstration windows. Defaults to False.
        verbose (bool, optional): If True, enable debug-level logging. Defaults to False.

    Returns:
        None
    """
    _configure_logging(verbose)
    preparer = ImageDatasetPreparer(
        rig_data_root, use_detector1=use_detector1, filter_images=filter_images
    )
    total_folders = len(rig_recorder_data_folder_set) if hasattr(rig_recorder_data_folder_set, "__len__") else None
    iterator = (
        tqdm(rig_recorder_data_folder_set, desc="Rig-recorder folders", unit="folder", total=total_folders)
        if tqdm is not None
        else rig_recorder_data_folder_set
    )

    try:
        for folder in iterator:
            logging.info("Processing folder: %s", folder)
            preparer.build_csv(folder, output_name=output_name)
    finally:
        if tqdm is not None and hasattr(iterator, "close"):
            iterator.close()


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent / "Data" / "rig_recorder_data"
    

    rig_data_root = default_root
    # rig_recorder_data_folder_set = [
    #     "2025_09_25-20_43",
    #     "2025_09_25-21_39",
    #     "2025_10_01-13_15",
    #     "2025_10_01-13_30",
    # ]# training/valid for PipetteFinder 
    # rig_recorder_data_folder_set = [
    # "2025_09_25-22_13"]
     # test dataset for PipetteFinder

    # rig_recorder_data_folder_set = [
    #     "2025_05_20-15_50",
    #     "2025_05_20-15_16",
    #     "2025_05_20-14_05",
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    # ] # HEK training data (5/20/2025, 4/10/2025)


    # rig_recorder_data_folder_set = ["2025_04_07-14_50"] # HEK testing data

    rig_recorder_data_folder_set = ["2025_10_10-15_12"]

    output_name = "cv_movement_recording.csv"
    use_detector1 = False
    verbose = True
    filter_images = True

    run_preparer(
        rig_data_root,
        rig_recorder_data_folder_set,
        output_name=output_name,
        use_detector1=use_detector1,
        filter_images=filter_images,
        verbose=verbose,
    )
