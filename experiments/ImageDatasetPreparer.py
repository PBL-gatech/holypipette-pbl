"""Build pipette coordinate records from saved rig-recorder frames."""

from __future__ import annotations


import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holypipette.deepLearning.pipetteDetector import PipetteDetector1, PipetteDetector2
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser


@dataclass
class FrameRecord:
    timestamp: float
    pi_x: float
    pi_y: float
    pi_z: float


class ImageDatasetPreparer:
    def __init__(
        self,
        rig_data_root: Path,
        *,
        use_detector1: bool = False,
    ) -> None:
        self.rig_data_root = Path(rig_data_root)
        if not self.rig_data_root.exists():
            raise FileNotFoundError(f"Rig-recorder root not found: {self.rig_data_root}")

        self.detector = PipetteDetector1() if use_detector1 else PipetteDetector2()
        self.focuser = PipetteFocuser()

    def build_csv(
        self,
        demo_folder: str,
        *,
        output_name: str = "cv_movement_recording.csv",
        camera_subdir: str = "camera_frames",
    ) -> Path:
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
        candidate = Path(demo_folder)
        if candidate.is_dir():
            return candidate
        resolved = self.rig_data_root / demo_folder
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Rig-recorder demo folder not found: {demo_folder}")

    @staticmethod
    def _collect_frame_paths(camera_dir: Path) -> List[Path]:
        frame_paths = [
            path
            for path in sorted(camera_dir.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        return frame_paths

    def _infer_pipette_coordinates(self, frame_paths: Sequence[Path]) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        skipped_without_timestamp = 0
        failed_to_load = 0
        failed_detection = 0

        for img_path in frame_paths:
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

        if skipped_without_timestamp:
            logging.warning("Skipped %d frames without valid timestamps", skipped_without_timestamp)
        if failed_to_load:
            logging.warning("Failed to load %d frames", failed_to_load)
        if failed_detection:
            logging.info("Detector returned no result for %d frames", failed_detection)
        return records

    @staticmethod
    def _parse_timestamp(filename: str) -> Optional[float]:
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def run_preparer(
    rig_data_root: Path,
    rig_recorder_data_folder_set: Sequence[str | Path],
    *,
    output_name: str = "cv_movement_recording.csv",
    use_detector1: bool = False,
    verbose: bool = False,
) -> None:
    _configure_logging(verbose)
    preparer = ImageDatasetPreparer(rig_data_root, use_detector1=use_detector1)
    for folder in rig_recorder_data_folder_set:
        logging.info("Processing folder: %s", folder)
        preparer.build_csv(folder, output_name=output_name)


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent / "Data" / "rig_recorder_data"
    

    rig_data_root = default_root
    rig_recorder_data_folder_set = [
        "2025_09_25-20_43",
        "2025_09_25-21_39",
        "2025_10_01-13_15",
        "2025_10_01-13_30",
    ]
    output_name = "cv_movement_recording.csv"
    use_detector1 = False
    verbose = True

    run_preparer(
        rig_data_root,
        rig_recorder_data_folder_set,
        output_name=output_name,
        use_detector1=use_detector1,
        verbose=verbose,
    )
