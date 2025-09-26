"""DatasetBuilder2
====================

How to use
----------
1.  Import :class:`DatasetBuilder2` (and optionally :class:`DatasetBuilderSettings`)
    from ``experiments.DatasetBuilder2``.
2.  Instantiate a settings object or pass keyword arguments mirroring the
    original ``DatasetBuilder`` signature.  Only the parameters you wish to
    change need to be supplied; everything else falls back to the same
    defaults as ``DatasetBuilder``.
3.  Create ``DatasetBuilder2`` with the settings and call :meth:`add_demo` for
    each rig-recorder folder.  All downstream helper methods retain their
    names and behaviour, so existing scripts can swap the import with minimal
    edits.

Example
:::::::

.. code-block:: python

    from experiments.DatasetBuilder2 import DatasetBuilder2

    builder = DatasetBuilder2(
        dataset_name="HEK_inference_set5.hdf5",
        calfile=r"C:\\path\\to\\average_calibration_full.pickle",
        val_ratio=0.0,
        load_next_obs=True,
    )

    for folder in ["2025_04_07-15_50", "2025_05_20-14_05"]:
        builder.add_demo(folder, record_to_file=True)

    builder.write_split_masks()

This produces the same dataset artefacts as the original builder while using a
more modular internal structure that groups related functionality.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import h5py
import numpy as np
import pandas as pd
from PIL import Image


ATL_TO_UTC_TIME_DELTA = 4  # March 9 - Nov 1: 4 hours, otherwise 5 hours


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterSettings:
    enable_random_filter: bool = False # set to true to enable albumentations filtering
    image_filter_prob: float = 0.65
    filter_train_only: bool = False
    filter_same_per_demo: bool = False


@dataclass(slots=True)
class DatasetBuilderSettings:
    dataset_name: str
    calfile: Optional[str] = None
    val_ratio: float = 1 / 6 # fraction of demos to reserve for validation
    omit_stage_movement: bool = True # set to true to only record demos when stage is stationary
    random_seed: int = 0
    rotate_valid: bool = False # set to true to augment validation set with rotations
    stage_y_axis_flip: bool = True # set to true if the stage Y axis is inverted
    pipette_rotation_deg: float = -60.75 # angle to rotate pipette coordinates into stage frame
    load_next_obs: bool = False # set to true for goal conditioning
    frequency_mod: int = 1 # downsample data by this factor (minimum 1)
    filter: FilterSettings = field(default_factory=FilterSettings)

    # Legacy toggles preserved for parity with DatasetBuilder
    calibrate: bool = False # set to true to apply calibration transform
    zero_values: bool = False # set to true to zero out starting positions
    center_crop: bool = True # set to true to center crop images around pipette
    rotate: bool = False # set to true to augment training set with rotations
    inaction: int = 1 # maximum number of consecutive zero-action steps to keep


@dataclass(slots=True)
class _StateDatasetContext:
    state_name: str
    dataset_name: str
    dataset_dir: Path
    dataset_path: Path
    metadata_filename: str
    split_keys: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _slugify_state_name(name: str) -> str:
    """Return a filesystem friendly slug for a state name."""

    cleaned = name.strip().lower().replace(" ", "_")
    slug = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in cleaned)
    slug = slug.strip('_')
    return slug or 'state'


def _stable_int_seed(*parts: object) -> int:
    """Create a deterministic 32-bit integer seed from arbitrary parts."""
    data = ("||".join(map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def _shift_forward(arr: np.ndarray) -> np.ndarray:
    """Return a copy of ``arr`` shifted left with the final element repeated."""

    out = np.empty_like(arr)
    out[:-1] = arr[1:]
    out[-1] = arr[-1]
    return out


def _parse_waveform_column(column: Sequence[str]) -> np.ndarray:
    """Parse JSON-encoded voltage/current columns and pad to equal length."""

    lists = [json.loads(value) for value in column]
    max_len = max(len(lst) for lst in lists)
    padded = [lst + [lst[-1]] * (max_len - len(lst)) for lst in lists]
    return np.asarray(padded, dtype=np.float64)


def _resolve_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    """Return dataset directory and HDF5 file path for ``dataset_name``."""

    datasets_root = Path("experiments/Datasets")
    name_path = Path(dataset_name)
    file_name = name_path.name
    folder_name = name_path.stem if name_path.suffix else name_path.name
    dataset_dir = datasets_root / folder_name
    dataset_path = dataset_dir / file_name
    return dataset_dir, dataset_path


def _ensure_dataset_stub(dataset_name: str, create_file: bool = False) -> Tuple[Path, Path]:
    """Ensure dataset directory exists and optionally prepare an empty HDF5 stub."""

    dataset_dir, dataset_path = _resolve_dataset_paths(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    legacy_path = Path("experiments/Datasets") / dataset_name
    if legacy_path.exists() and not dataset_path.exists():
        legacy_path.replace(dataset_path)

    if create_file and not dataset_path.exists():
        with h5py.File(dataset_path, "w") as hf:
            group = hf.create_group("data")
            group.attrs["num_demos"] = 0

    return dataset_dir, dataset_path


class CalibrationMixin:
    """Retains the calibration-related API, mirroring DatasetBuilder."""

    calfile: Optional[str]

    def load_calfile(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Placeholder for calibration loading (maintains original stub)."""

        return None, None

    def apply_transform(
        self, stage_positions: np.ndarray, pipette_positions: np.ndarray, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Placeholder matching the original unimplemented behaviour."""

        raise NotImplementedError("Calibration transform not implemented.")

    def pixel_coordinate_transform(
        self, stage_positions: np.ndarray, pipette_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert stage/pipette coordinates into calibrated pixel space."""
        M, r0 = self.load_calfile()
        if M is None:
            print(f"no calibration matrix found in {self.calfile}!")
            print("passing uncalibrated inputs...")
            return stage_positions, pipette_positions
        return self.apply_transform(stage_positions, pipette_positions, M)


class RandomFilterMixin:
    """Albumentations augmentation wrapper kept functionally identical."""

    def __init__(self, settings: DatasetBuilderSettings):
        """Configure Albumentations filters from :class:`DatasetBuilderSettings`."""
        self._filter_settings = settings.filter
        self._rng_seed = settings.random_seed
        self._albu_filter = None
        if self._filter_settings.enable_random_filter:
            self._albu_filter = A.Compose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0,
                            ),
                            A.Sharpen(p=1.0),
                        ],
                        p=self._filter_settings.image_filter_prob,
                    )
                ],
                seed=settings.random_seed,
            )
        self._filter_active_for_demo = False
        self._albu_replay_comp: Optional[A.ReplayCompose] = None
        self._albu_replay_state = None

    # --- filter context -------------------------------------------------
    def begin_filter_context(self, split_label: str, demo_seed: int) -> None:
        """Prepare Albumentations state for a demo.

        Parameters
        ----------
        split_label:
            Either ``"train"`` or ``"valid"`` to indicate which dataset split
            the demo belongs to.  The flag controls whether filtering is
            enabled when ``filter_train_only`` is set.
        demo_seed:
            Integer seed that keeps per-demo replay filters deterministic when
            ``filter_same_per_demo`` is active.
        """
        cfg = self._filter_settings
        if not cfg.enable_random_filter or (
            cfg.filter_train_only and split_label != "train"
        ):
            self._filter_active_for_demo = False
            self._albu_replay_comp = None
            self._albu_replay_state = None
            return

        self._filter_active_for_demo = True
        if cfg.filter_same_per_demo:
            self._albu_replay_comp = A.ReplayCompose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0,
                            ),
                            A.Sharpen(p=1.0),
                        ],
                        p=cfg.image_filter_prob,
                    )
                ],
                seed=int(demo_seed),
            )
            self._albu_replay_state = None
        else:
            self._albu_replay_comp = None
            self._albu_replay_state = None

    def end_filter_context(self) -> None:
        """Reset augmentation state after finishing a demo."""
        self._filter_active_for_demo = False
        self._albu_replay_comp = None
        self._albu_replay_state = None

    def apply_albu_filter_to_pil(self, pil_image: Image.Image) -> Image.Image:
        """Apply the configured Albumentations pipeline to a PIL image.

        Parameters
        ----------
        pil_image:
            RGB or grayscale :class:`PIL.Image.Image` frame from the rig
            recorder dataset.

        Returns
        -------
        PIL.Image.Image
            The potentially augmented frame.  If filtering is disabled the
            input image object is returned unchanged.
        """
        if not self._filter_active_for_demo:
            return pil_image
        if self._albu_filter is None and self._albu_replay_comp is None:
            return pil_image

        np_img = np.array(pil_image.convert("RGB"))
        if self._albu_replay_comp is not None:
            if self._albu_replay_state is None:
                out = self._albu_replay_comp(image=np_img)
                self._albu_replay_state = out["replay"]
                np_img = out["image"]
            else:
                np_img = A.ReplayCompose.replay(
                    self._albu_replay_state, image=np_img
                )["image"]
        else:
            np_img = self._albu_filter(image=np_img)["image"]
        return Image.fromarray(np_img)


class DatasetBuilder2(CalibrationMixin, RandomFilterMixin):
    """Reorganised DatasetBuilder with identical public surface area."""

    def __init__(self, **kwargs):
        """Initialise the builder with keyword arguments from DatasetBuilder.

        Parameters
        ----------
        **kwargs:
            Any parameter accepted by the legacy :class:`DatasetBuilder`
            constructor.  See :class:`DatasetBuilderSettings` for the full list
            and default values.
        """
        settings = DatasetBuilderSettings(**kwargs)
        self.settings = settings
        super().__init__(settings)

        self.dataset_name = settings.dataset_name
        self.calfile = settings.calfile
        self.calibrate = settings.calibrate
        self.zero_values = settings.zero_values
        self.center_crop = settings.center_crop
        self.rotate = settings.rotate
        self.rotate_valid = settings.rotate_valid
        self.inaction = settings.inaction
        self.val_ratio = settings.val_ratio
        self.omit_stage_movement = settings.omit_stage_movement
        self.rng = np.random.default_rng(settings.random_seed)
        self.stage_y_axis_flip = settings.stage_y_axis_flip
        self.pipette_rotation_deg = settings.pipette_rotation_deg
        self.load_next_obs = settings.load_next_obs
        self.frequency_mod = int(max(1, settings.frequency_mod))

        self.dataset_dir, self.dataset_path = _ensure_dataset_stub(settings.dataset_name, create_file=False)
        self._base_dataset_name = self.dataset_name
        self._metadata_filename = "metadata.json"
        self._state_contexts: Dict[str, _StateDatasetContext] = {}

        if self.val_ratio == 0:
            self._split_keys = {"train": []}
        else:
            self._split_keys = {"train": [], "valid": []}

        self._write_metadata_files()

    def _ensure_state_context(self, state_name: str) -> _StateDatasetContext:
        """Create or return cached dataset bookkeeping for ``state_name``."""

        slug = _slugify_state_name(state_name)
        if slug in self._state_contexts:
            return self._state_contexts[slug]

        base_path = Path(self._base_dataset_name)
        suffix = base_path.suffix or ".hdf5"
        stem = base_path.stem if base_path.suffix else base_path.name
        dataset_name = f"{stem}_{slug}{suffix}"
        dataset_dir = self.dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / dataset_name

        if self.val_ratio == 0:
            split_keys = {"train": []}
        else:
            split_keys = {"train": [], "valid": []}

        metadata_filename = f"metadata_{slug}.json"
        context = _StateDatasetContext(
            state_name=slug,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            dataset_path=dataset_path,
            metadata_filename=metadata_filename,
            split_keys=split_keys,
        )
        self._state_contexts[slug] = context
        return context

    @contextmanager
    def _use_state_context(self, state_name: str):
        """Temporarily switch builder bookkeeping to a state-specific dataset."""

        context = self._ensure_state_context(state_name)
        original_name = self.dataset_name
        original_dir = self.dataset_dir
        original_path = self.dataset_path
        original_split_keys = self._split_keys
        original_metadata = self._metadata_filename

        self.dataset_name = context.dataset_name
        self.dataset_dir = context.dataset_dir
        self.dataset_path = context.dataset_path
        self._split_keys = context.split_keys
        self._metadata_filename = context.metadata_filename
        try:
            yield
        finally:
            context.split_keys = self._split_keys
            self.dataset_name = original_name
            self.dataset_dir = original_dir
            self.dataset_path = original_path
            self._split_keys = original_split_keys
            self._metadata_filename = original_metadata

    # ------------------------------------------------------------------
    def _transform_pipette_positions(
        self, stage_positions: np.ndarray, pipette_positions: np.ndarray
    ) -> np.ndarray:
        """Express pipette coordinates in the stage frame.

        Parameters
        ----------
        stage_positions:
            Array of stage XYZ coordinates for each timestep.
        pipette_positions:
            Array of manipulator XYZ coordinates for each timestep.

        Returns
        -------
        np.ndarray
            Pipette coordinates translated into the stage frame while keeping
            the Z axis untouched.
        """
        stage_adj = stage_positions.copy()
        if self.stage_y_axis_flip:
            stage_adj[:, 1] = -stage_adj[:, 1]

        pip_rot = self._rotate_positions(pipette_positions, self.pipette_rotation_deg)
        pip_rot[:, :2] += stage_adj[:, :2]
        return pip_rot

    # --- rotation helpers -------------------------------------------------
    @staticmethod
    def _rotate_positions(positions: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate XY coordinates by ``angle_degrees`` while keeping Z intact."""
        rad = np.deg2rad(angle_degrees)
        cos_val, sin_val = np.cos(rad), np.sin(rad)
        rotated = positions.copy()
        rotated[:, 0] = cos_val * positions[:, 0] - sin_val * positions[:, 1]
        rotated[:, 1] = sin_val * positions[:, 0] + cos_val * positions[:, 1]
        return rotated

    def _rotate_actions(self, actions: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate stage and pipette XY velocity components by ``angle_degrees``."""
        rad = np.deg2rad(angle_degrees)
        cos_val, sin_val = np.cos(rad), np.sin(rad)
        actions_rot = actions.copy()
        x, y = actions[:, 0], actions[:, 1]
        actions_rot[:, 0] = cos_val * x - sin_val * y
        actions_rot[:, 1] = sin_val * x + cos_val * y

        if actions.shape[1] >= 5:
            x_pip, y_pip = actions[:, 3], actions[:, 4]
            actions_rot[:, 3] = cos_val * x_pip - sin_val * y_pip
            actions_rot[:, 4] = sin_val * x_pip + cos_val * y_pip
        return actions_rot

    # --- filtering --------------------------------------------------------
    def filter_inactive_actions(self, actions: np.ndarray, *arrays: np.ndarray) -> tuple:
        """Drop contiguous segments where all action components remain zero."""
        if self.inaction == 0:
            return (actions,) + arrays

        inactive = (actions.sum(axis=1) == 0).astype(np.int8)
        diff = np.diff(np.concatenate(([0], inactive, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        keep = np.ones_like(inactive, dtype=bool)
        for s, e in zip(starts, ends):
            if e - s >= self.inaction:
                keep[s:e] = False

        filtered = (actions[keep],) + tuple(arr[keep] for arr in arrays)
        return filtered

    def _decimate_by_step(self, *arrays: Optional[np.ndarray], keep_last: bool = True):
        """Downsample arrays and return the shared index used for decimation."""
        step = self.frequency_mod
        if step <= 1:
            return arrays, None

        ref = next((a for a in arrays if a is not None), None)
        if ref is None:
            return arrays, None
        N = len(ref)
        if N == 0:
            return arrays, None

        idx = np.arange(0, N, step, dtype=np.int64)
        if keep_last:
            if idx.size == 0:
                idx = np.array([N - 1], dtype=np.int64)
            elif idx[-1] != N - 1:
                idx = np.concatenate([idx, np.array([N - 1], dtype=np.int64)])

        out = []
        for a in arrays:
            out.append(None if a is None else a[idx])
        return tuple(out), idx

    def _aggregate_actions_over_windows(
        self, actions: np.ndarray, idx: Optional[np.ndarray], mode: str = "sum"
    ) -> np.ndarray:
        """Aggregate action vectors between successive decimated indices."""
        if idx is None or len(idx) == 0:
            return actions

        if mode == "last":
            return actions[idx]
        if mode != "sum":
            raise ValueError("mode must be 'sum' or 'last'.")

        N = len(actions)
        ends = np.concatenate([idx[1:] - 1, np.array([N - 1], dtype=idx.dtype)])
        out = np.zeros((len(idx), actions.shape[1]), dtype=actions.dtype)
        for k, (a, b) in enumerate(zip(idx, ends)):
            out[k] = actions[a : b + 1].sum(axis=0)
        return out

    # --- CSV conversion utilities ----------------------------------------
    def convert_graph_recording_csv_to_new_format(self, demo_file_path: str) -> None:
        """Rewrite ``graph_recording.csv`` with semicolon-separated fields.

        Parameters
        ----------
        demo_file_path:
            Name of the rig-recorder folder containing ``graph_recording.csv``.
        """
        file_path = Path("experiments/Data/rig_recorder_data") / demo_file_path / "graph_recording.csv"
        graph_values = pd.read_csv(file_path, delimiter=":")

        converted_file_strings = ["timestamp;pressure;resistance;current;voltage\n"]
        for _, row in graph_values.iterrows():
            timestamp = row[1][:-10]
            pressure = row[2][:-12]
            resistance = row[3][:-9]
            current = row[4][:-8]
            voltage = row[5]
            converted_file_strings.append(
                f"{timestamp};{pressure};{resistance};{current};{voltage}\n"
            )

        with open(file_path, "w") as f:
            f.writelines(converted_file_strings)

        print(pd.read_csv(file_path, delimiter=";"))

    def convert_movement_recording_csv_to_new_format(self, demo_file_path: str) -> None:
        """Rewrite ``movement_recording.csv`` into the new semicolon format."""
        file_path = Path("experiments/Data/rig_recorder_data") / demo_file_path / "movement_recording.csv"
        movement_values = pd.read_csv(file_path, delimiter=":")

        converted_file_strings = ["timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z\n"]
        for _, row in movement_values.iterrows():
            timestamp = row[1][:-6]
            st_x = row[2][:-6]
            st_y = row[3][:-6]
            st_z = row[4][:-6]
            pi_x = row[5][:-5]
            pi_y = row[6][:-5]
            pi_z = row[7]
            converted_file_strings.append(
                f"{timestamp};{st_x};{st_y};{st_z};{pi_x};{pi_y};{pi_z}\n"
            )

        with open(file_path, "w") as f:
            f.writelines(converted_file_strings)

        print(pd.read_csv(file_path, delimiter=";"))

    # --- Experiment loading ----------------------------------------------
    def load_experiment_data(
        self, rig_recorder_data_folder: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load graph, movement, and log tables for a given experiment folder."""
        base = Path("experiments/Data/rig_recorder_data") / rig_recorder_data_folder
        graph_values = pd.read_csv(base / "graph_recording.csv", delimiter=";").to_numpy()
        movement_values = pd.read_csv(base / "movement_recording.csv", delimiter=";").to_numpy()
        log_file = Path("experiments/Data/log_data") / f"logs_{rig_recorder_data_folder[:10]}.csv"
        log_values = pd.read_csv(log_file, on_bad_lines="skip")
        return graph_values, movement_values, log_values

    # --- Log parsing -----------------------------------------------------
    def get_timestamps_for_all_experiment_recordings(
        self,
        log_values: pd.DataFrame,
        experiment_first_timestamp: float,
        experiment_last_timestamp: float,
    ) -> List[Tuple[float, float]]:
        """Return (start, end) timestamps for each recording within a session."""
        started_mask = log_values["Message"].str.contains(
            "Recording started", na=False
        )
        started_logs = log_values.loc[started_mask].copy()
        started_logs.loc[:, "Full Time"] = (
            pd.to_datetime(
                started_logs["Time(HH:MM:SS)"] + "." + started_logs["Time(ms)"].astype(str),
                format="%Y-%m-%d %H:%M:%S.%f",
            )
            + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
        ).apply(lambda x: x.timestamp())

        curr_started = started_logs[started_logs["Full Time"] > experiment_first_timestamp]
        curr_started = curr_started[curr_started["Full Time"] < experiment_last_timestamp]
        filtered_started = curr_started.drop_duplicates()

        ended_mask = log_values["Message"].str.contains(
            "Recording stopped", na=False
        )
        ended_logs = log_values.loc[ended_mask].copy()
        ended_logs.loc[:, "Full Time"] = (
            pd.to_datetime(
                ended_logs["Time(HH:MM:SS)"] + "." + ended_logs["Time(ms)"].astype(str),
                format="%Y-%m-%d %H:%M:%S.%f",
            )
            + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
        ).apply(lambda x: x.timestamp())

        curr_ended = ended_logs[ended_logs["Full Time"] > experiment_first_timestamp]
        curr_ended = curr_ended[curr_ended["Full Time"] < experiment_last_timestamp]
        filtered_ended = curr_ended.drop_duplicates()

        recording_start_times = list(filtered_started["Full Time"])
        recording_time_ranges: List[Tuple[float, float]] = []
        for end_ts in filtered_ended["Full Time"]:
            for i, start_ts in enumerate(recording_start_times):
                if i < len(recording_start_times) - 1:
                    if start_ts < end_ts < recording_start_times[i + 1]:
                        recording_time_ranges.append((start_ts, end_ts))
                else:
                    if start_ts < end_ts < experiment_last_timestamp:
                        recording_time_ranges.append((start_ts, end_ts))
        return recording_time_ranges


    def get_timestamps_for_all_successful_state_attempts(
        self,
        rig_recorder_data_folder: str,
        log_values: pd.DataFrame,
        recording_timestamp_ranges: Iterable[Tuple[float, float]],
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract successful attempt windows for each state using JSON logs."""

        state_attempts: Dict[str, List[Tuple[float, float]]] = {}
        rec_ranges = list(recording_timestamp_ranges)
        if not rec_ranges:
            return state_attempts

        state_root = Path("experiments/Data/state_recorder_data")
        day_token = rig_recorder_data_folder.split('-', 1)[0]
        tolerance = 0.5

        if state_root.exists():
            for day_dir in sorted(state_root.glob(f"{day_token}*")):
                if not day_dir.is_dir():
                    continue
                for attempt_dir in sorted(day_dir.glob('attempt_*')):
                    if not attempt_dir.is_dir():
                        continue
                    for json_path in sorted(attempt_dir.glob('*.json')):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as fh:
                                payload = json.load(fh)
                        except (OSError, json.JSONDecodeError):
                            continue

                        outcome = payload.get('outcome')
                        started = payload.get('started')
                        finished = payload.get('finished')
                        if outcome != 0 or started is None or finished is None:
                            continue
                        if finished <= started:
                            continue

                        stem = json_path.stem
                        parts = stem.split('_')
                        if len(parts) >= 3:
                            state_name = '_'.join(parts[1:-1])
                        else:
                            state_name = stem
                        slug = _slugify_state_name(state_name)

                        for rec_start, rec_end in rec_ranges:
                            if (rec_start - tolerance) <= started and finished <= (rec_end + tolerance):
                                state_attempts.setdefault(slug, []).append((started, finished))
                                break

        if 'hunt_cell' not in state_attempts:
            try:
                fallback = self.get_timestamps_for_all_successful_hunt_cell_attempts(
                    log_values, rec_ranges
                )
            except (TypeError, pd.errors.InvalidComparison):
                fallback = []
            if fallback:
                state_attempts['hunt_cell'] = fallback

        for attempts in state_attempts.values():
            attempts.sort(key=lambda window: window[0])

        return state_attempts

    def get_timestamps_for_all_successful_hunt_cell_attempts(
        self, log_values: pd.DataFrame, recording_timestamp_ranges: Iterable[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Extract attempt windows bracketed by resistance drop and cell events."""
        successful_ranges: List[Tuple[float, float]] = []
        for start_timestamp, end_timestamp in recording_timestamp_ranges:
            ir_mask = log_values["Message"].str.contains(
                "Initial resistance:", na=False
            )
            ir_logs = log_values.loc[ir_mask].copy()
            ir_logs.loc[:, "Full Time"] = (
                pd.to_datetime(
                    ir_logs["Time(HH:MM:SS)"] + "." + ir_logs["Time(ms)"].astype(str),
                    format="%Y-%m-%d %H:%M:%S.%f",
                )
                + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())
            ir_logs = ir_logs[(ir_logs["Full Time"] > start_timestamp) & (ir_logs["Full Time"] < end_timestamp)]
            filtered_ir_logs = ir_logs.drop_duplicates()

            cell_mask = log_values["Message"].str.contains(
                "Cell detected: True", na=False
            )
            cell_logs = log_values.loc[cell_mask].copy()
            cell_logs.loc[:, "Full Time"] = (
                pd.to_datetime(
                    cell_logs["Time(HH:MM:SS)"] + "." + cell_logs["Time(ms)"].astype(str),
                    format="%Y-%m-%d %H:%M:%S.%f",
                )
                + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())
            cell_logs = cell_logs[(cell_logs["Full Time"] > start_timestamp) & (cell_logs["Full Time"] < end_timestamp)]
            filtered_cell_logs = cell_logs.drop_duplicates()

            ir_start_times = list(filtered_ir_logs["Full Time"])
            for cell_ts in filtered_cell_logs["Full Time"]:
                for i, ir_ts in enumerate(ir_start_times):
                    if i < len(ir_start_times) - 1:
                        if ir_ts < cell_ts < ir_start_times[i + 1]:
                            successful_ranges.append((ir_ts, cell_ts))
                    else:
                        if ir_ts < cell_ts < end_timestamp:
                            successful_ranges.append((ir_ts, cell_ts))
        return successful_ranges

    # --- Graph / movement alignment -------------------------------------
    def truncate_graph_values(
        self, graph_values: np.ndarray, first_timestamp: float, last_timestamp: float
    ) -> np.ndarray:
        """Return graph rows closest to the provided time interval (inclusive)."""
        ts = graph_values[:, 0]
        i0 = np.searchsorted(ts, first_timestamp, side="left")
        if i0 == len(ts):
            i0 = len(ts) - 1
        elif i0 and abs(ts[i0 - 1] - first_timestamp) < abs(ts[i0] - first_timestamp):
            i0 -= 1

        i1 = np.searchsorted(ts, last_timestamp, side="right") - 1
        if i1 < 0:
            i1 = 0
        elif i1 + 1 < len(ts) and abs(ts[i1 + 1] - last_timestamp) < abs(ts[i1] - last_timestamp):
            i1 += 1
        return graph_values[i0 : i1 + 1]

    @staticmethod
    def associate_attempt_movement_and_graph_values(
        attempt_graph_values: np.ndarray, movement_values: np.ndarray
    ) -> np.ndarray:
        """Align each graph row with the closest movement sample in time."""
        g_ts = attempt_graph_values[:, 0]
        m_ts = movement_values[:, 0]
        right = np.searchsorted(m_ts, g_ts)
        left = np.clip(right - 1, 0, len(m_ts) - 1)
        right = np.clip(right, 0, len(m_ts) - 1)
        choose_right = np.abs(m_ts[right] - g_ts) < np.abs(m_ts[left] - g_ts)
        indices = np.where(choose_right, right, left)
        return movement_values[indices]

    # --- Attempt level feature extraction --------------------------------
    def get_attempt_dones(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Generate a dones vector with a terminal 1 at the final timestep."""
        dones = np.zeros(len(attempt_graph_values))
        dones[-1] = 1
        return dones

    def get_attempt_pressure_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Return the pressure column for the current attempt."""
        return attempt_graph_values[:, 1].astype(np.float64)

    def get_attempt_resistance_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Return resistance values (optionally zero-offset)."""
        resistance_values = attempt_graph_values[:, 2].astype(np.float64)
        if self.zero_values:
            resistance_values[:] -= resistance_values[0]
        return resistance_values

    def get_attempt_current_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Parse JSON-encoded current waveform samples for the attempt."""
        return _parse_waveform_column(attempt_graph_values[:, 3])

    def get_attempt_voltage_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Parse JSON-encoded voltage waveform samples for the attempt."""
        return _parse_waveform_column(attempt_graph_values[:, 4])

    def get_attempt_stage_positions(self, attempt_movement_values: np.ndarray) -> np.ndarray:
        """Return stage XYZ positions (optionally zero-offset)."""
        stage_positions = attempt_movement_values[:, 1:4].astype(np.float64)
        if self.zero_values:
            stage_positions -= stage_positions[0]
        return stage_positions

    def get_attempt_pipette_positions(self, attempt_movement_values: np.ndarray) -> np.ndarray:
        """Return pipette XYZ positions (optionally zero-offset)."""
        pipette_positions = attempt_movement_values[:, 4:].astype(np.float64)
        if self.zero_values:
            pipette_positions -= pipette_positions[0]
        return pipette_positions

    # --- Camera helpers --------------------------------------------------
    @staticmethod
    def crop_image_center(pil_image: Image.Image) -> Image.Image:
        """Return a centred half-resolution crop of ``pil_image``."""
        width, height = pil_image.size
        new_width = width // 2
        new_height = height // 2
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return pil_image.crop((left, top, right, bottom))

    def get_attempt_camera_frames(
        self,
        rig_recorder_data_folder: str,
        attempt_graph_values: np.ndarray,
        rotation_angle: Optional[float] = None,
    ) -> np.ndarray:
        """Load rig camera frames aligned to ``attempt_graph_values`` timestamps."""
        base = Path("experiments/Data/rig_recorder_data") / rig_recorder_data_folder / "camera_frames"
        camera_files = sorted(os.listdir(base))
        frames_list: List[np.ndarray] = []
        last_index = 0

        for i, graph_row in enumerate(attempt_graph_values):
            target_timestamp = graph_row[0]
            if i == len(attempt_graph_values) - 1:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i - 1][0])
            else:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i + 1][0])

            valid_indices: List[int] = []
            valid_timestamps: List[float] = []
            for j in range(last_index, len(camera_files)):
                camera_file = camera_files[j]
                underscore_index = camera_file.find("_")
                last_period_index = camera_file.rfind(".")
                camera_timestamp = float(camera_file[underscore_index + 1 : last_period_index])
                if abs(target_timestamp - camera_timestamp) < timestamp_range:
                    valid_indices.append(j)
                    valid_timestamps.append(camera_timestamp)
                else:
                    if valid_indices:
                        break

            if not valid_indices:
                warnings.warn(
                    f"No camera frame matched timestamp {target_timestamp} in {rig_recorder_data_folder}; skipping attempt.",
                    RuntimeWarning,
                )
                return None

            min_idx = valid_indices[0]
            min_diff = float("inf")
            for idx, ts in zip(valid_indices, valid_timestamps):
                diff = abs(target_timestamp - ts)
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            pil_image = Image.open(base / camera_files[min_idx])
            pil_image = self.apply_albu_filter_to_pil(pil_image)
            if rotation_angle is not None:
                pil_image = pil_image.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
            if self.center_crop:
                pil_image = self.crop_image_center(pil_image)
            frames_list.append(np.array(pil_image.resize((85, 85))))
            last_index = max(0, min_idx - 1)

        return np.array(frames_list)

    # --- Observations ----------------------------------------------------
    def get_attempt_observations(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        rig_recorder_data_folder: str,
        include_camera: bool = True,
        rotation_angle: Optional[float] = None,
    ):
        """Return pressure, resistance, waveform, position, and optional image arrays."""
        pressure_values = self.get_attempt_pressure_values(attempt_graph_values)
        resistance_values = self.get_attempt_resistance_values(attempt_graph_values)
        current_values = self.get_attempt_current_values(attempt_graph_values)
        voltage_values = self.get_attempt_voltage_values(attempt_graph_values)
        stage_positions = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions = self.get_attempt_pipette_positions(attempt_movement_values)

        pipette_positions = self._transform_pipette_positions(stage_positions, pipette_positions)

        if rotation_angle is not None:
            stage_positions = self._rotate_positions(stage_positions, rotation_angle)
            pipette_positions = self._rotate_positions(pipette_positions, rotation_angle)

        if include_camera:
            camera_frames = self.get_attempt_camera_frames(
                rig_recorder_data_folder, attempt_graph_values, rotation_angle=rotation_angle
            )
            if camera_frames is None:
                return None
            return (
                pressure_values,
                resistance_values,
                current_values,
                voltage_values,
                stage_positions,
                pipette_positions,
                camera_frames,
            )

        return (
            pressure_values,
            resistance_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
        )

    def get_attempt_next_observations(
        self,
        attempt_graph_values: np.ndarray,
        current_values: np.ndarray,
        voltage_values: np.ndarray,
        stage_positions: np.ndarray,
        pipette_positions: np.ndarray,
        camera_frames: Optional[np.ndarray],
        include_next_obs: bool = False,
        include_camera: bool = True,
    ):
        """Compute next-step observation arrays using :func:`_shift_forward`."""
        if not include_next_obs:
            return (None,) * 7

        pressure = attempt_graph_values[:, 1].astype(np.float64)
        resistance = attempt_graph_values[:, 2].astype(np.float64)

        next_pressure_values = _shift_forward(pressure)
        next_resistance_values = _shift_forward(resistance)
        next_current_values = _shift_forward(current_values)
        next_voltage_values = _shift_forward(voltage_values)
        next_stage_positions = _shift_forward(stage_positions)
        next_pipette_positions = _shift_forward(pipette_positions)

        if include_camera and camera_frames is not None:
            next_camera_frames = _shift_forward(camera_frames)
            return (
                next_pressure_values,
                next_resistance_values,
                next_current_values,
                next_voltage_values,
                next_stage_positions,
                next_pipette_positions,
                next_camera_frames,
            )

        return (
            next_pressure_values,
            next_resistance_values,
            next_current_values,
            next_voltage_values,
            next_stage_positions,
            next_pipette_positions,
            None,
        )

    # --- Action computation ----------------------------------------------
    def get_attempt_actions(
        self,
        attempt_movement_values: np.ndarray,
        attempt_graph_values: np.ndarray,
        log_values: pd.DataFrame,
        include_high_level_actions: bool = False,
    ) -> np.ndarray:
        """Return low-level deltas (and optional command hashes) per timestep."""
        movement_actions = np.diff(attempt_movement_values[:, 1:], axis=0)
        movement_actions = np.vstack([
            np.zeros(attempt_movement_values.shape[1] - 1),
            movement_actions,
        ])

        stage_delta = movement_actions[:, :3].copy()
        pip_raw_delta = movement_actions[:, 3:6].copy()

        if self.stage_y_axis_flip:
            stage_delta[:, 1] = -stage_delta[:, 1]

        pip_rot_delta = self._rotate_positions(pip_raw_delta, self.pipette_rotation_deg)

        movement_actions[:, :3] = stage_delta
        movement_actions[:, 3:5] = stage_delta[:, :2] + pip_rot_delta[:, :2]
        movement_actions[:, 5] = pip_rot_delta[:, 2]

        if include_high_level_actions:
            action_logs = log_values[
                log_values["Message"].str.contains("Executing command", na=False)
            ].copy()
            action_logs.loc[:, "Full Time"] = (
                pd.to_datetime(
                    action_logs["Time(HH:MM:SS)"] + "." + action_logs["Time(ms)"].astype(str),
                    format="%Y-%m-%d %H:%M:%S.%f",
                )
                + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())

            mask = (action_logs["Full Time"] > attempt_movement_values[0, 0]) & (
                action_logs["Full Time"] < attempt_movement_values[-1, 0]
            )
            action_logs = action_logs.loc[mask].drop_duplicates()

            hi_lvl = np.full((movement_actions.shape[0], 1), hash("None"), dtype=np.int64)
            for ts, msg in zip(action_logs["Full Time"], action_logs["Message"]):
                idx = np.argmin(np.abs(attempt_graph_values[:, 0] - ts))
                hi_lvl[idx, 0] = hash(msg[19:])
            movement_actions = np.hstack([movement_actions, hi_lvl])

        return movement_actions

    # --- Dataset writing -------------------------------------------------
    def add_attempt_demo_to_dataset(
        self,
        num_samples: int,
        actions: np.ndarray,
        dones: np.ndarray,
        pressure_values: np.ndarray,
        resistance_values: np.ndarray,
        current_values: np.ndarray,
        voltage_values: np.ndarray,
        stage_positions: np.ndarray,
        pipette_positions: np.ndarray,
        camera_frames: Optional[np.ndarray],
        next_pressure_values: Optional[np.ndarray],
        next_resistance_values: Optional[np.ndarray],
        next_current_values: Optional[np.ndarray],
        next_voltage_values: Optional[np.ndarray],
        next_stage_positions: Optional[np.ndarray],
        next_pipette_positions: Optional[np.ndarray],
        next_camera_frames: Optional[np.ndarray],
        include_next_obs: bool = False,
        include_camera: bool = True,
        split_label: str = "train",
    ) -> str:
        """Persist a demo to disk and return the HDF5 key used for the group."""
        payload: List[Optional[np.ndarray]] = [
            actions,
            dones,
            pressure_values,
            resistance_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
        ]
        if include_camera:
            payload.append(camera_frames)
        if include_next_obs:
            payload.extend(
                [
                    next_pressure_values,
                    next_resistance_values,
                    next_current_values,
                    next_voltage_values,
                    next_stage_positions,
                    next_pipette_positions,
                ]
            )
            if include_camera:
                payload.append(next_camera_frames)

        filtered = self.filter_inactive_actions(*payload)
        cursor = 0
        actions = filtered[cursor]; cursor += 1
        dones = filtered[cursor]; cursor += 1
        pressure_values = filtered[cursor]; cursor += 1
        resistance_values = filtered[cursor]; cursor += 1
        current_values = filtered[cursor]; cursor += 1
        voltage_values = filtered[cursor]; cursor += 1
        stage_positions = filtered[cursor]; cursor += 1
        pipette_positions = filtered[cursor]; cursor += 1
        if include_camera:
            camera_frames = filtered[cursor]; cursor += 1

        if include_next_obs:
            next_pressure_values = filtered[cursor]; cursor += 1
            next_resistance_values = filtered[cursor]; cursor += 1
            next_current_values = filtered[cursor]; cursor += 1
            next_voltage_values = filtered[cursor]; cursor += 1
            next_stage_positions = filtered[cursor]; cursor += 1
            next_pipette_positions = filtered[cursor]; cursor += 1
            if include_camera:
                next_camera_frames = filtered[cursor]

        num_samples = actions.shape[0]

        if self.frequency_mod > 1:
            if include_camera:
                (
                    _,
                    dones,
                    pressure_values,
                    resistance_values,
                    current_values,
                    voltage_values,
                    stage_positions,
                    pipette_positions,
                    camera_frames,
                ), idx = self._decimate_by_step(
                    None,
                    dones,
                    pressure_values,
                    resistance_values,
                    current_values,
                    voltage_values,
                    stage_positions,
                    pipette_positions,
                    camera_frames,
                )
            else:
                (
                    _,
                    dones,
                    pressure_values,
                    resistance_values,
                    current_values,
                    voltage_values,
                    stage_positions,
                    pipette_positions,
                    _,
                ), idx = self._decimate_by_step(
                    None,
                    dones,
                    pressure_values,
                    resistance_values,
                    current_values,
                    voltage_values,
                    stage_positions,
                    pipette_positions,
                    None,
                )

            actions = self._aggregate_actions_over_windows(actions, idx, mode="sum")
            num_samples = actions.shape[0]

            if include_next_obs and num_samples > 0:
                next_resistance_values = _shift_forward(resistance_values)
                next_stage_positions = _shift_forward(stage_positions)
                next_pipette_positions = _shift_forward(pipette_positions)
                if include_camera and camera_frames is not None:
                    next_camera_frames = _shift_forward(camera_frames)

        with h5py.File(self.dataset_path, "a") as hf:
            if "data" not in hf:
                data_group = hf.create_group("data")
                data_group.attrs["num_demos"] = 0
            else:
                data_group = hf["data"]
            demo_number = data_group.attrs["num_demos"]
            demo_name = f"demo_{demo_number}"
            demo = data_group.create_group(demo_name)
            demo.attrs["num_samples"] = num_samples
            demo.attrs["split"] = split_label

            demo.create_dataset("actions", data=actions)
            demo.create_dataset("dones", data=dones)

            observations = demo.create_group("obs")
            observations.create_dataset("resistance", data=resistance_values.reshape(-1, 1))
            observations.create_dataset("stage_positions", data=stage_positions)
            observations.create_dataset("pipette_positions", data=pipette_positions)
            if include_camera and camera_frames is not None:
                observations.create_dataset("camera_image", data=camera_frames)

            if include_next_obs:
                next_obs = demo.create_group("next_obs")
                next_obs.create_dataset("resistance", data=next_resistance_values.reshape(-1, 1))
                next_obs.create_dataset("stage_positions", data=next_stage_positions)
                next_obs.create_dataset("pipette_positions", data=next_pipette_positions)
                if include_camera and next_camera_frames is not None:
                    next_obs.create_dataset("camera_image", data=next_camera_frames)

            data_group.attrs["num_demos"] = demo_number + 1
            print(
                f"Added {split_label} {demo_name} to dataset '{self.dataset_name}' with {num_samples} samples."
            )
        self._write_metadata_files()
        return demo_name

    # --- Dataset bookkeeping --------------------------------------------
    def _collect_metadata(self) -> dict:
        """Aggregate dataset metadata for JSON/CSV export."""

        settings_dict = asdict(self.settings)
        toggles = {
            "calibrate": self.calibrate,
            "zero_values": self.zero_values,
            "center_crop": self.center_crop,
            "rotate": self.rotate,
            "inaction": self.inaction,
        }

        split_keys = {key: list(values) for key, values in self._split_keys.items()}
        split_counts = {key: 0 for key in split_keys}
        num_demos = 0

        if self.dataset_path.exists():
            with h5py.File(self.dataset_path, "r") as hf:
                if "data" in hf:
                    data_group = hf["data"]
                    num_demos = int(data_group.attrs.get("num_demos", 0))
                    for demo_name in data_group.keys():
                        split = data_group[demo_name].attrs.get("split")
                        if isinstance(split, bytes):
                            split = split.decode()
                        if split is None:
                            continue
                        split_counts[split] = split_counts.get(split, 0) + 1

        metadata = {
            "dataset_name": self.dataset_name,
            "dataset_directory": str(self.dataset_dir),
            "dataset_path": str(self.dataset_path),
            "num_demos": num_demos,
            "split_counts": split_counts,
            "split_keys": split_keys,
            "settings": settings_dict,
            "toggles": toggles,
        }

        json_path = self.dataset_dir / self._metadata_filename
        created_at = None
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                created_at = existing.get("created_at")
            except (OSError, json.JSONDecodeError, TypeError):
                created_at = None

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["created_at"] = created_at or now_iso
        metadata["updated_at"] = now_iso
        return metadata

    def _write_metadata_files(self) -> None:
        """Persist dataset metadata to JSON beside the HDF5 file."""

        metadata = self._collect_metadata()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.dataset_dir / self._metadata_filename
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

    def write_split_masks(self) -> None:
        """Write train/valid demo names into each dataset's ``mask`` group."""
        if self.val_ratio == 0:
            print(
                "val_ratio is 0 - dataset contains only training demos; skipping mask creation."
            )
            self._write_metadata_files()
            for state in sorted(self._state_contexts):
                with self._use_state_context(state):
                    self._write_metadata_files()
            return

        def _write_current_masks() -> None:
            if not self.dataset_path.exists():
                return
            with h5py.File(self.dataset_path, "a") as hf:
                if "data" not in hf:
                    return
                if "mask" in hf:
                    del hf["mask"]
                mask_grp = hf.create_group("mask")
                for name in ("train", "valid"):
                    keys = np.asarray(self._split_keys[name], dtype="S")
                    mask_grp.create_dataset(name, data=keys)
            valid_count_local = len(self._split_keys.get("valid", []))
            print(
                f"wrote split masks: {len(self._split_keys['train'])} train | {valid_count_local} valid"
            )
            self._write_metadata_files()

        _write_current_masks()
        for state in sorted(self._state_contexts):
            with self._use_state_context(state):
                _write_current_masks()

    # Maintain backward-compatible private name
    def _write_split_masks(self) -> None:
        """Compatibility shim delegating to :meth:`write_split_masks`."""
        self.write_split_masks()

    # Backward-compatible helper aliases
    def _stable_int_seed(self, *parts: object) -> int:
        """Compatibility shim delegating to module-level :func:`_stable_int_seed`."""
        return _stable_int_seed(*parts)

    def _begin_demo_filter_context(self, split_label: str, demo_seed: int) -> None:
        """Compatibility shim for :meth:`begin_filter_context`."""
        self.begin_filter_context(split_label, demo_seed)

    def _end_demo_filter_context(self) -> None:
        """Compatibility shim for :meth:`end_filter_context`."""
        self.end_filter_context()

    def _apply_albu_filter_to_pil(self, pil_image: Image.Image) -> Image.Image:
        """Compatibility shim for :meth:`apply_albu_filter_to_pil`."""
        return self.apply_albu_filter_to_pil(pil_image)

    def _pixel_coordinate_transform(
        self, stage_positions: np.ndarray, pipette_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility shim for :meth:`pixel_coordinate_transform`."""
        return self.pixel_coordinate_transform(stage_positions, pipette_positions)

    def _load_calfile(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compatibility shim for :meth:`CalibrationMixin.load_calfile`."""
        return self.load_calfile()

    def _apply_transform(
        self, stage_positions: np.ndarray, pipette_positions: np.ndarray, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility shim for :meth:`CalibrationMixin.apply_transform`."""
        return self.apply_transform(stage_positions, pipette_positions, M)

    # --- High level orchestration ---------------------------------------

    def add_demo(self, rig_recorder_data_folder: str, record_to_file: bool = False) -> None:
        """Parse a rig-recorder folder, extracting successful attempts into per-state datasets."""
        print(f"Adding demos from rig_recorder_data_folder: {rig_recorder_data_folder}")

        include_next_obs = self.load_next_obs
        include_camera = True
        include_high_level_actions = False

        graph_values, movement_values, log_values = self.load_experiment_data(
            rig_recorder_data_folder
        )

        experiment_first_timestamp = graph_values[0][0] - 1
        experiment_last_timestamp = graph_values[-1][0] + 1

        rec_ranges = self.get_timestamps_for_all_experiment_recordings(
            log_values, experiment_first_timestamp, experiment_last_timestamp
        )
        state_attempts = self.get_timestamps_for_all_successful_state_attempts(
            rig_recorder_data_folder, log_values, rec_ranges
        )

        if not state_attempts:
            print("  no successful state attempts detected; skipping demo export")
            return

        for state_name, attempt_ranges in state_attempts.items():
            if not attempt_ranges:
                continue
            print(f"  processing state '{state_name}' with {len(attempt_ranges)} attempts")
            with self._use_state_context(state_name):
                for attempt_first_timestamp, attempt_last_timestamp in attempt_ranges:
                    attempt_graph_values = self.truncate_graph_values(
                        graph_values, attempt_first_timestamp, attempt_last_timestamp
                    )
                    attempt_movement_values = self.associate_attempt_movement_and_graph_values(
                        attempt_graph_values, movement_values
                    )

                    dones = self.get_attempt_dones(attempt_graph_values)

                    split_lbl = "valid" if self.rng.random() < self.val_ratio else "train"

                    demo_seed = _stable_int_seed(
                        self.dataset_name,
                        rig_recorder_data_folder,
                        attempt_first_timestamp,
                        attempt_last_timestamp,
                        state_name,
                    )
                    self.begin_filter_context(split_lbl, demo_seed)

                    observations = self.get_attempt_observations(
                        attempt_graph_values,
                        attempt_movement_values,
                        rig_recorder_data_folder,
                        include_camera=include_camera,
                        rotation_angle=None,
                    )
                    if observations is None:
                        print("    skipped - missing camera frames")
                        self.end_filter_context()
                        continue

                    (
                        pressure_values,
                        resistance_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                    ) = observations

                    next_obs = self.get_attempt_next_observations(
                        attempt_graph_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                        include_next_obs=include_next_obs,
                        include_camera=include_camera,
                    )

                    actions = self.get_attempt_actions(
                        attempt_movement_values,
                        attempt_graph_values,
                        log_values,
                        include_high_level_actions=include_high_level_actions,
                    )

                    if self.omit_stage_movement and np.any(actions[:, :3]):
                        print("    skipped - demo contains stage movement")
                        self.end_filter_context()
                        continue

                    if record_to_file:
                        demo_key = self.add_attempt_demo_to_dataset(
                            num_samples=attempt_graph_values.shape[0],
                            actions=actions,
                            dones=dones,
                            pressure_values=pressure_values,
                            resistance_values=resistance_values,
                            current_values=current_values,
                            voltage_values=voltage_values,
                            stage_positions=stage_positions,
                            pipette_positions=pipette_positions,
                            camera_frames=camera_frames,
                            next_pressure_values=next_obs[0],
                            next_resistance_values=next_obs[1],
                            next_current_values=next_obs[2],
                            next_voltage_values=next_obs[3],
                            next_stage_positions=next_obs[4],
                            next_pipette_positions=next_obs[5],
                            next_camera_frames=next_obs[6],
                            include_next_obs=include_next_obs,
                            include_camera=include_camera,
                            split_label=split_lbl,
                        )
                        self._split_keys[split_lbl].append(demo_key)
                        print(f"    added original {split_lbl} demo")

                        self.end_filter_context()
                        self._write_metadata_files()
                    else:
                        self.end_filter_context()

                    if self.rotate and record_to_file and (
                        split_lbl == "train" or (split_lbl == "valid" and self.rotate_valid)
                    ):
                        angles = np.linspace(0, 360, num=10, endpoint=False)[1:]
                        for angle in angles:
                            aug_demo_seed = _stable_int_seed(
                                self.dataset_name,
                                rig_recorder_data_folder,
                                attempt_first_timestamp,
                                attempt_last_timestamp,
                                f"angle={angle}",
                            )
                            self.begin_filter_context(split_lbl, aug_demo_seed)

                            aug_observations = self.get_attempt_observations(
                                attempt_graph_values,
                                attempt_movement_values,
                                rig_recorder_data_folder,
                                include_camera=include_camera,
                                rotation_angle=angle,
                            )
                            if aug_observations is None:
                                print("    skipped augmented demo - missing camera frames")
                                self.end_filter_context()
                                continue

                            (
                                aug_pressure,
                                aug_resistance,
                                aug_current,
                                aug_voltage,
                                aug_stage_pos,
                                aug_pipette_pos,
                                aug_cam,
                            ) = aug_observations

                            aug_actions = self._rotate_actions(actions, angle)

                            aug_next_obs = self.get_attempt_next_observations(
                                attempt_graph_values,
                                current_values,
                                voltage_values,
                                aug_stage_pos,
                                aug_pipette_pos,
                                aug_cam,
                                include_next_obs=include_next_obs,
                                include_camera=include_camera,
                            )

                            aug_key = self.add_attempt_demo_to_dataset(
                                num_samples=attempt_graph_values.shape[0],
                                actions=aug_actions,
                                dones=dones,
                                pressure_values=aug_pressure,
                                resistance_values=aug_resistance,
                                current_values=aug_current,
                                voltage_values=aug_voltage,
                                stage_positions=aug_stage_pos,
                                pipette_positions=aug_pipette_pos,
                                camera_frames=aug_cam,
                                next_pressure_values=aug_next_obs[0],
                                next_resistance_values=aug_next_obs[1],
                                next_current_values=aug_next_obs[2],
                                next_voltage_values=aug_next_obs[3],
                                next_stage_positions=aug_next_obs[4],
                                next_pipette_positions=aug_next_obs[5],
                                next_camera_frames=aug_next_obs[6],
                                include_next_obs=include_next_obs,
                                include_camera=include_camera,
                                split_label=split_lbl,
                            )
                            self._split_keys[split_lbl].append(aug_key)
                            print(f"    added augmented {split_lbl} demo (angle {angle} deg)")

                            self.end_filter_context()
                            self._write_metadata_files()



__all__ = [
    "DatasetBuilder2",
    "DatasetBuilderSettings",
    "FilterSettings",
]


if __name__ == "__main__":
    dataset_name = "PatcherBot_dataset_v0_001.hdf5"
    # rig_recorder_data_folder_set =  ["2025_03_11-16_32"] # inference test data (3/11/2025), unseen
    rig_recorder_data_folder_set = ["2025_09_25-20_43"]
    # rig_recorder_data_folder_set = [
    #     "2025_05_20-15_50",
    #     "2025_05_20-15_16",
    #     "2025_05_20-14_05",
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    # ]


    builder = DatasetBuilder2(
        dataset_name=dataset_name,
        calfile=r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\experiments\\Datasets\\average_calibration_full.pickle",
        val_ratio=0,
        omit_stage_movement=True,
        random_seed=0,
        load_next_obs=False,
    )

    for folder in rig_recorder_data_folder_set:
        print(f"Processing folder: {folder}")
        builder.add_demo(rig_recorder_data_folder=folder, record_to_file=True)

    builder.write_split_masks()

