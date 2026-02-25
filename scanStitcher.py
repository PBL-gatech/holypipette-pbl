import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class ScanStitcher:
    STANDARD_DATA_ROOT = 'experiments/Data/rig_recorder_data'

    def __init__(
        self,
        date_time_folder=None,
        date_time_root=STANDARD_DATA_ROOT,
        use_calibration=True,
        coord_cols=(1, 2),
        flip_x=False,
        flip_y=True,
        prefer_exts=(".webp", ".jpg", ".jpeg"),
        stitch_fraction=1,
        downsample=10,
        max_canvas_gb=1.0,
        use_pixel_drift_stitch=True,
    ):
        print("[ScanStitcher] Initializing...")
        if date_time_folder is None:
            raise ValueError(
                "ScanStitcher now requires a date_time_folder. "
                "Expected standard layout under experiments/Data/rig_recorder_data/<date_time_folder>/"
            )

        self.load_inputs_from_date_time_folder(
            date_time_folder=date_time_folder,
            date_time_root=date_time_root,
        )
        self.prefer_exts = prefer_exts
        self.use_calibration = use_calibration
        self.coord_cols = coord_cols
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.stitch_fraction = float(stitch_fraction)
        self.downsample = self._validate_downsample(downsample)
        self.max_canvas_gb = float(max_canvas_gb)
        self.use_pixel_drift_stitch = bool(use_pixel_drift_stitch)
        self._calibration_reported = False
        self.calibration = self._load_calibration()

        self.coord_data = self._load_coords(self.coord_file)
        self.time_stamps = self.coord_data[:, 0]
        self.canvas = None
        self._streaming_canvas_path = None
        self._streaming_canvas_shape = None
        self.frame_catalog = self._build_frame_catalog()
        print(f"[ScanStitcher] Loaded {len(self.coord_data)} movement rows and {len(self.frame_catalog)} frames.")
        print(f"[ScanStitcher] Stitch mode: {'pixel_drift' if self.use_pixel_drift_stitch else 'projection'}")
        print(f"[ScanStitcher] Stitch downsample: 1/{self.downsample}")

    @staticmethod
    def _validate_downsample(downsample):
        try:
            value = int(downsample)
        except (TypeError, ValueError) as exc:
            raise ValueError("downsample must be a positive integer.") from exc
        if value < 1:
            raise ValueError("downsample must be >= 1.")
        return value

    def _resolve_date_time_folder(self, date_time_folder, date_time_root=STANDARD_DATA_ROOT):
        if os.path.isabs(date_time_folder):
            if os.path.isdir(date_time_folder):
                return os.path.normpath(date_time_folder)
            raise FileNotFoundError(f"date_time folder was not found: {date_time_folder}")

        candidates = [
            os.path.join(date_time_root, date_time_folder),
        ]

        for candidate in candidates:
            resolved = os.path.normpath(candidate)
            if os.path.isdir(resolved):
                return resolved

        searched = "\n  - ".join(os.path.normpath(c) for c in candidates)
        raise FileNotFoundError(
            f"Could not resolve date_time folder '{date_time_folder}'. Searched:\n  - {searched}"
        )

    def load_inputs_from_date_time_folder(self, date_time_folder, date_time_root=STANDARD_DATA_ROOT):
        dataset_dir = self._resolve_date_time_folder(
            date_time_folder=date_time_folder,
            date_time_root=date_time_root,
        )

        movement_path = None
        for name in ("cv_movement_recording.csv", "movement_recording.csv"):
            candidate = os.path.join(dataset_dir, name)
            if os.path.exists(candidate):
                movement_path = candidate
                break
        if movement_path is None:
            raise FileNotFoundError(
                f"Missing movement recording in {dataset_dir}. "
                "Expected cv_movement_recording.csv or movement_recording.csv."
            )

        photo_dir = os.path.join(dataset_dir, "camera_frames")
        if not os.path.isdir(photo_dir):
            raise FileNotFoundError(f"Missing camera_frames folder in dataset directory: {dataset_dir}")

        calibration_file = os.path.join(dataset_dir, "calibration.json")

        self.coord_file = movement_path
        self.photo_dir = photo_dir
        self.calibration_file = calibration_file
        print(f"[ScanStitcher] Using dataset folder: {dataset_dir}")
        print(f"[ScanStitcher] movement file: {self.coord_file}")
        print(f"[ScanStitcher] camera frames: {self.photo_dir}")
        print(f"[ScanStitcher] calibration file: {self.calibration_file}")
        return dataset_dir

    def _dataset_output_dir(self):
        return os.path.dirname(os.path.abspath(self.photo_dir))

    def _load_calibration(self):
        if not self.use_calibration:
            return None
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"use_calibration=True but calibration file was not found: {self.calibration_file}"
            )

        with open(self.calibration_file, "r") as f:
            payload = json.load(f)

        if "stage" not in payload:
            raise ValueError(
                f"use_calibration=True but 'stage' key is missing in calibration file: {self.calibration_file}"
            )

        stage = payload.get("stage", {})
        M = np.asarray(stage.get("M", []), dtype=float)
        r0 = np.asarray(stage.get("r0", [0, 0]), dtype=float)
        if M.size == 0:
            raise ValueError(
                f"use_calibration=True but stage matrix M is empty in: {self.calibration_file}"
            )

        print(f"[ScanStitcher] Stage calibration enabled from: {self.calibration_file}")
        print(f"[ScanStitcher] stage.M shape={M.shape}, stage.r0={r0.tolist()}")
        return payload

    def _apply_calibration(self, x, y):
        if self.calibration is None:
            return x, y

        manip = self.calibration.get("stage", {})
        M = np.asarray(manip.get("M", []), dtype=float)
        r0 = np.asarray(manip.get("r0", [0, 0]), dtype=float)

        if M.size == 0:
            return x, y

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.column_stack((x - r0[0], y - r0[1]))

        if M.shape == (2, 2):
            mapped = pts @ M.T
        elif M.shape == (2, 3):
            mapped = np.column_stack((x - r0[0], y - r0[1], np.ones_like(x))) @ M.T
        elif M.shape == (3, 3):
            mapped = np.column_stack((x - r0[0], y - r0[1], np.ones_like(x))) @ M.T
            mapped = mapped[:, :2]
        else:
            return x, y

        if not self._calibration_reported and mapped.shape[0] > 0:
            dx = mapped[:, 0] - x
            dy = mapped[:, 1] - y
            mad_x = float(np.mean(np.abs(dx)))
            mad_y = float(np.mean(np.abs(dy)))
            print(
                "[ScanStitcher] Calibration applied. "
                f"Mean absolute delta: dx={mad_x:.4f}, dy={mad_y:.4f}"
            )
            self._calibration_reported = True

        return mapped[:, 0], mapped[:, 1]

    def _aligned_coords_from_raw(self, coords):
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coords must be an array-like Nx2 of x, y positions.")
        x = coords_arr[:, 0]
        y = coords_arr[:, 1]
        x, y = self._apply_calibration(x, y)
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return np.column_stack((x, y))

    def _to_pixels(self, x, y):
        if self.calibration is None:
            return x, y

        manip = self.calibration.get("stage", {})
        M = np.asarray(manip.get("M", []), dtype=float)
        r0 = np.asarray(manip.get("r0", [0, 0]), dtype=float)
        if M.size == 0:
            return x, y

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.column_stack((x, y))

        M_h = np.eye(3)
        if M.shape == (2, 2):
            M_h[:2, :2] = M
        elif M.shape == (2, 3):
            M_h[:2, :3] = M
        elif M.shape == (3, 3):
            M_h = M
        else:
            return x, y

        rhs = np.column_stack((pts - r0[:2], np.ones((pts.shape[0], 1))))
        mapped = rhs @ np.linalg.inv(M_h).T

        return mapped[:, 0], mapped[:, 1]

    def _load_coords(self, coord_path):
        print(f"[ScanStitcher] Loading movement coordinates from: {coord_path}")
        ext = os.path.splitext(coord_path)[1].lower()
        if ext == ".csv":
            rows = []

            with open(coord_path, "r") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    if ";" in clean_line:
                        fields = [v.strip() for v in clean_line.split(";")]
                    else:
                        fields = [v.strip() for v in clean_line.split(",")]

                    try:
                        rows.append([float(v) for v in fields if v != ""])
                    except ValueError:
                        continue

            out = np.array(rows, dtype=float)
            print(f"[ScanStitcher] Parsed {len(out)} movement rows from CSV.")
            return out
        out = pd.read_excel(coord_path).values
        print(f"[ScanStitcher] Parsed {len(out)} movement rows from spreadsheet.")
        return out

    def _build_frame_catalog(self):
        print(f"[ScanStitcher] Indexing frames in: {self.photo_dir}")
        records = []
        prefer_exts = tuple(ext.lower() for ext in self.prefer_exts)
        all_files = sorted(glob.glob(os.path.join(self.photo_dir, "*")), key=lambda p: self._camera_order_key(os.path.basename(p)))
        for frame_path in tqdm(all_files, desc="Indexing frame timestamps", unit="file"):
            if not os.path.isfile(frame_path):
                continue
            if os.path.splitext(frame_path)[1].lower() not in prefer_exts:
                continue
            ts = self._extract_timestamp_from_name(frame_path)
            if ts is not None:
                records.append((ts, frame_path))

        if not records:
            raise FileNotFoundError(f"No images found in {self.photo_dir} for extensions {self.prefer_exts}")

        print(f"[ScanStitcher] Indexed {len(records)} timestamped frames.")
        return sorted(records, key=lambda item: item[0])

    @staticmethod
    def _camera_order_key(name):
        underscore_index = name.find('_')
        dot_index = name.rfind('.')
        if underscore_index == -1:
            return name
        segment = name[underscore_index + 1 : dot_index if dot_index != -1 else None]
        if segment and segment.replace('.', '', 1).isdigit():
            segment = segment.zfill(3)
        suffix = name[dot_index:] if dot_index != -1 else ''
        return f"{name[:underscore_index + 1]}{segment}{suffix}"

    def _extract_timestamp_from_name(self, frame_path):
        filename = os.path.basename(frame_path)
        underscore_index = filename.find("_")
        last_period_index = filename.rfind(".")
        if underscore_index == -1:
            return None
        segment = filename[underscore_index + 1 : last_period_index if last_period_index != -1 else None]
        try:
            return float(segment)
        except (TypeError, ValueError):
            return None

    def _align_all_frames_to_movement(self):
        print("[ScanStitcher] Aligning all frames to nearest movement timestamp...")
        if self.coord_data.size == 0:
            raise ValueError("movement data is empty.")
        if len(self.frame_catalog) == 0:
            raise ValueError("frame catalog is empty.")

        movement = np.asarray(self.coord_data, dtype=float)
        col_x, col_y = self.coord_cols
        if movement.shape[1] <= max(0, col_x, col_y):
            raise ValueError("movement data does not include requested coord_cols.")

        order = np.argsort(movement[:, 0])
        movement = movement[order]
        m_ts = movement[:, 0]
        frame_ts = np.asarray([ts for ts, _ in self.frame_catalog], dtype=float)
        frame_paths = [path for _, path in self.frame_catalog]

        right = np.searchsorted(m_ts, frame_ts)
        left = np.clip(right - 1, 0, len(m_ts) - 1)
        right = np.clip(right, 0, len(m_ts) - 1)
        choose_right = np.abs(m_ts[right] - frame_ts) < np.abs(m_ts[left] - frame_ts)
        indices = np.where(choose_right, right, left)
        matched = movement[indices]
        coords = matched[:, [col_x, col_y]]
        print(f"[ScanStitcher] Aligned {len(frame_ts)} frames to movement samples.")
        return frame_ts, coords, frame_paths

    def _sample_aligned_data(self, timestamps, coords, frame_paths):
        n = len(timestamps)
        if n == 0:
            return timestamps, coords, frame_paths
        frac = min(max(self.stitch_fraction, 0.0), 1.0)
        if frac >= 1.0:
            print(f"[ScanStitcher] stitch_fraction={frac:.3f}; using all {n} frames.")
            return timestamps, coords, frame_paths
        keep = max(1, int(np.ceil(n * frac)))
        print(f"[ScanStitcher] stitch_fraction={frac:.3f}; using first {keep}/{n} frames for stitching.")
        return timestamps[:keep], coords[:keep], frame_paths[:keep]

    def _downsample_aligned_data(self, timestamps, coords, frame_paths):
        n = len(timestamps)
        if n == 0:
            return timestamps, coords, frame_paths
        if self.downsample <= 1:
            return timestamps, coords, frame_paths

        indices = np.arange(0, n, self.downsample, dtype=int)
        sampled_timestamps = np.asarray(timestamps)[indices]
        sampled_coords = np.asarray(coords)[indices]
        sampled_paths = [frame_paths[int(i)] for i in indices]
        print(
            f"[ScanStitcher] downsample=1/{self.downsample}; "
            f"using {len(indices)}/{n} frames for stitching."
        )
        return sampled_timestamps, sampled_coords, sampled_paths

    def _pixel_drift_draw_order(self, coordinates):
        """
        Build frame draw order for pixel-drift stitching.
        - Left-to-right rows: keep original order (later frames over earlier).
        - Right-to-left rows: reverse within the row (later frames under earlier).
        - Vertical transitions keep row stack order unchanged.
        """
        coords_arr = np.asarray(coordinates, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        n = coords_arr.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        row_starts = [0]
        for i in range(1, n):
            dx = abs(coords_arr[i, 0] - coords_arr[i - 1, 0])
            dy = abs(coords_arr[i, 1] - coords_arr[i - 1, 1])
            if dy > dx:
                row_starts.append(i)
        row_starts.append(n)

        draw_order = []
        rtl_rows = 0
        for start, end in zip(row_starts[:-1], row_starts[1:]):
            if end - start <= 1:
                draw_order.extend(range(start, end))
                continue
            if coords_arr[end - 1, 0] < coords_arr[start, 0]:
                rtl_rows += 1
                draw_order.extend(range(end - 1, start - 1, -1))
            else:
                draw_order.extend(range(start, end))

        if rtl_rows > 0:
            print(
                "[ScanStitcher] Applied underlay ordering for "
                f"{rtl_rows} right-to-left row(s)."
            )

        return np.asarray(draw_order, dtype=int)

    def _build_timestamp_aligned_frame_paths(self, target_timestamps):
        """Align camera frames to timestamps using local-window nearest matching."""
        target_ts = np.asarray(target_timestamps, dtype=float).reshape(-1)
        if target_ts.size == 0:
            raise ValueError("target_timestamps is empty.")

        frame_times = np.array([ts for ts, _ in self.frame_catalog], dtype=float)
        if frame_times.size == 0:
            raise FileNotFoundError(f"No indexed frames available in {self.photo_dir}.")

        aligned_paths = []
        last_index = 0

        for i, target_timestamp in enumerate(target_ts):
            if target_ts.size == 1:
                timestamp_range = float("inf")
            elif i == target_ts.size - 1:
                timestamp_range = abs(target_timestamp - target_ts[i - 1])
            else:
                timestamp_range = abs(target_timestamp - target_ts[i + 1])

            if timestamp_range <= 0:
                timestamp_range = np.finfo(float).eps

            valid_indices = []
            valid_timestamps = []
            for j in range(last_index, len(frame_times)):
                camera_timestamp = frame_times[j]
                if abs(target_timestamp - camera_timestamp) < timestamp_range:
                    valid_indices.append(j)
                    valid_timestamps.append(camera_timestamp)
                elif valid_indices:
                    break

            if not valid_indices:
                raise RuntimeError(
                    f"No camera frame matched timestamp {target_timestamp} in {self.photo_dir}."
                )

            min_idx = valid_indices[0]
            min_diff = float("inf")
            for idx, ts in zip(valid_indices, valid_timestamps):
                diff = abs(target_timestamp - ts)
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            aligned_paths.append(self.frame_catalog[min_idx][1])
            last_index = max(0, min_idx - 1)

        return aligned_paths

    def _find_frame(self, target_time):
        aligned = self._build_timestamp_aligned_frame_paths([target_time])
        return aligned[0]

    def _to_bgr(self, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Unable to decode image: {image_path}")

        if ext == ".webp":
            success, jpg_bytes = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Failed to convert WebP to JPEG in-memory: {image_path}")
            frame = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)

        return frame

    def _to_rgb(self, image_path):
        return cv2.cvtColor(self._to_bgr(image_path), cv2.COLOR_BGR2RGB)

    def _load_frame_rgb(self, target_time):
        image_path = self._find_frame(target_time)
        return self._to_rgb(image_path)

    def save_projection_image_from_aligned_data(
        self,
        aligned_timestamps,
        aligned_coordinates,
        aligned_frame_paths,
        output_path="projection_image.png",
        coordinate_scale=1.0,
        frame_padding=0,
    ):
        """Build projection_image.png from already timestamp-aligned data."""
        timestamps_arr = np.asarray(aligned_timestamps, dtype=float).reshape(-1)
        coords_arr = np.asarray(aligned_coordinates, dtype=float)
        frame_paths = list(aligned_frame_paths)

        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("aligned_coordinates must be an array-like Nx2 of x, y positions.")
        if len(timestamps_arr) != len(coords_arr):
            raise ValueError("aligned_timestamps and aligned_coordinates must have the same number of rows.")
        if len(frame_paths) != len(coords_arr):
            raise ValueError("aligned_frame_paths and aligned_coordinates must have the same length.")
        if len(frame_paths) == 0:
            raise ValueError("aligned_frame_paths is empty.")

        print(f"[ScanStitcher] Loading {len(frame_paths)} aligned frames for projection image...")
        camera_frames = [
            self._to_rgb(path)
            for path in tqdm(frame_paths, desc="Loading aligned frames", unit="frame")
        ]
        return self.save_projection_image(
            timestamps=timestamps_arr,
            coordinates=coords_arr[:, :2],
            camera_frames=camera_frames,
            output_path=output_path,
            coordinate_scale=coordinate_scale,
            frame_padding=frame_padding,
        )

    def save_projection_image(
        self,
        timestamps,
        coordinates,
        camera_frames,
        output_path=None,
        coordinate_scale=1.0,
        frame_padding=0,
    ):
        """Save a non-overlapping, block-based projection from timestamp-aligned frames."""
        print("[ScanStitcher] Creating projection canvas...")
        canvas = self._create_projection_canvas(
            timestamps=timestamps,
            coordinates=coordinates,
            camera_frames=camera_frames,
            coordinate_scale=coordinate_scale,
            frame_padding=frame_padding,
        )
        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "projection_image.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"[ScanStitcher] Writing projection image: {output_path}")
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write projection image: {output_path}")
        return output_path

    def _create_projection_canvas(
        self,
        timestamps,
        coordinates,
        camera_frames,
        coordinate_scale=1.0,
        frame_padding=0,
    ):
        print("[ScanStitcher] Building non-overlapping projection canvas...")
        timestamps_arr = np.asarray(timestamps, dtype=float).reshape(-1)
        coords_arr = np.asarray(coordinates, dtype=float)
        frames_list = list(camera_frames)

        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        if len(timestamps_arr) != len(coords_arr):
            raise ValueError("timestamps and coordinates must have the same number of rows.")
        if len(frames_list) != len(coords_arr):
            raise ValueError("camera_frames and coordinates must have the same length.")
        if len(frames_list) == 0:
            raise ValueError("camera_frames is empty.")

        valid_xy = np.isfinite(coords_arr[:, 0]) & np.isfinite(coords_arr[:, 1])
        if not np.all(valid_xy):
            timestamps_arr = timestamps_arr[valid_xy]
            coords_arr = coords_arr[valid_xy]
            frames_list = [frame for frame, keep in zip(frames_list, valid_xy) if keep]

        if len(frames_list) == 0:
            raise ValueError("No valid entries after filtering non-finite coordinates.")

        sample = np.asarray(frames_list[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        if sample.shape[2] < 3:
            raise ValueError("camera frames must contain 3 channels.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]

        stride_x = max(1, tile_w + int(max(frame_padding, 0)))
        stride_y = max(1, tile_h + int(max(frame_padding, 0)))

        scale = float(coordinate_scale)
        if scale <= 0:
            raise ValueError("coordinate_scale must be > 0.")

        x = (coords_arr[:, 0] - np.min(coords_arr[:, 0])) * scale
        y = (coords_arr[:, 1] - np.min(coords_arr[:, 1])) * scale
        gx = np.rint(x).astype(int)
        gy = np.rint(y).astype(int)
        gx -= int(np.min(gx))
        gy -= int(np.min(gy))

        canvas_h = int(np.max(gy) + 1) * stride_y + tile_h
        canvas_w = int(np.max(gx) + 1) * stride_x + tile_w
        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            gx = np.floor(gx * shrink).astype(int)
            gy = np.floor(gy * shrink).astype(int)
            gx -= int(np.min(gx))
            gy -= int(np.min(gy))
            canvas_h = int(np.max(gy) + 1) * stride_y + tile_h
            canvas_w = int(np.max(gx) + 1) * stride_x + tile_w
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Canvas too large; auto-downscaled grid "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        occupied = set()

        for idx, frame in enumerate(tqdm(frames_list, desc="Placing frames on canvas", unit="frame")):
            frame_rgb = np.asarray(frame)
            if frame_rgb.ndim == 2:
                frame_rgb = np.repeat(frame_rgb[:, :, None], 3, axis=2)
            elif frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
                continue
            frame_rgb = frame_rgb[:, :, :3]
            if frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0:
                continue
            if frame_rgb.shape[:2] != (tile_h, tile_w):
                frame_rgb = cv2.resize(frame_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            left = int(gx[idx]) * stride_x
            top = int(gy[idx]) * stride_y
            while (top, left) in occupied:
                left += stride_x

            right = left + tile_w
            bottom = top + tile_h
            if right > canvas.shape[1]:
                canvas = np.concatenate(
                    [canvas, np.zeros((canvas.shape[0], right - canvas.shape[1], 3), dtype=np.uint8)],
                    axis=1,
                )
            if bottom > canvas.shape[0]:
                canvas = np.concatenate(
                    [canvas, np.zeros((bottom - canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)],
                    axis=0,
                )

            canvas[top:bottom, left:right] = frame_rgb
            occupied.add((top, left))
        print("[ScanStitcher] Canvas build complete.")
        return canvas

    def _create_pixel_drift_canvas(self, coordinates, camera_frames):
        print("[ScanStitcher] Building pixel-drift stitched canvas...")
        coords_arr = np.asarray(coordinates, dtype=float)
        frames_list = list(camera_frames)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        if len(frames_list) != len(coords_arr):
            raise ValueError("camera_frames and coordinates must have the same length.")
        if len(frames_list) == 0:
            raise ValueError("camera_frames is empty.")

        sample = np.asarray(frames_list[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3 or sample.shape[2] < 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]

        # Zero at first point so drift is relative to acquisition start.
        drift = coords_arr[:, :2] - coords_arr[0, :2]
        dx = np.rint(drift[:, 0]).astype(int)
        dy = np.rint(drift[:, 1]).astype(int)

        min_dx = int(np.min(dx))
        min_dy = int(np.min(dy))
        max_dx = int(np.max(dx))
        max_dy = int(np.max(dy))
        canvas_w = (max_dx - min_dx) + tile_w
        canvas_h = (max_dy - min_dy) + tile_h

        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            dx = np.floor(dx * shrink).astype(int)
            dy = np.floor(dy * shrink).astype(int)
            min_dx = int(np.min(dx))
            min_dy = int(np.min(dy))
            max_dx = int(np.max(dx))
            max_dy = int(np.max(dy))
            canvas_w = (max_dx - min_dx) + tile_w
            canvas_h = (max_dy - min_dy) + tile_h
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Drift canvas too large; auto-downscaled offsets "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Drift canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        draw_order = self._pixel_drift_draw_order(coords_arr)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for draw_idx in tqdm(draw_order, desc="Blending frames by drift", unit="frame"):
            frame_rgb = np.asarray(frames_list[int(draw_idx)])
            if frame_rgb.ndim == 2:
                frame_rgb = np.repeat(frame_rgb[:, :, None], 3, axis=2)
            elif frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
                continue
            frame_rgb = frame_rgb[:, :, :3]
            if frame_rgb.shape[:2] != (tile_h, tile_w):
                frame_rgb = cv2.resize(frame_rgb, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            left = int(dx[int(draw_idx)] - min_dx)
            top = int(dy[int(draw_idx)] - min_dy)
            right = left + tile_w
            bottom = top + tile_h
            canvas[top:bottom, left:right] = frame_rgb

        print("[ScanStitcher] Pixel-drift canvas build complete.")
        return canvas

    def stitch_streaming(
        self,
        output_path=None,
        metadata_path=None,
        memmap_path=None,
        cleanup_memmap=True,
    ):
        """
        Stream pixel-drift stitching into a disk-backed canvas.
        This avoids loading every frame into RAM at once.
        """
        if not self.use_pixel_drift_stitch:
            raise RuntimeError(
                "stitch_streaming currently supports only pixel-drift stitch mode."
            )

        print("[ScanStitcher] Starting streaming stitch pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        if len(frame_paths) == 0:
            raise ValueError("No frames available for streaming stitch.")

        sample = self._to_bgr(frame_paths[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3 or sample.shape[2] < 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]

        # Zero at first point so drift is relative to acquisition start.
        drift = aligned_coords[:, :2] - aligned_coords[0, :2]
        dx = np.rint(drift[:, 0]).astype(int)
        dy = np.rint(drift[:, 1]).astype(int)

        min_dx = int(np.min(dx))
        min_dy = int(np.min(dy))
        max_dx = int(np.max(dx))
        max_dy = int(np.max(dy))
        canvas_w = (max_dx - min_dx) + tile_w
        canvas_h = (max_dy - min_dy) + tile_h

        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            dx = np.floor(dx * shrink).astype(int)
            dy = np.floor(dy * shrink).astype(int)
            min_dx = int(np.min(dx))
            min_dy = int(np.min(dy))
            max_dx = int(np.max(dx))
            max_dy = int(np.max(dy))
            canvas_w = (max_dx - min_dx) + tile_w
            canvas_h = (max_dy - min_dy) + tile_h
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Drift canvas too large; auto-downscaled offsets "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Drift canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "stitched_scan.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if memmap_path is None:
            stem, _ = os.path.splitext(output_path)
            memmap_path = f"{stem}.memmap"
        elif not os.path.isabs(memmap_path):
            memmap_path = os.path.join(self._dataset_output_dir(), memmap_path)
        memmap_path = os.path.normpath(memmap_path)
        memmap_dir = os.path.dirname(memmap_path)
        if memmap_dir:
            os.makedirs(memmap_dir, exist_ok=True)

        draw_order = self._pixel_drift_draw_order(aligned_coords)
        print(f"[ScanStitcher] Allocating disk-backed canvas: {memmap_path}")
        canvas_mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(canvas_h, canvas_w, 3))
        canvas_mm[:] = 0

        for draw_idx in tqdm(draw_order, desc="Streaming stitch frames", unit="frame"):
            draw_idx = int(draw_idx)
            frame_bgr = sample if draw_idx == 0 else self._to_bgr(frame_paths[draw_idx])
            if frame_bgr.ndim == 2:
                frame_bgr = np.repeat(frame_bgr[:, :, None], 3, axis=2)
            elif frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
                continue
            frame_bgr = frame_bgr[:, :, :3]
            if frame_bgr.shape[:2] != (tile_h, tile_w):
                frame_bgr = cv2.resize(frame_bgr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

            left = int(dx[draw_idx] - min_dx)
            top = int(dy[draw_idx] - min_dy)
            right = left + tile_w
            bottom = top + tile_h
            canvas_mm[top:bottom, left:right] = frame_bgr

        canvas_mm.flush()
        print(f"[ScanStitcher] Writing streamed stitched image: {output_path}")
        if not cv2.imwrite(output_path, canvas_mm):
            raise RuntimeError(f"Failed to write streamed stitched image: {output_path}")

        self.canvas = None
        self._streaming_canvas_path = memmap_path
        self._streaming_canvas_shape = tuple(canvas_mm.shape)

        if metadata_path is None and output_path:
            metadata_path = f"{os.path.splitext(output_path)[0]}_meta.json"
        if metadata_path is not None:
            if not os.path.isabs(metadata_path):
                metadata_path = os.path.join(self._dataset_output_dir(), metadata_path)
            metadata_path = os.path.normpath(metadata_path)
            metadata_dir = os.path.dirname(metadata_path)
            if metadata_dir:
                os.makedirs(metadata_dir, exist_ok=True)
            metadata = {
                "coord_file": self.coord_file,
                "photo_dir": self.photo_dir,
                "use_calibration": self.use_calibration,
                "flip_x": self.flip_x,
                "flip_y": self.flip_y,
                "use_pixel_drift_stitch": self.use_pixel_drift_stitch,
                "stitch_fraction": self.stitch_fraction,
                "downsample": self.downsample,
                "stitch_mode": "pixel_drift_streaming",
                "canvas_shape": self._streaming_canvas_shape,
                "num_frames": len(frame_paths),
                "streaming_memmap_path": memmap_path,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        del canvas_mm
        if cleanup_memmap:
            try:
                os.remove(memmap_path)
                self._streaming_canvas_path = None
            except OSError as exc:
                print(f"[ScanStitcher] Warning: failed to remove memmap file {memmap_path}: {exc}")

        print("[ScanStitcher] Streaming stitch complete.")
        return output_path

    def stitch(self):
        print("[ScanStitcher] Starting stitch pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        print(f"[ScanStitcher] Loading {len(frame_paths)} frames into memory...")
        camera_frames = [
            self._to_rgb(path)
            for path in tqdm(frame_paths, desc="Loading frames", unit="frame")
        ]
        if self.use_pixel_drift_stitch:
            self.canvas = self._create_pixel_drift_canvas(
                coordinates=aligned_coords,
                camera_frames=camera_frames,
            )
        else:
            self.canvas = self._create_projection_canvas(
                timestamps=frame_timestamps,
                coordinates=aligned_coords,
                camera_frames=camera_frames,
                coordinate_scale=1.0,
                frame_padding=0,
            )
        print("[ScanStitcher] Stitch complete.")
        return self.canvas

    def save_projection_from_dataset(self, output_path=None):
        print("[ScanStitcher] Starting projection-image pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "projection_image.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        saved = self.save_projection_image_from_aligned_data(
            aligned_timestamps=frame_timestamps,
            aligned_coordinates=aligned_coords,
            aligned_frame_paths=frame_paths,
            output_path=output_path,
            coordinate_scale=1.0,
            frame_padding=0,
        )
        print(f"[ScanStitcher] Projection image saved: {saved}")
        return saved

    def save_horizontal_overlap_from_dataset(self, output_path=None, use_sampling=False):
        print("[ScanStitcher] Building horizontal-motion overlap image...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        if use_sampling:
            frame_timestamps, coords, frame_paths = self._sample_aligned_data(
                frame_timestamps, coords, frame_paths
            )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )

        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available.")
        keep = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dx >= dy and (dx > 0 or dy > 0):
                keep.append(i)
        if len(keep) < 2:
            raise RuntimeError("Not enough horizontal-motion frames to build overlap image.")

        h_coords = aligned_coords[keep]
        h_paths = [frame_paths[i] for i in keep]
        print(f"[ScanStitcher] Horizontal overlap frames selected: {len(h_paths)} / {n}")
        h_frames = [
            self._to_rgb(path)
            for path in tqdm(h_paths, desc="Loading horizontal overlap frames", unit="frame")
        ]
        canvas = self._create_pixel_drift_canvas(
            coordinates=h_coords,
            camera_frames=h_frames,
        )

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "horizontal_overlap_stitched.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write horizontal overlap image: {output_path}")
        print(f"[ScanStitcher] Horizontal overlap image saved: {output_path}")
        return output_path

    def save_vertical_overlap_from_dataset(self, output_path=None, use_sampling=False):
        print("[ScanStitcher] Building vertical-motion overlap image...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        if use_sampling:
            frame_timestamps, coords, frame_paths = self._sample_aligned_data(
                frame_timestamps, coords, frame_paths
            )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )

        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available.")
        keep = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dy > dx and (dx > 0 or dy > 0):
                keep.append(i)
        if len(keep) < 2:
            raise RuntimeError("Not enough vertical-motion frames to build overlap image.")

        v_coords = aligned_coords[keep]
        v_paths = [frame_paths[i] for i in keep]
        print(f"[ScanStitcher] Vertical overlap frames selected: {len(v_paths)} / {n}")
        v_frames = [
            self._to_rgb(path)
            for path in tqdm(v_paths, desc="Loading vertical overlap frames", unit="frame")
        ]
        canvas = self._create_pixel_drift_canvas(
            coordinates=v_coords,
            camera_frames=v_frames,
        )

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "vertical_overlap_stitched.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write vertical overlap image: {output_path}")
        print(f"[ScanStitcher] Vertical overlap image saved: {output_path}")
        return output_path

    @staticmethod
    def _concat_frames(frames, concat_axis):
        if not frames:
            return None
        base = np.asarray(frames[0])
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        elif base.ndim != 3 or base.shape[2] < 3:
            return None
        base = base[:, :, :3]
        h, w = base.shape[:2]

        normalized = []
        for frame in frames:
            f = np.asarray(frame)
            if f.ndim == 2:
                f = np.repeat(f[:, :, None], 3, axis=2)
            elif f.ndim != 3 or f.shape[2] < 3:
                continue
            f = f[:, :, :3]
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            normalized.append(np.clip(f, 0, 255).astype(np.uint8))

        if not normalized:
            return None
        return np.concatenate(normalized, axis=concat_axis)

    def save_axis_debug_images(self, horizontal_output_path=None, vertical_output_path=None):
        print("[ScanStitcher] Building axis debug images...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available for axis debug images.")

        horizontal_indices = [0]
        vertical_indices = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dx == 0 and dy == 0:
                continue
            if dx >= dy:
                horizontal_indices.append(i)
            if dy > dx:
                vertical_indices.append(i)

        print(
            "[ScanStitcher] Axis split counts: "
            f"horizontal={len(horizontal_indices)}, vertical={len(vertical_indices)}"
        )

        h_frames = [
            self._to_rgb(frame_paths[i])
            for i in tqdm(horizontal_indices, desc="Loading horizontal frames", unit="frame")
        ]
        v_frames = [
            self._to_rgb(frame_paths[i])
            for i in tqdm(vertical_indices, desc="Loading vertical frames", unit="frame")
        ]

        horizontal_img = self._concat_frames(h_frames, concat_axis=0)
        vertical_img = self._concat_frames(v_frames, concat_axis=1)
        if horizontal_img is None or vertical_img is None:
            raise RuntimeError("Failed to build one or both axis debug images.")

        if horizontal_output_path is None:
            horizontal_output_path = os.path.join(self._dataset_output_dir(), "horizontal_only_debug.tif")
        elif not os.path.isabs(horizontal_output_path):
            horizontal_output_path = os.path.join(self._dataset_output_dir(), horizontal_output_path)

        if vertical_output_path is None:
            vertical_output_path = os.path.join(self._dataset_output_dir(), "vertical_only_debug.tif")
        elif not os.path.isabs(vertical_output_path):
            vertical_output_path = os.path.join(self._dataset_output_dir(), vertical_output_path)

        horizontal_output_path = os.path.normpath(horizontal_output_path)
        vertical_output_path = os.path.normpath(vertical_output_path)
        os.makedirs(os.path.dirname(horizontal_output_path), exist_ok=True) if os.path.dirname(horizontal_output_path) else None
        os.makedirs(os.path.dirname(vertical_output_path), exist_ok=True) if os.path.dirname(vertical_output_path) else None

        if not cv2.imwrite(horizontal_output_path, cv2.cvtColor(horizontal_img, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write horizontal debug image: {horizontal_output_path}")
        if not cv2.imwrite(vertical_output_path, cv2.cvtColor(vertical_img, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write vertical debug image: {vertical_output_path}")

        print(f"[ScanStitcher] Horizontal debug image saved: {horizontal_output_path}")
        print(f"[ScanStitcher] Vertical debug image saved: {vertical_output_path}")
        return horizontal_output_path, vertical_output_path

    def plot(self):
        if self.canvas is None:
            raise RuntimeError("Call stitch() before plot().")
        plt.figure(figsize=(10, 10))
        plt.imshow(self.canvas)
        plt.title("Image placement by coordinates")
        plt.axis("off")
        plt.show()

    def save(self, output_path=None, metadata_path=None):
        if self.canvas is None:
            raise RuntimeError("Call stitch() before save().")

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "stitched_scan.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        print(f"[ScanStitcher] Saving stitched image: {output_path}")
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".npy":
            np.save(output_path, self.canvas)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            bgr = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(output_path, bgr):
                raise RuntimeError(f"Failed to write image to {output_path}")
        else:
            raise ValueError(f"Unsupported image extension: {ext}")

        if metadata_path is None and output_path:
            metadata_path = f"{os.path.splitext(output_path)[0]}_meta.json"
        if metadata_path is not None:
            if not os.path.isabs(metadata_path):
                metadata_path = os.path.join(self._dataset_output_dir(), metadata_path)
            metadata_path = os.path.normpath(metadata_path)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True) if os.path.dirname(metadata_path) else None
            metadata = {
                "coord_file": self.coord_file,
                "photo_dir": self.photo_dir,
                "use_calibration": self.use_calibration,
                "flip_x": self.flip_x,
                "flip_y": self.flip_y,
                "use_pixel_drift_stitch": self.use_pixel_drift_stitch,
                "stitch_fraction": self.stitch_fraction,
                "downsample": self.downsample,
                "canvas_shape": self.canvas.shape,
                "num_frames": len(self.time_stamps),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path


def main():
    use_pixel_drift_stitch = True
    use_streaming_stitch = True
    downsample = 1
    date_time_folder = "2026_02_24-18_19"
    stitcher = ScanStitcher(
        date_time_folder=date_time_folder,
        use_calibration=True,
        flip_x=True,
        downsample=downsample,
        use_pixel_drift_stitch=use_pixel_drift_stitch,
    )
    # h_overlap = stitcher.save_horizontal_overlap_from_dataset("horizontal_overlap_stitched.tif")
    # print(f"Saved horizontal overlap image: {h_overlap}")
    # v_overlap = stitcher.save_vertical_overlap_from_dataset("vertical_overlap_stitched.tif")
    # print(f"Saved vertical overlap image: {v_overlap}")
    # h_dbg, v_dbg = stitcher.save_axis_debug_images(
    #     "horizontal_only_debug.tif",
    #     "vertical_only_debug.tif",
    # )
    # print(f"Saved horizontal debug image: {h_dbg}")
    # print(f"Saved vertical debug image: {v_dbg}")
    # saved_projection = stitcher.save_projection_from_dataset("projection_image.tif")
    # print(f"Saved projection image: {saved_projection}")
    if use_streaming_stitch and use_pixel_drift_stitch:
        saved_scan = stitcher.stitch_streaming(
            output_path="stitched_scan.tif",
            metadata_path="meta.json",
            cleanup_memmap=True,
        )
    else:
        stitcher.stitch()
        saved_scan = stitcher.save("stitched_scan.tif", "meta.json")
    print(f"Saved stitched scan: {saved_scan}")
    # stitcher.plot()



if __name__ == "__main__":
    main()
