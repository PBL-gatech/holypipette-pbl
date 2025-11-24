# necessary imports
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List


class ModelImporter:
    def __init__(self, onnx_model_path: str, input_normalization_npz_path: str,
                 output_normalization_npz_path: str, model_desc_json_path: str) -> None:
        """Store file system handles for a single exported policy."""
        self.onnx_path = Path(onnx_model_path)
        print(onnx_model_path)
        self.obs_npz_path = Path(input_normalization_npz_path) if input_normalization_npz_path else None
        self.action_npz_path = Path(output_normalization_npz_path) if output_normalization_npz_path else None
        self.model_json_path = Path(model_desc_json_path) if model_desc_json_path else None
        self.session = None
        self.obs_stats = {}
        self.action_stats = {}
        self.metadata = {}
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}

    def _find_first(self, base: Path, suffix: str) -> Path:
        """Locate the first file with the requested suffix under the provided path."""
        path = Path(base)
        if path.is_file():
            return path
        candidates = sorted(path.rglob(f"*{suffix}"))
        if not candidates:
            raise FileNotFoundError(f"Unable to locate {suffix} under {path}")
        return candidates[0]

    def load(self):
        """Instantiate the ONNX runtime session and collect metadata."""
        import json

        def _load_obs_stats(npz_path: Path) -> dict:
            stats = {}
            if npz_path and npz_path.exists():
                with np.load(npz_path) as data:
                    for key in data.files:
                        if "::" not in key:
                            continue
                        obs_key, attr = key.split("::")
                        entry = stats.setdefault(obs_key, {})
                        entry[attr] = data[key].astype(np.float32)
            return stats

        def _load_action_stats(npz_path: Path) -> dict:
            if npz_path and npz_path.exists():
                with np.load(npz_path) as data:
                    offset = data.get("offset")
                    scale = data.get("scale")
                    if offset is not None and scale is not None:
                        return {
                            "offset": offset.astype(np.float32),
                            "scale": scale.astype(np.float32),
                        }
            return {}

        resolve = lambda path, ending: self._find_first(path, ending) if path else None

        self.onnx_path = self._find_first(self.onnx_path, ".onnx")
        print(f"ONNX model: {self.onnx_path}")
        self.obs_npz_path = resolve(self.obs_npz_path, ".npz")
        self.action_npz_path = resolve(self.action_npz_path, ".npz")
        self.model_json_path = resolve(self.model_json_path, ".json")

        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.obs_stats = _load_obs_stats(self.obs_npz_path)
        self.action_stats = _load_action_stats(self.action_npz_path)
        if self.model_json_path and self.model_json_path.exists():
            self.metadata = json.loads(self.model_json_path.read_text(encoding="utf-8"))
        self.input_names = list(self.metadata.get("inputs", [])) or [inp.name for inp in self.session.get_inputs()]
        self.output_names = list(self.metadata.get("outputs", [])) or [out.name for out in self.session.get_outputs()]
        for inp in self.session.get_inputs():
            shape = tuple(1 if dim is None else int(dim) for dim in inp.shape)
            self.input_shapes[inp.name] = shape
        return self


class ModelInferencer:
    def __init__(self, model_importer: ModelImporter) -> None:
        """Bind a loaded policy to simple preprocessing and postprocessing utilities."""
        self.crop_size = 1 # set to 1 for version 0.200 and beyond
        self.image_resize = (85, 85)
        self.action_unnorm_object = None
        self.obs_norm_object = None
        self.goal_required = False
        self.goal = None
        self.internal_states = [0, 0]
        self._last_frame_params: Optional[Dict[str, float]] = None
        self._pipette_action_dim: Optional[int] = None
        self._debug_run_uid = datetime.now().strftime("%Y_%m_%d-%H_%M")


        self.importer = model_importer.load()
        self.session = self.importer.session
        self.obs_norm_object = self.importer.obs_stats
        self.action_unnorm_object = self.importer.action_stats
        self.metadata = self.importer.metadata
        self.obs_keys = list(self.metadata.get("observation_keys", []))
        self.goal_keys = list(self.metadata.get("goal_keys", []))
        self.goal_required = bool(self.goal_keys)
        self.requires_preprocessing = bool(self.metadata.get("requires_preprocessing", True))
        default_post = bool(self.action_unnorm_object)
        meta_post = self.metadata.get("requires_postprocessing")
        self.requires_postprocessing = default_post if meta_post is None else bool(meta_post)
        if not self.requires_postprocessing:
            self.action_unnorm_object = {}
        self.input_names = list(self.importer.input_names)
        self.output_names = list(self.importer.output_names)
        self.image_layout = "CHW"
        camera_shape = self.importer.input_shapes.get("obs::camera_image")
        if camera_shape and 3 in camera_shape and camera_shape.index(3) == len(camera_shape) - 1:
            self.image_layout = "HWC"
        if camera_shape:
            target_height = None
            target_width = None
            if self.image_layout == "CHW" and len(camera_shape) >= 4:
                target_height = camera_shape[-2]
                target_width = camera_shape[-1]
            elif self.image_layout == "HWC":
                if len(camera_shape) >= 4:
                    target_height = camera_shape[-3]
                    target_width = camera_shape[-2]
                elif len(camera_shape) == 3:
                    target_height = camera_shape[0]
                    target_width = camera_shape[1]
            if (
                isinstance(target_width, (int, float))
                and isinstance(target_height, (int, float))
                and target_width > 0
                and target_height > 0
            ):
                self.image_resize = (int(target_width), int(target_height))
        self.image_resize = (int(self.image_resize[0]), int(self.image_resize[1]))
        self.state_names = [name for name in self.input_names if not name.startswith("obs::") and not name.startswith("goal::")]
        self.state_buffers = {}
        self.default_inputs = {}
        for name, shape in self.importer.input_shapes.items():
            target = [1 if dim in (None, -1) else dim for dim in shape]
            zeros = np.zeros(target, dtype=np.float32)
            self.default_inputs[name] = zeros
            if name in self.state_names:
                self.state_buffers[name] = zeros.copy()
        self.action_dim = None

    def _prepare_image(
        self,
        image: np.ndarray,
        frame_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Center-crop and resize an RGB frame from the microscope."""
        frame = np.asarray(image, dtype=np.uint8)
        h, w = frame.shape[:2]
        params = frame_params or self._compute_frame_params((h, w))
        if params is None:
            return frame
        crop_h = int(params["crop_h"])
        crop_w = int(params["crop_w"])
        y0 = int(params["offset_y"])
        x0 = int(params["offset_x"])
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        return np.array(
            Image.fromarray(cropped).resize(self.image_resize, Image.BILINEAR),
            dtype=np.uint8,
        )

    def _ensure_rgb_channels(self, frame: np.ndarray) -> np.ndarray:
        """Ensure inference sees RGB frames even when the camera is mono."""
        if frame.ndim == 2:
            return np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim == 3:
            if frame.shape[-1] == 1:
                return np.repeat(frame, 3, axis=-1)
            if frame.shape[-1] > 3:
                return frame[..., :3]
            if frame.shape[0] == 1 and frame.shape[-1] != 3:
                base = frame[0]
                return np.repeat(base[..., None], 3, axis=-1)
        return frame

    def _compute_frame_params(self, frame_shape: Tuple[int, int]) -> Optional[Dict[str, float]]:
        """Return crop offsets and scale factors that mirror DatasetBuilder2."""
        if not frame_shape or len(frame_shape) < 2:
            return None
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if h <= 0 or w <= 0:
            return None
        if self.crop_size >= 1.0:
            crop_h = h
            crop_w = w
        else:
            crop_h = max(int(h * self.crop_size), 1)
            crop_w = max(int(w * self.crop_size), 1)
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        offset_y = max((h - crop_h) // 2, 0)
        offset_x = max((w - crop_w) // 2, 0)
        resize_w, resize_h = self.image_resize
        scale_x = resize_w / crop_w if crop_w else 1.0
        scale_y = resize_h / crop_h if crop_h else 1.0
        return {
            "crop_h": float(crop_h),
            "crop_w": float(crop_w),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
        }

    def _scale_pipette_for_model(
        self,
        pipette_positions: Optional[np.ndarray],
        frame_params: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        """Apply crop/resize scaling to planar pipette coordinates."""
        if pipette_positions is None:
            return None
        scaled = np.asarray(pipette_positions, dtype=np.float32).copy()
        if frame_params is None or scaled.size == 0:
            return scaled
        if scaled.ndim == 0:
            scaled = scaled.reshape(1)
        if scaled.shape[-1] >= 2:
            scale_x = frame_params.get("scale_x", 1.0)
            scale_y = frame_params.get("scale_y", 1.0)
            offset_x = frame_params.get("offset_x", 0.0)
            offset_y = frame_params.get("offset_y", 0.0)
            scaled[..., 0] = (scaled[..., 0] - offset_x) * scale_x
            scaled[..., 1] = (scaled[..., 1] - offset_y) * scale_y
        return scaled

    def _restore_pipette_from_model(
        self,
        pipette_components: np.ndarray,
        frame_params: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Undo crop/resize scaling on planar pipette coordinates."""
        restored = np.asarray(pipette_components, dtype=np.float32).copy()
        if frame_params is None or restored.size == 0:
            return restored
        if restored.ndim == 0:
            restored = restored.reshape(1)
        if restored.shape[-1] >= 2:
            scale_x = frame_params.get("scale_x", 1.0)
            scale_y = frame_params.get("scale_y", 1.0)
            offset_x = frame_params.get("offset_x", 0.0)
            offset_y = frame_params.get("offset_y", 0.0)
            if scale_x != 0:
                restored[..., 0] = restored[..., 0] / scale_x + offset_x
            if scale_y != 0:
                restored[..., 1] = restored[..., 1] / scale_y + offset_y
        return restored

    def _get_obs_stats(self, key: str):
        stats = self.obs_norm_object.get(key) if self.obs_norm_object else None
        if not stats:
            return None, None
        offset = stats.get("offset")
        scale = stats.get("scale")
        if offset is None or scale is None:
            return None, None
        return offset.astype(np.float32), scale.astype(np.float32)

    def _normalize_low_dim(self, key: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        return arr

    def _prepare_camera_payload(self, frame: np.ndarray) -> np.ndarray:
        frame_f = np.asarray(frame, dtype=np.float32)
        if frame_f.ndim < 3:
            raise ValueError("camera image must be at least 3D")
        if self.image_layout == "CHW" and frame_f.ndim == 3:
            frame_f = frame_f.transpose(2, 0, 1)
        return frame_f.astype(np.float32, copy=False)

    def _apply_obs_normalization(self, observation: dict) -> dict:
        """Cast per-key observations to float32 without applying normalization."""
        if not observation:
            return observation
        normalized = {}
        for key, value in observation.items():
            normalized[key] = np.asarray(value, dtype=np.float32)
        return normalized

    def action_unnorm(self, action: np.ndarray) -> np.ndarray:
        """Return the model action as float32 without undoing normalization."""
        arr = np.asarray(action, dtype=np.float32)
        return arr.astype(np.float32)

    def process_obs(self, observation, is_demo: bool = False):
        """Package a single observation for the model inputs."""
        cvpi, stage, image, resistance = observation
        obs_values: Dict[str, np.ndarray] = {}

        # debug_save_dir = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\testing")
        debug_save_dir = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\agent_movement_data")
        debug_timestamp: Optional[str] = None
        run_dir = debug_save_dir / self._debug_run_uid

        frame_params: Optional[Dict[str, float]] = None
        prepared_image: Optional[np.ndarray]
        image_arr = None if image is None else np.asarray(image)
        if not is_demo and image_arr is not None:
            frame_params = self._compute_frame_params(image_arr.shape[:2])
            prepared_image = self._prepare_image(image_arr, frame_params) if frame_params else image_arr
            self._last_frame_params = frame_params
        else:
            prepared_image = image_arr
            self._last_frame_params = None

        pipette_array: Optional[np.ndarray] = None
        if cvpi is not None:
            pipette_array = np.asarray(cvpi, dtype=np.float32)
            if not is_demo and frame_params is not None:
                print(f"pipette_array before scaling: {pipette_array}")
                pipette_array = self._scale_pipette_for_model(pipette_array, frame_params)
                print(f"pipette_array after scaling: {pipette_array}")
        if pipette_array is not None and pipette_array.ndim >= 1 and pipette_array.shape[-1] > 0:
            self._pipette_action_dim = int(pipette_array.shape[-1])

        if "camera_image" in self.obs_keys:
            if prepared_image is None:
                raise ValueError("camera image is required but missing from observation")
            frame_arr = np.asarray(prepared_image, dtype=np.float32)
            frame_arr = self._ensure_rgb_channels(frame_arr)
            save_frame = np.clip(frame_arr, 0, 255).astype(np.uint8)
            # Save processed camera frame for debugging.
            try:
                debug_save_dir.mkdir(parents=True, exist_ok=True)
                if debug_timestamp is None:
                    debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_image = Image.fromarray(save_frame)
                overlay_points = []
                if pipette_array is not None:
                    pipette_debug = np.asarray(pipette_array, dtype=np.float32)
                    if pipette_debug.size > 0:
                        if pipette_debug.ndim == 1:
                            pipette_debug = pipette_debug.reshape(1, -1)
                        elif pipette_debug.ndim > 2:
                            pipette_debug = pipette_debug.reshape(-1, pipette_debug.shape[-1])
                        if pipette_debug.ndim >= 2 and pipette_debug.shape[-1] >= 2:
                            height, width = save_frame.shape[:2]
                            max_x = max(width - 1.0, 0.0)
                            max_y = max(height - 1.0, 0.0)
                            for point in pipette_debug:
                                if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
                                    continue
                                x = float(np.clip(point[0], 0.0, max_x))
                                y = float(np.clip(point[1], 0.0, max_y))
                                overlay_points.append((x, y))
                if overlay_points:
                    draw = ImageDraw.Draw(debug_image)
                    radius = max(2, int(min(debug_image.size) * 0.02))
                    for x, y in overlay_points:
                        bbox = (x - radius, y - radius, x + radius, y + radius)
                        draw.ellipse(bbox, fill=(255, 0, 0), outline=(255, 255, 255))
                    if not is_demo:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        debug_image.save(run_dir / f"camera_image_{debug_timestamp}.png")
            except Exception:
                pass
            if self.requires_preprocessing:
                if self.image_layout == "CHW" and frame_arr.ndim == 3:
                    frame_arr = frame_arr.transpose(2, 0, 1)
            obs_values["camera_image"] = frame_arr
        if "pipette_positions" in self.obs_keys:
            if pipette_array is None:
                raise ValueError("pipette positions are required but missing from observation")
            pipette_values = np.asarray(pipette_array, dtype=np.float32)
            obs_values["pipette_positions"] = pipette_values

        if "stage_positions" in self.obs_keys:
            obs_values["stage_positions"] = np.asarray(stage, dtype=np.float32)

        if "resistance" in self.obs_keys:
            obs_values["resistance"] = np.asarray(resistance, dtype=np.float32)

        goal_values: Dict[str, np.ndarray] = {}
        if self.goal_required and self.goal_keys:
            fallback_goal = pipette_array if pipette_array is not None else cvpi
            goal_source = self.get_goal() if self.get_goal() is not None else fallback_goal
            for key in self.goal_keys:
                raw_value = goal_source
                if isinstance(goal_source, dict):
                    raw_value = goal_source.get(key, goal_source)
                value_arr = np.asarray(raw_value, dtype=np.float32)
                if key == "camera_image":
                    if value_arr.ndim < 3:
                        raise ValueError("Goal for camera_image requires image data")
                    if self.requires_preprocessing:
                        if self.image_layout == "CHW" and value_arr.ndim == 3:
                            value_arr = value_arr.transpose(2, 0, 1)
                goal_values[key] = value_arr
        payload = {}
        for name in self.input_names:
            shape = self.importer.input_shapes.get(name)
            value = None
            if name.startswith("obs::"):
                key = name.split("::", 1)[1]
                value = obs_values.get(key)
            elif name.startswith("goal::"):
                key = name.split("::", 1)[1]
                value = goal_values.get(key)
            else:
                value = self.state_buffers.get(name)
            fallback = self.default_inputs.get(name, np.zeros((1,), dtype=np.float32))
            tensor = np.asarray(value if value is not None else fallback, dtype=np.float32)
            if shape:
                while tensor.ndim < len(shape):
                    tensor = np.expand_dims(tensor, axis=0)
            payload[name] = tensor.astype(np.float32, copy=False)
        return payload

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize and pad the model action."""
        arr = self.action_unnorm(action).reshape(-1).astype(np.float32, copy=False)
        if self.action_dim and arr.shape[0] < self.action_dim:
            padding = self.action_dim - arr.shape[0]
            if padding > 0:
                arr = np.pad(arr, (0, padding)).astype(np.float32, copy=False)
        if self._last_frame_params and self._pipette_action_dim:
            pip_dim = min(int(self._pipette_action_dim), arr.shape[0])
            if pip_dim > 0:
                pip_slice = slice(arr.shape[0] - pip_dim, arr.shape[0])
                pipette_components = np.asarray(arr[pip_slice], dtype=np.float32)
                before_str = np.array2string(pipette_components, precision=7, separator=", ")
                print(
                    f"[process_action] pipette components (model space): {before_str} "
                    f"with frame params {self._last_frame_params}"
                )
                restored = self._restore_pipette_from_model(pipette_components, self._last_frame_params)
                after_str = np.array2string(restored, precision=7, separator=", ")
                print(f"[process_action] pipette components (image space): {after_str}")
                arr[pip_slice] = restored.reshape(pip_dim)
        return arr.astype(np.float32)

    def set_goal(self, goal: float) -> None:
        """Store the latest goal for goal-conditioned policies."""
        self.goal = goal if self.goal_required and goal is not None else None

    def get_goal(self):
        """Return the currently stored goal."""
        return self.goal if self.goal_required else None

    def predict(self, payload: dict):
        """Execute the ONNX session and refresh recurrent state."""
        outputs = self.session.run(self.output_names, payload)
        action = outputs[0]
        new_states = outputs[1:]
        for name, value in zip(self.state_names, new_states):
            self.state_buffers[name] = value.astype(np.float32)
        if new_states:
            self.internal_states = [self.state_buffers[name] for name in self.state_names]
        return action, new_states

    def inference(self, observation, goal=None, is_demo: bool = False):
        """Run the full observation?action pipeline for one step."""
        inputs = self.process_obs(observation, is_demo=is_demo)
        self.set_goal(goal)
        action, _ = self.predict(inputs)
        print(f"[inference] raw action from model: {action}")
        processed_action = self.process_action(action)
        return processed_action


class DemoReplayAgent:
    def __init__(self, actions: Optional[np.ndarray] = None) -> None:
        """Replay a cached sequence of actions without using live observations."""
        self._actions: Optional[np.ndarray] = None
        self._cursor: int = 0
        self._image_size: Optional[Tuple[int, int]] = None  # (height, width)
        if actions is not None:
            self.load_actions(actions)

    def load_actions(self, actions: np.ndarray) -> None:
        """Store the sequence of actions that will be returned on subsequent calls."""
        replay = np.asarray(actions, dtype=np.float32)
        if replay.ndim == 1:
            replay = replay.reshape(1, -1)
        if replay.size == 0:
            raise ValueError("DemoReplayAgent received an empty action array")
        self._actions = replay.astype(np.float32, copy=False)
        self._cursor = 0

    def set_image_size(self, image_shape: Tuple[int, int]) -> None:
        """Provide the target image height/width for scaling pixel-level actions."""
        if image_shape is None or len(image_shape) != 2:
            raise ValueError("Image shape must be a (height, width) tuple")
        self._image_size = (int(image_shape[0]), int(image_shape[1]))

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized (85x85) pixel deltas up to the full image resolution."""
        if self._image_size is None:
            return action.astype(np.float32, copy=False)
        height, width = self._image_size
        print(f"Image dimensions: {width}x{height} px")
        scaled = np.asarray(action, dtype=np.float32).copy()
        if scaled.size >= 2:
            scaled[0] = scaled[0] * (1280 / 85.0)
            scaled[1] = scaled[1] * (1280 / 85.0)
        if scaled.size >= 4:
            scaled[2] = scaled[2] * (1280 / 85.0)
            scaled[3] = scaled[3] * (1280 / 85.0)
        return scaled.astype(np.float32, copy=False)

    def inference(self, observation, goal=None, is_demo: bool = False):
        """Return the next cached action, ignoring all inputs once initialized."""
        if self._actions is None:
            raise RuntimeError("DemoReplayAgent requires actions to be loaded before inference")
        index = min(self._cursor, self._actions.shape[0] - 1)
        action = self._actions[index]
        if self._cursor < self._actions.shape[0]:
            self._cursor += 1
        return self._scale_action(action)


class PipetteFinder(ModelInferencer):
    def __init__(self, model_path: str = r"patcherbot\deepLearning\patchModel\Agents\PipetteFinder"):
        """Initialize the pipette finder policy."""
        base = Path(model_path)
        # print(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 2


class CellHunter(ModelInferencer):
    def __init__(self, model_path: str = r"patcherbot\deepLearning\patchModel\Agents\CellHunter"):
        """Initialize the cell hunting policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 6


class GigaSealer(ModelInferencer):
    def __init__(self, model_path: str = r"patcherbot\deepLearning\patchModel\Agents\GigaSealer"):
        """Initialize the gigaseal acquisition policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 3


class Burglar(ModelInferencer):
    def __init__(self, model_path: str = r"patcherbot\deepLearning\patchModel\Agents\Burglar"):
        """Initialize the break-in policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 3
