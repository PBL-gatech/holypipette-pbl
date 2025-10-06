# necessary imports
import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque


class ModelImporter:
    def __init__(self, onnx_model_path: str, input_normalization_npz_path: str,
                 output_normalization_npz_path: str, model_desc_json_path: str) -> None:
        """Store file system handles for a single exported policy."""
        self.onnx_path = Path(onnx_model_path)
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
        self.crop_size = 0.5
        self.image_resize = (85, 85)
        self.action_unnorm_object = None
        self.obs_norm_object = None
        self.goal_required = False
        self.goal = None
        self.internal_states = [0, 0]

        self.importer = model_importer.load()
        self.session = self.importer.session
        self.obs_norm_object = self.importer.obs_stats
        self.action_unnorm_object = self.importer.action_stats
        self.metadata = self.importer.metadata
        self.obs_keys = list(self.metadata.get("observation_keys", []))
        self.goal_keys = list(self.metadata.get("goal_keys", []))
        self.goal_required = bool(self.goal_keys)
        self.input_names = list(self.importer.input_names)
        self.output_names = list(self.importer.output_names)
        self.image_layout = "CHW"
        camera_shape = self.importer.input_shapes.get("obs::camera_image")
        if camera_shape and 3 in camera_shape and camera_shape.index(3) == len(camera_shape) - 1:
            self.image_layout = "HWC"
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

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Center-crop and resize an RGB frame from the microscope."""
        frame = np.asarray(image, dtype=np.uint8)
        h, w = frame.shape[:2]
        crop_h, crop_w = int(h * self.crop_size), int(w * self.crop_size)
        y0, x0 = max((h - crop_h) // 2, 0), max((w - crop_w) // 2, 0)
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        return np.array(Image.fromarray(cropped).resize(self.image_resize, Image.BILINEAR))

    def obs_norm(self, observation: dict) -> dict:
        """Apply saved observation statistics to a per-key dictionary."""
        if not observation or not self.obs_norm_object:
            return observation
        normed = {}
        for key, value in observation.items():
            stats = self.obs_norm_object.get(key)
            arr = np.asarray(value, dtype=np.float32)
            if stats and "offset" in stats and "scale" in stats:
                offset = stats["offset"]
                scale = stats["scale"]
                while arr.ndim < offset.ndim:
                    arr = np.expand_dims(arr, axis=0)
                normed[key] = (arr - offset) / (scale + 1e-6)
            else:
                normed[key] = arr
        return normed

    def action_unnorm(self, action: np.ndarray) -> np.ndarray:
        """Undo action normalization before sending commands downstream."""
        arr = np.asarray(action, dtype=np.float32)
        if not self.action_unnorm_object:
            return arr
        offset = self.action_unnorm_object.get("offset")
        scale = self.action_unnorm_object.get("scale")
        if offset is not None and scale is not None:
            while arr.ndim < offset.ndim:
                arr = np.expand_dims(arr, axis=0)
            arr = arr * scale + offset
            arr = np.squeeze(arr, axis=0)
        return arr.astype(np.float32)

    def process_obs(self, observation, is_demo: bool = False):
        """Package a single observation for the model inputs."""
        cvpi, stage, image, resistance = observation
        obs_map = {}
        if "camera_image" in self.obs_keys:
            frame = image if is_demo else self._prepare_image(image)
            frame = frame.astype(np.float32)
            if self.image_layout == "CHW" and frame.ndim == 3:
                frame = frame.transpose(2, 0, 1)
            obs_map["camera_image"] = frame
        if "pipette_positions" in self.obs_keys:
            obs_map["pipette_positions"] = np.asarray(cvpi, dtype=np.float32)
        if "stage_positions" in self.obs_keys:
            obs_map["stage_positions"] = np.asarray(stage, dtype=np.float32)
        if "resistance" in self.obs_keys:
            obs_map["resistance"] = np.asarray(resistance, dtype=np.float32)
        obs_map = self.obs_norm(obs_map)
        goal_map = {}
        if self.goal_required and self.goal_keys:
            target = self.goal if self.goal is not None else cvpi
            goal_map[self.goal_keys[0]] = np.asarray(target, dtype=np.float32)
            goal_map = self.obs_norm(goal_map)
        payload = {}
        for name in self.input_names:
            shape = self.importer.input_shapes.get(name)
            value = None
            if name.startswith("obs::"):
                key = name.split("::", 1)[1]
                value = obs_map.get(key)
            elif name.startswith("goal::"):
                key = name.split("::", 1)[1]
                value = goal_map.get(key)
            else:
                value = self.state_buffers.get(name)
            fallback = self.default_inputs.get(name, np.zeros((1,), dtype=np.float32))
            tensor = np.asarray(value if value is not None else fallback, dtype=np.float32)
            if shape:
                while tensor.ndim < len(shape):
                    tensor = np.expand_dims(tensor, axis=0)
            payload[name] = tensor.astype(np.float32)
        return payload

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize and pad the model action."""
        arr = self.action_unnorm(action).reshape(-1)
        if self.action_dim and arr.shape[0] < self.action_dim:
            arr = np.pad(arr, (0, self.action_dim - arr.shape[0]))
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
        processed_action = self.process_action(action)
        return processed_action


class PipetteFinder(ModelInferencer):
    def __init__(self, model_path: str = r"holypipette\deepLearning\patchModel\Agents\PipetteFinder"):
        """Initialize the pipette finder policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 6


class CellHunter(ModelInferencer):
    def __init__(self, model_path: str = r"holypipette\deepLearning\patchModel\Agents\CellHunter"):
        """Initialize the cell hunting policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 6


class GigaSealer(ModelInferencer):
    def __init__(self, model_path: str = r"holypipette\deepLearning\patchModel\Agents\GigaSealer"):
        """Initialize the gigaseal acquisition policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 3


class Burglar(ModelInferencer):
    def __init__(self, model_path: str = r"holypipette\deepLearning\patchModel\Agents\Burglar"):
        """Initialize the break-in policy."""
        base = Path(model_path)
        importer = ModelImporter(base, base, base, base)
        super().__init__(importer)
        self.action_dim = 3
