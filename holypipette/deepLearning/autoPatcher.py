
import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque


def _default_providers():
    available = list(ort.get_available_providers())
    preferred_order = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DirectMLExecutionProvider",
    ]
    providers = [ep for ep in preferred_order if ep in available]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available or ["CPUExecutionProvider"]
    ordered = []
    seen = set()
    for ep in providers:
        if ep not in seen:
            ordered.append(ep)
            seen.add(ep)
    return ordered


class _ModelIO:
    @staticmethod
    def matches(name: str, logical: str) -> bool:
        return name == logical or name.startswith(f"{logical}.")

    @staticmethod
    def shape_from_desc(desc):
        shape = getattr(desc, "shape", None)
        if not shape:
            return None
        dims = []
        for dim in shape:
            try:
                dims.append(int(dim))
            except Exception:
                return None
        return tuple(dims)

    @staticmethod
    def dtype_from_desc(desc):
        dtype_str = getattr(desc, "type", None)
        mapping = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        return mapping.get(dtype_str, None)

    @staticmethod
    def infer_action_horizon(desc):
        if desc is None:
            return None
        shape = _ModelIO.shape_from_desc(desc)
        if shape and len(shape) >= 2:
            try:
                return int(shape[-2])
            except Exception:
                return None
        return None

    @staticmethod
    def expand_for_desc(desc, array):
        arr = np.asarray(array)
        target_rank = len(getattr(desc, "shape", ())) if desc else arr.ndim
        while arr.ndim < target_rank:
            arr = np.expand_dims(arr, 0)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def resize_for_desc(desc, img_hwc):
        if desc is None:
            return np.asarray(img_hwc)
        shape = getattr(desc, "shape", None)
        if not shape or len(shape) < 4:
            return np.asarray(img_hwc)
        try:
            height = int(shape[-3])
            width = int(shape[-2])
        except Exception:
            return np.asarray(img_hwc)
        arr = np.asarray(img_hwc)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{getattr(desc, 'name', 'image')}: expected HWC image, got {arr.shape}")
        try:
            import cv2
            arr = cv2.resize(arr, (width, height))
        except Exception:
            arr = np.asarray(Image.fromarray(arr).resize((width, height)))
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def ensure_three_channel(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return arr

    @staticmethod
    def center_crop(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[top:top + new_h, left:left + new_w]

    @staticmethod
    def prepare_image(image, img_size: int, center_crop: bool) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.asarray(image)
        else:
            arr = np.asarray(image)
        arr = _ModelIO.ensure_three_channel(arr)
        if center_crop:
            arr = _ModelIO.center_crop(arr)
        try:
            import cv2
            arr = cv2.resize(arr, (img_size, img_size))
        except Exception:
            arr = np.asarray(Image.fromarray(arr).resize((img_size, img_size)))
        arr = arr.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))
    @staticmethod
    def coerce_stage(stage):
        s = np.asarray(stage, np.float32).reshape(-1)
        if s.shape[0] == 2:
            s = np.concatenate([s, [0.0]]).astype(np.float32)
        else:
            s = s[:3].astype(np.float32)
        return s

    @staticmethod
    def default_state_value(name, desc, num_layers, hidden_size, action_horizon):
        shape = _ModelIO.shape_from_desc(desc)
        dtype = _ModelIO.dtype_from_desc(desc) or np.float32
        if shape is None:
            if name.startswith("h0") or name.startswith("c0"):
                shape = (num_layers, 1, hidden_size)
            elif name == "cache_index":
                shape = (1,)
            else:
                shape = (1,)
        if name == "cache_index":
            fill = action_horizon or (shape[-1] if shape else 1)
            return np.full(shape, fill, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def copy_value(value):
        if value is None:
            return None
        return np.array(value, copy=True)

    @staticmethod
    def coerce_state_value(name, value, template):
        if value is None:
            return None
        arr = np.asarray(value)
        if template is not None:
            if arr.shape != template.shape:
                try:
                    arr = arr.reshape(template.shape)
                except Exception as exc:
                    raise ValueError(
                        f"{name}: incompatible state shape {arr.shape}, expected {template.shape}"
                    ) from exc
            if arr.dtype != template.dtype:
                arr = arr.astype(template.dtype)
        return np.array(arr, copy=True)


class _StateManager:
    def __init__(self, num_layers: int, hidden_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_desc = {}
        self.input_names = ()
        self.output_names = ()
        self.state_inputs = []
        self.output_map = {}
        self.templates = {}
        self.buffers = {}
        self.wrapper_history = {}
        self.action_key = None
        self.action_horizon = None

    def configure(self, input_desc, input_names, output_names):
        self.input_desc = input_desc or {}
        self.input_names = tuple(input_names or ())
        self.output_names = tuple(output_names or ())
        self.state_inputs = []
        self.output_map = {}
        self.wrapper_history = {}
        outputs = list(self.output_names)

        for candidate in ("actions", "action"):
            match = next((name for name in outputs if _ModelIO.matches(name, candidate)), None)
            if match is not None:
                self.action_key = match
                break
        else:
            self.action_key = outputs[0] if outputs else None

        def _register(name):
            if name not in self.state_inputs:
                self.state_inputs.append(name)

        for in_name in self.input_names:
            if _ModelIO.matches(in_name, "h0"):
                _register(in_name)
            elif _ModelIO.matches(in_name, "c0"):
                _register(in_name)
            elif _ModelIO.matches(in_name, "cached_actions"):
                _register(in_name)
            elif _ModelIO.matches(in_name, "cache_index"):
                _register(in_name)
            elif any(in_name.startswith(prefix) for prefix in ("state_", "cache_", "past_", "key", "value")):
                _register(in_name)

        for out_name in outputs:
            if _ModelIO.matches(out_name, "h1"):
                target = next((nm for nm in self.state_inputs if _ModelIO.matches(nm, "h0")), None)
                if target is not None:
                    self.output_map[out_name] = target
                    continue
            if _ModelIO.matches(out_name, "c1"):
                target = next((nm for nm in self.state_inputs if _ModelIO.matches(nm, "c0")), None)
                if target is not None:
                    self.output_map[out_name] = target
                    continue
            mapped = False
            for in_name in self.state_inputs:
                base = in_name
                if base.endswith("0"):
                    guess = base[:-1] + "1"
                    if _ModelIO.matches(out_name, guess):
                        self.output_map[out_name] = in_name
                        mapped = True
                        break
                if _ModelIO.matches(out_name, base):
                    self.output_map[out_name] = in_name
                    mapped = True
                    break
            if mapped:
                continue

        action_desc = self.input_desc.get("cached_actions") or self.input_desc.get(self.action_key)
        self.action_horizon = _ModelIO.infer_action_horizon(action_desc)

        self.templates = {}
        for name in self.state_inputs:
            desc = self.input_desc.get(name)
            template = _ModelIO.default_state_value(
                name, desc, self.num_layers, self.hidden_size, self.action_horizon
            )
            self.templates[name] = _ModelIO.copy_value(template)

        self.reset()
        return self
    def reset(self):
        self.buffers = {name: _ModelIO.copy_value(template) for name, template in self.templates.items()}
        self.wrapper_history = {}
        return self

    def prepare(self, feed_dict, *, h_override=None, c_override=None, input_names=None):
        feed = dict(feed_dict or {})
        if input_names is None:
            names = set(self.input_names)
        else:
            names = set(input_names)
        for name in list(self.state_inputs):
            if names and name not in names:
                continue
            value = None
            if _ModelIO.matches(name, "h0") and h_override is not None:
                value = h_override
            elif _ModelIO.matches(name, "c0") and c_override is not None:
                value = c_override
            else:
                value = self.buffers.get(name)
            if value is None:
                desc = self.input_desc.get(name)
                value = _ModelIO.default_state_value(
                    name, desc, self.num_layers, self.hidden_size, self.action_horizon
                )
            template = self.templates.get(name)
            coerced = _ModelIO.coerce_state_value(name, value, template)
            self.buffers[name] = _ModelIO.copy_value(coerced)
            feed[name] = _ModelIO.copy_value(coerced)
        return feed, feed.get("h0"), feed.get("c0")

    def update(self, output_dict):
        for out_name, in_name in self.output_map.items():
            if out_name not in output_dict:
                continue
            template = self.templates.get(in_name)
            self.buffers[in_name] = _ModelIO.coerce_state_value(
                in_name, output_dict[out_name], template
            )

    def get(self, logical):
        for name, value in self.buffers.items():
            if _ModelIO.matches(name, logical):
                return value
        return None

    def snapshot(self):
        return {name: _ModelIO.copy_value(value) for name, value in self.buffers.items()}

    def restore(self, state_dict):
        if not state_dict:
            return
        for name, value in state_dict.items():
            if name in self.templates:
                template = self.templates[name]
                self.buffers[name] = _ModelIO.coerce_state_value(name, value, template)

    def sequence(self, input_name, array):
        desc = self.input_desc.get(input_name)
        expanded = _ModelIO.expand_for_desc(desc, array)
        if desc is None:
            return expanded
        shape = getattr(desc, "shape", None)
        if not shape or len(shape) < 3:
            return expanded
        try:
            horizon = int(shape[1])
        except Exception:
            horizon = None
        if not horizon or horizon <= 1:
            return expanded
        history = self.wrapper_history.setdefault(input_name, deque(maxlen=horizon))
        if expanded.ndim >= 3:
            frame = np.array(expanded[0, -1], copy=True)
        elif expanded.ndim == 2:
            frame = np.array(expanded[0], copy=True)
        else:
            frame = np.array(expanded, copy=True)
        history.append(frame)
        while len(history) < horizon:
            history.appendleft(np.array(history[0], copy=True))
        stack = np.stack(list(history), axis=0).astype(
            expanded.dtype if expanded.dtype != np.float64 else np.float32, copy=False
        )
        return stack[None]


class AutoPatcher:
    """Base class for auto-patching policies."""

    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400):
        self.session = None
        self.input_names = None
        self.output_names = None
        self._input_desc = {}
        self.seq_len = None
        self.img_size = None
        self.prefill_init = False
        self.center_crop = True
        self._prefilled = False
        self._model_info = None

        self.state = _StateManager(num_layers=num_layers, hidden_size=hidden_size)

        if onnx_path is not None:
            self.load_model(onnx_path, providers)

    def load_model(self, onnx_path=None, providers=None):
        if providers is None:
            providers = _default_providers()
        if onnx_path is None or not Path(onnx_path).exists():
            model_dir = Path(__file__).parent / "patchModel/models"
            try:
                onnx_path = next(model_dir.glob("*.onnx"))
            except StopIteration:
                raise FileNotFoundError(f"No .onnx model found in {model_dir}")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [i.name for i in inputs]
        self.output_names = [o.name for o in outputs]
        self._input_desc = {i.name: i for i in inputs}
        self._model_info = None
        print(
            f"Loaded model {onnx_path} using providers {providers} with inputs {self.input_names} -> outputs {self.output_names}"
        )
        self.state.configure(self._input_desc, self.input_names, self.output_names)
        if self.state.action_key is None:
            raise RuntimeError("Could not determine action output key for ONNX model")
        return self.session, self.input_names, self.output_names

    def identify_model(self):
        """Cache and report basic metadata about the currently loaded ONNX model."""
        if self.input_names is None:
            raise RuntimeError("Model is not loaded; call load_model before identify_model.")
        if self._model_info is not None:
            return self._model_info
        uses_wrapper = any(name.startswith("obs::") for name in self.input_names)
        has_goal = any(name.startswith("goal::") for name in self.input_names)
        info = {
            "uses_wrapper": uses_wrapper,
            "has_goal": has_goal,
            "input_names": tuple(self.input_names),
            "output_names": tuple(self.output_names) if self.output_names else (),
            "providers": tuple(self.session.get_providers()) if self.session else (),
        }
        action_desc = self._input_desc.get(self.state.action_key)
        info["action_shape"] = _ModelIO.shape_from_desc(action_desc)
        info["action_dtype"] = getattr(action_desc, "type", None) if action_desc is not None else None
        self._model_info = info
        return info

    def inference(self, inputs=None, h0=None, c0=None):
        feed_dict = self._prepare_inputs(inputs, h0, c0)
        feed_dict, _, _ = self.state.prepare(feed_dict, h_override=h0, c_override=c0)
        filtered_inputs = {k: v for k, v in feed_dict.items() if k in (self.input_names or [])}
        try:
            outputs = self.session.run(None, filtered_inputs)
        except Exception:
            print("[ERROR] onnxruntime session.run failed. Provided inputs summary:")
            try:
                for k, v in filtered_inputs.items():
                    a = np.asarray(v)
                    amin = a.min() if a.size else "NA"
                    amax = a.max() if a.size else "NA"
                    print(f"  {k}: shape={a.shape}, dtype={a.dtype}, min={amin}, max={amax}")
            except Exception:
                print("  (could not print input summaries)")
            raise
        output_dict = dict(zip(self.output_names, outputs))
        self.state.update(output_dict)
        actions = output_dict.get(self.state.action_key)
        if actions is None:
            raise RuntimeError(f"Action output '{self.state.action_key}' not produced by model")
        actions = np.asarray(actions)
        if actions.ndim == 3:
            action = actions[:, -1, :]
        elif actions.ndim == 2:
            action = actions
        else:
            action = actions.reshape(1, -1)
        new_h0 = self.state.get("h0")
        new_c0 = self.state.get("c0")
        return np.asarray(action), new_h0, new_c0

    def reset_state(self):
        self.state.reset()
        self._prefilled = False

    def get_state_snapshot(self):
        return self.state.snapshot()

    def set_state_snapshot(self, state_dict):
        self.state.restore(state_dict)
    def prepare_image(self, image) -> np.ndarray:
        if self.img_size is None:
            raise AttributeError("img_size must be set before calling prepare_image")
        return _ModelIO.prepare_image(image, self.img_size, self.center_crop)

    def _prepare_inputs(self, inputs, h0, c0):
        raise NotImplementedError("Subclasses must implement `_prepare_inputs` to supply model inputs.")

    def _matches_state_name(self, name, logical):
        return _ModelIO.matches(name, logical)

    def _configure_state_management(self):
        self.state.configure(self._input_desc, self.input_names, self.output_names)

    def _initialize_state_buffers(self):
        self.state.reset()

    def _ensure_state_inputs(self, inputs, h0, c0, *, input_names=None):
        return self.state.prepare(inputs, h_override=h0, c_override=c0, input_names=input_names)

    def _update_state_from_outputs(self, output_dict):
        self.state.update(output_dict)

    def _shape_from_desc(self, desc):
        return _ModelIO.shape_from_desc(desc)

    def _dtype_from_desc(self, desc):
        return _ModelIO.dtype_from_desc(desc)

    def _infer_action_horizon(self, desc=None):
        if desc is not None:
            return _ModelIO.infer_action_horizon(desc)
        return self.state.action_horizon

    def _default_state_value(self, name):
        desc = self._input_desc.get(name)
        return _ModelIO.default_state_value(
            name, desc, self.state.num_layers, self.state.hidden_size, self.state.action_horizon
        )

    def _copy_state_value(self, value):
        return _ModelIO.copy_value(value)

    def _coerce_state_value(self, name, value):
        template = self.state.templates.get(name)
        return _ModelIO.coerce_state_value(name, value, template)

    def _get_state_value(self, logical):
        return self.state.get(logical)

    def _get_input_desc(self):
        return self._input_desc

    def _resize_to_expected(self, input_name, img_hwc):
        return _ModelIO.resize_for_desc(self._input_desc.get(input_name), img_hwc)

    def _expand_for_input(self, input_name, array):
        return _ModelIO.expand_for_desc(self._input_desc.get(input_name), array)

    def _coerce_stage(self, stage):
        return _ModelIO.coerce_stage(stage)

    def _crop_center(self, arr: np.ndarray) -> np.ndarray:
        return _ModelIO.center_crop(arr)

    def _prepare_wrapper_sequence(self, input_name, array):
        return self.state.sequence(input_name, array)
class CellHunter(AutoPatcher):
    """
    Cell-hunting policy.

    Accepts a single live observation from `observe()` per call in the form
    (pipette, stage, image, resistance). Internally accumulates a 16-step
    history and maps it to the ONNX feeds.

    Also supports wrapper-style models that expose inputs as `obs::` and
    optionally `goal::` keys. In that case it switches to single-step raw HWC
    images and mirrors missing goal fields from the observation.
    """

    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400,
                 *, seq_len: int = 16, img_size: int = 85, prefill_init: bool = False,
                 center_crop: bool = True):
        super().__init__(onnx_path=onnx_path, providers=providers,
                         num_layers=num_layers, hidden_size=hidden_size)
        import collections
        self.seq_len = seq_len
        self.img_size = img_size
        self.prefill_init = prefill_init
        self.center_crop = center_crop
        self._prefilled = False

        self._img_q = collections.deque(maxlen=self.seq_len)
        self._pip_q = collections.deque(maxlen=self.seq_len)
        self._stage_q = collections.deque(maxlen=self.seq_len)
        self._res_q = collections.deque(maxlen=self.seq_len)

    def _prepare_inputs(self, model_input, h0, c0):
        in_desc = self._get_input_desc()
        input_names = set(in_desc.keys())
        uses_obs_prefix = any(n.startswith("obs::") for n in input_names)

        if uses_obs_prefix:
            if isinstance(model_input, dict):
                obs_pkg = model_input.get("obs", None)
                if obs_pkg is None and all(
                    k in model_input for k in ("pipette_positions", "stage_positions", "camera_image", "resistance")
                ):
                    obs_pkg = (
                        model_input["pipette_positions"],
                        model_input["stage_positions"],
                        model_input["camera_image"],
                        model_input["resistance"],
                    )
                goal_pkg = model_input.get("goal", None)
            else:
                obs_pkg = model_input
                goal_pkg = None

            if isinstance(obs_pkg, dict):
                pip = np.asarray(obs_pkg["pipette_positions"], np.float32).reshape(-1)
                stage = self._coerce_stage(obs_pkg["stage_positions"])
                img = np.asarray(obs_pkg["camera_image"])
                res = np.asarray(obs_pkg["resistance"], np.float32).reshape(-1)
            else:
                pip, stage, img, res = obs_pkg
                pip = np.asarray(pip, np.float32).reshape(-1)
                stage = self._coerce_stage(stage)
                img = np.asarray(img)
                res = np.asarray(res, np.float32).reshape(-1)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            inputs = {}
            for nm in input_names:
                if not nm.startswith("obs::"):
                    continue
                key = nm.split("obs::", 1)[1]
                if key == "camera_image":
                    img_r = self._resize_to_expected(nm, img)
                    inputs[nm] = self._prepare_wrapper_sequence(nm, img_r)
                elif key == "pipette_positions":
                    inputs[nm] = self._prepare_wrapper_sequence(nm, pip)
                elif key == "stage_positions":
                    inputs[nm] = self._prepare_wrapper_sequence(nm, stage)
                elif key == "resistance":
                    inputs[nm] = self._prepare_wrapper_sequence(nm, res)

            if any(nm.startswith("goal::") for nm in input_names):
                gpip, gstage, gimg, gres = pip, stage, img, res

                if goal_pkg is not None:
                    if isinstance(goal_pkg, dict):
                        if "pipette_positions" in goal_pkg:
                            gpip = np.asarray(goal_pkg["pipette_positions"], np.float32).reshape(-1)
                        if "stage_positions" in goal_pkg:
                            gstage = self._coerce_stage(goal_pkg["stage_positions"])
                        if "resistance" in goal_pkg:
                            gres = np.asarray(goal_pkg["resistance"], np.float32).reshape(-1)
                        if "camera_image" in goal_pkg and goal_pkg["camera_image"] is not None:
                            gimg = _ModelIO.ensure_three_channel(np.asarray(goal_pkg["camera_image"]))
                    else:
                        gpip, gstage, gimg, gres = goal_pkg
                        gpip = np.asarray(gpip, np.float32).reshape(-1)
                        gstage = self._coerce_stage(gstage)
                        gres = np.asarray(gres, np.float32).reshape(-1)
                        if gimg is None:
                            gimg = img
                        else:
                            gimg = _ModelIO.ensure_three_channel(np.asarray(gimg))

                for nm in input_names:
                    if not nm.startswith("goal::"):
                        continue
                    key = nm.split("goal::", 1)[1]
                    if key == "camera_image":
                        inputs[nm] = self._prepare_wrapper_sequence(
                            nm, self._resize_to_expected(nm, gimg)
                        )
                    elif key == "pipette_positions":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gpip)
                    elif key == "stage_positions":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gstage)
                    elif key == "resistance":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gres)

            inputs, h0, c0 = self._ensure_state_inputs(inputs, h0, c0, input_names=input_names)
            return inputs

        pip, stage, img, res = model_input
        pip = np.asarray(pip, dtype=np.float32).reshape(3)
        stage = self._coerce_stage(stage)
        res = np.float32(res)

        img = self.prepare_image(img)
        self._img_q.append(img)
        self._pip_q.append(pip)
        self._stage_q.append(stage)
        self._res_q.append(res)

        if self.prefill_init and not self._prefilled and len(self._img_q) == 1:
            for _ in range(self.seq_len - 1):
                self._img_q.append(self._img_q[0].copy())
                self._pip_q.append(self._pip_q[0].copy())
                self._stage_q.append(self._stage_q[0].copy())
                self._res_q.append(np.float32(self._res_q[0]))
            self._prefilled = True

        while len(self._img_q) < self.seq_len:
            self._img_q.append(self._img_q[-1].copy())
            self._pip_q.append(self._pip_q[-1].copy())
            self._stage_q.append(self._stage_q[-1].copy())
            self._res_q.append(np.float32(self._res_q[-1]))

        inputs = {
            "camera_image": np.stack(list(self._img_q), 0)[None],
            "pipette_positions": np.stack(list(self._pip_q), 0)[None],
            "stage_positions": np.stack(list(self._stage_q), 0)[None],
            "resistance": np.stack(list(self._res_q), 0).reshape(1, -1),
        }

        inputs, h0, c0 = self._ensure_state_inputs(inputs, h0, c0, input_names=input_names)
        return inputs
class GigaSealer(AutoPatcher):
    """Gigasealing policy."""

    def _prepare_inputs(self, inputs, h0, c0):
        try:
            img_q, pip_q, stage_q, res_q = inputs
        except Exception as exc:
            raise ValueError("GigaSealer expects a 4-tuple history: (img_q, pip_q, stage_q, res_q)") from exc
        feed = {
            "pressure": np.stack(res_q, 0).reshape(1, -1),
            "resistance": np.stack(res_q, 0).reshape(1, -1),
            "stage_positions": np.stack(stage_q, 0)[None],
            "pipette_positions": np.stack(pip_q, 0)[None],
        }
        feed, h0, c0 = self._ensure_state_inputs(feed, h0, c0)
        return feed


class Burglar(AutoPatcher):
    """Break-in policy."""

    def _prepare_inputs(self, inputs, h0, c0):
        try:
            img_q, pip_q, stage_q, res_q = inputs
        except Exception as exc:
            raise ValueError("Burglar expects a 4-tuple history: (img_q, pip_q, stage_q, res_q)") from exc
        feed = {
            "capacitance": np.stack(res_q, 0).reshape(1, -1),
            "stage_positions": np.stack(stage_q, 0)[None],
        }
        feed, h0, c0 = self._ensure_state_inputs(feed, h0, c0)
        return feed


class PipetteFinder(AutoPatcher):
    """Pipette finding policy."""

    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400,
                 *, seq_len: int = 16, img_size: int = 85, prefill_init: bool = False,
                 center_crop: bool = True):
        super().__init__(onnx_path=onnx_path, providers=providers,
                         num_layers=num_layers, hidden_size=hidden_size)
        import collections
        self.seq_len = seq_len
        self.img_size = img_size
        self.prefill_init = prefill_init
        self.center_crop = center_crop
        self._prefilled = False
        self._img_q = collections.deque(maxlen=self.seq_len)
        self._pip_q = collections.deque(maxlen=self.seq_len)
        self._stage_q = collections.deque(maxlen=self.seq_len)

    def _prepare_inputs(self, model_input, h0, c0):
        in_desc = self._get_input_desc()
        input_names = set(in_desc.keys())
        uses_obs_prefix = any(n.startswith("obs::") for n in input_names)

        def _unpack_obs(pkg, allow_image_none=False):
            if isinstance(pkg, dict):
                pip = np.asarray(pkg["pipette_positions"], np.float32).reshape(-1)
                stage = self._coerce_stage(pkg["stage_positions"])
                img = pkg.get("camera_image", None)
            else:
                try:
                    pip, stage, img = pkg[:3]
                except Exception as exc:
                    raise ValueError("PipetteFinder expects (pip, stage, image)") from exc
                pip = np.asarray(pip, np.float32).reshape(-1)
                stage = self._coerce_stage(stage)
            if img is None:
                if allow_image_none:
                    arr = None
                else:
                    raise ValueError("camera_image is required for PipetteFinder observation")
            else:
                arr = _ModelIO.ensure_three_channel(np.asarray(img))
            return pip, stage, arr

        if uses_obs_prefix:
            if isinstance(model_input, dict):
                obs_pkg = model_input.get("obs", None)
                if obs_pkg is None and all(
                    k in model_input for k in ("pipette_positions", "stage_positions", "camera_image")
                ):
                    obs_pkg = (
                        model_input["pipette_positions"],
                        model_input["stage_positions"],
                        model_input["camera_image"],
                    )
                goal_pkg = model_input.get("goal", None)
            else:
                obs_pkg = model_input
                goal_pkg = None

            pip, stage, img = _unpack_obs(obs_pkg)
            inputs = {}
            for nm in input_names:
                if not nm.startswith("obs::"):
                    continue
                key = nm.split("obs::", 1)[1]
                if key == "camera_image":
                    inputs[nm] = self._prepare_wrapper_sequence(
                        nm, _ModelIO.resize_for_desc(in_desc.get(nm), img)
                    )
                elif key == "pipette_positions":
                    inputs[nm] = self._prepare_wrapper_sequence(nm, pip)
                elif key == "stage_positions":
                    inputs[nm] = self._prepare_wrapper_sequence(nm, stage)

            if any(nm.startswith("goal::") for nm in input_names):
                gpip, gstage, gimg = pip, stage, img
                if goal_pkg is not None:
                    if isinstance(goal_pkg, dict):
                        if "pipette_positions" in goal_pkg:
                            gpip = np.asarray(goal_pkg["pipette_positions"], np.float32).reshape(-1)
                        if "stage_positions" in goal_pkg:
                            gstage = self._coerce_stage(goal_pkg["stage_positions"])
                        if "camera_image" in goal_pkg and goal_pkg["camera_image"] is not None:
                            gimg = _ModelIO.ensure_three_channel(np.asarray(goal_pkg["camera_image"]))
                    else:
                        seq = list(goal_pkg)
                        if len(seq) >= 1 and seq[0] is not None:
                            gpip = np.asarray(seq[0], np.float32).reshape(-1)
                        if len(seq) >= 2 and seq[1] is not None:
                            gstage = self._coerce_stage(seq[1])
                        if len(seq) >= 3 and seq[2] is not None:
                            gimg = _ModelIO.ensure_three_channel(np.asarray(seq[2]))
                for nm in input_names:
                    if not nm.startswith("goal::"):
                        continue
                    key = nm.split("goal::", 1)[1]
                    if key == "camera_image":
                        inputs[nm] = self._prepare_wrapper_sequence(
                            nm, _ModelIO.resize_for_desc(in_desc.get(nm), gimg)
                        )
                    elif key == "pipette_positions":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gpip)
                    elif key == "stage_positions":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gstage)

            inputs, h0, c0 = self._ensure_state_inputs(inputs, h0, c0, input_names=input_names)
            return inputs

        if isinstance(model_input, dict):
            if not all(k in model_input for k in ("pipette_positions", "stage_positions", "camera_image")):
                raise ValueError("PipetteFinder expects keys pipette_positions, stage_positions, camera_image")
            pip = np.asarray(model_input["pipette_positions"], np.float32).reshape(-1)
            stage = self._coerce_stage(model_input["stage_positions"])
            img = np.asarray(model_input["camera_image"])
        else:
            if len(model_input) < 3:
                raise ValueError("PipetteFinder expects (pip, stage, image)")
            pip, stage, img = model_input[:3]
            pip = np.asarray(pip, np.float32).reshape(-1)
            stage = self._coerce_stage(stage)
            img = np.asarray(img)

        img = self.prepare_image(img)
        self._img_q.append(img)
        self._pip_q.append(pip)
        self._stage_q.append(stage)

        if self.prefill_init and not self._prefilled and len(self._img_q) == 1:
            for _ in range(self.seq_len - 1):
                self._img_q.append(self._img_q[0].copy())
                self._pip_q.append(self._pip_q[0].copy())
                self._stage_q.append(self._stage_q[0].copy())
            self._prefilled = True

        while len(self._img_q) < self.seq_len:
            self._img_q.append(self._img_q[-1].copy())
            self._pip_q.append(self._pip_q[-1].copy())
            self._stage_q.append(self._stage_q[-1].copy())

        inputs = {}
        if "camera_image" in input_names:
            inputs["camera_image"] = np.stack(list(self._img_q), 0)[None]
        if "pipette_positions" in input_names:
            inputs["pipette_positions"] = np.stack(list(self._pip_q), 0)[None]
        if "stage_positions" in input_names:
            inputs["stage_positions"] = np.stack(list(self._stage_q), 0)[None]
        if "pipette_position" in input_names and "pipette_position" not in inputs:
            inputs["pipette_position"] = np.stack(list(self._pip_q), 0)[None]
        if "stage_position" in input_names and "stage_position" not in inputs:
            inputs["stage_position"] = np.stack(list(self._stage_q), 0)[None]

        inputs, h0, c0 = self._ensure_state_inputs(inputs, h0, c0, input_names=input_names)
        return inputs
