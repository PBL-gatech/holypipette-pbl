import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque


class AutoPatcher:
    """
    Base class for auto-patching policies.  Subclasses must implement `_prepare_inputs` to map
    inputs into ONNX feed dictionaries.
    """
    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400):
        self.session = None
        self.input_names = None
        self.output_names = None
        self._input_desc = {}
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = None
        self.img_size = None
        self.prefill_init = False
        self.center_crop = True
        self._prefilled = False

        # runtime state bookkeeping
        self._state_inputs = set()
        self._state_output_map = {}
        self._state_buffers = {}
        self._state_templates = {}
        self._action_output_key = "actions"
        self._uses_action_cache = False
        self._is_recurrent = False
        self._action_horizon = None
        self._wrapper_history = {}

        if onnx_path is not None:
            self.load_model(onnx_path, providers)

    def load_model(self, onnx_path=None, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
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
        print(f"Loaded model {onnx_path} with inputs {self.input_names} -> outputs {self.output_names}")
        self._configure_state_management()
        return self.session, self.input_names, self.output_names

    def inference(self, inputs=None, h0=None, c0=None):
        """
        Run inference using the mapping supplied by `_prepare_inputs`.
        Adds robust diagnostics: prints expected vs provided shapes/dtypes if ORT fails.
        """
        feed_dict = self._prepare_inputs(inputs, h0, c0)
        feed_dict, _, _ = self._ensure_state_inputs(feed_dict, h0, c0, input_names=set(self.input_names or []))

        filtered_inputs = {k: v for k, v in feed_dict.items() if k in self.input_names}

        try:
            in_desc = self._get_input_desc()
            for k, v in filtered_inputs.items():
                arr = np.asarray(v)
                desc = in_desc.get(k)
                shape_attr = getattr(desc, "shape", None) if desc is not None else None
                exp_shape = tuple(shape_attr) if shape_attr else None
                exp_type = getattr(desc, "type", None) if desc is not None else None
        except Exception:
            pass

        try:
            outputs = self.session.run(None, filtered_inputs)
        except Exception:
            print("[ERROR] onnxruntime session.run failed. Provided inputs summary:")
            try:
                for k, v in filtered_inputs.items():
                    a = np.asarray(v)
                    amin = a.min() if a.size else "NA"
                    amax = a.max() if a.size else "NA"
            except Exception:
                print("  (could not print input summaries)")
            raise

        output_dict = dict(zip(self.output_names, outputs))
        self._update_state_from_outputs(output_dict)

        if self._action_output_key is None:
            raise RuntimeError("Could not determine action output key for ONNX model")
        actions = output_dict.get(self._action_output_key)
        if actions is None:
            raise RuntimeError(f"Action output '{self._action_output_key}' not produced by model")
        actions = np.asarray(actions)
        if actions.ndim == 3:
            action = actions[:, -1, :]
        elif actions.ndim == 2:
            action = actions
        else:
            action = actions.reshape(1, -1)

        new_h0 = self._copy_state_value(self._get_state_value("h0"))
        new_c0 = self._copy_state_value(self._get_state_value("c0"))
        return np.asarray(action), new_h0, new_c0

    def reset_state(self):
        if getattr(self, "_state_templates", None):
            self._state_buffers = {k: self._copy_state_value(v) for k, v in self._state_templates.items()}
        else:
            self._state_buffers = {}
        self._prefilled = False
        self._wrapper_history = {}

    def get_state_snapshot(self):
        return {k: self._copy_state_value(v) for k, v in self._state_buffers.items()}

    def set_state_snapshot(self, state_dict):
        if not state_dict:
            return
        for name, value in state_dict.items():
            if name in self._state_templates:
                self._state_buffers[name] = self._coerce_state_value(name, value)

    def _matches_state_name(self, name, logical):
        return name == logical or name.startswith(f"{logical}.")

    def _get_state_value(self, logical):
        for name, value in self._state_buffers.items():
            if self._matches_state_name(name, logical):
                return value
        return None

    def _prepare_wrapper_sequence(self, input_name, array):
        desc = self._get_input_desc().get(input_name)
        expanded = self._expand_for_input(input_name, array)
        shape = self._shape_from_desc(desc)
        if not shape or len(shape) < 3:
            return expanded
        horizon = int(shape[1]) if shape[1] else 1
        if horizon <= 1:
            return expanded
        history = self._wrapper_history.setdefault(input_name, deque(maxlen=horizon))
        frame = expanded[0]
        if expanded.ndim >= 3:
            frame = frame[-1]
        frame = np.array(frame, copy=True)
        history.append(frame)
        while len(history) < horizon:
            history.appendleft(np.array(history[0], copy=True))
        stack = np.stack(list(history), axis=0)
        stack = stack.astype(expanded.dtype if expanded.dtype != np.float64 else np.float32, copy=False)
        return stack[None]

    def _configure_state_management(self):
        self._state_inputs = set()
        self._state_output_map = {}
        self._uses_action_cache = False
        self._is_recurrent = False
        self._action_horizon = None

        input_names = list(self.input_names or [])
        output_names = list(self.output_names or [])

        for candidate in ("actions", "action"):
            match = next((out_name for out_name in output_names if self._matches_state_name(out_name, candidate)), None)
            if match is not None:
                self._action_output_key = match
                break
        else:
            self._action_output_key = output_names[0] if output_names else None

        self._wrapper_history = {}

        for in_name in input_names:
            if self._matches_state_name(in_name, "h0"):
                self._state_inputs.add(in_name)
                for out_name in output_names:
                    if self._matches_state_name(out_name, "h1"):
                        self._state_output_map[out_name] = in_name
                self._is_recurrent = True
            elif self._matches_state_name(in_name, "c0"):
                self._state_inputs.add(in_name)
                for out_name in output_names:
                    if self._matches_state_name(out_name, "c1"):
                        self._state_output_map[out_name] = in_name
                self._is_recurrent = True
            elif self._matches_state_name(in_name, "cached_actions"):
                self._state_inputs.add(in_name)
                for out_name in output_names:
                    if self._matches_state_name(out_name, "action_cache"):
                        self._state_output_map[out_name] = in_name
                self._uses_action_cache = True
            elif self._matches_state_name(in_name, "cache_index"):
                self._state_inputs.add(in_name)
                for out_name in output_names:
                    if self._matches_state_name(out_name, "cache_index"):
                        self._state_output_map[out_name] = in_name
                self._uses_action_cache = True

        self._initialize_state_buffers()

    def _initialize_state_buffers(self):
        self._state_templates = {}
        self._state_buffers = {}
        self._action_horizon = self._infer_action_horizon()
        for name in self._state_inputs:
            template = self._default_state_value(name)
            self._state_templates[name] = self._copy_state_value(template)
            self._state_buffers[name] = self._copy_state_value(template)

    def _ensure_state_inputs(self, inputs, h0, c0, *, input_names=None):
        feed = dict(inputs or {})
        names = input_names or set(self.input_names or [])

        for in_name in list(names):
            if self._matches_state_name(in_name, "h0"):
                value = h0 if h0 is not None else self._state_buffers.get(in_name, self._state_buffers.get("h0"))
                if value is None:
                    value = self._default_state_value(in_name)
                value = self._coerce_state_value(in_name, value)
                self._state_buffers[in_name] = value
                feed[in_name] = self._copy_state_value(value)
            elif self._matches_state_name(in_name, "c0"):
                value = c0 if c0 is not None else self._state_buffers.get(in_name, self._state_buffers.get("c0"))
                if value is None:
                    value = self._default_state_value(in_name)
                value = self._coerce_state_value(in_name, value)
                self._state_buffers[in_name] = value
                feed[in_name] = self._copy_state_value(value)

        for name in self._state_inputs:
            if self._matches_state_name(name, "h0") or self._matches_state_name(name, "c0"):
                continue
            value = feed.get(name)
            if value is None:
                value = self._state_buffers.get(name)
            if value is None:
                value = self._default_state_value(name)
            value = self._coerce_state_value(name, value)
            self._state_buffers[name] = value
            feed[name] = self._copy_state_value(value)

        return feed, feed.get("h0"), feed.get("c0")

    def _update_state_from_outputs(self, output_dict):
        for out_name, in_name in self._state_output_map.items():
            if out_name not in output_dict:
                continue
            self._state_buffers[in_name] = self._coerce_state_value(in_name, output_dict[out_name])

    def _shape_from_desc(self, desc):
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

    def _dtype_from_desc(self, desc):
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

    def _infer_action_horizon(self, desc=None):
        if desc is None:
            desc = self._get_input_desc().get("cached_actions")
        if desc is None:
            return None
        shape = self._shape_from_desc(desc)
        if shape and len(shape) >= 2:
            try:
                return int(shape[-2])
            except Exception:
                return None
        return None

    def _default_state_value(self, name):
        desc = self._get_input_desc().get(name)
        shape = self._shape_from_desc(desc)
        dtype = self._dtype_from_desc(desc) or np.float32
        if shape is None:
            if name in ("h0", "c0"):
                shape = (self.num_layers, 1, self.hidden_size)
            elif name == "cache_index":
                shape = (1,)
            else:
                shape = (1,)
        if name == "cache_index":
            fill = self._action_horizon or (shape[-1] if shape else 1)
            return np.full(shape, fill, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _copy_state_value(self, value):
        if value is None:
            return None
        return np.array(value, copy=True)

    def _coerce_state_value(self, name, value):
        if value is None:
            return None
        arr = np.asarray(value)
        template = self._state_templates.get(name)
        if template is not None:
            if arr.shape != template.shape:
                try:
                    arr = arr.reshape(template.shape)
                except Exception as exc:
                    raise ValueError(f"{name}: incompatible state shape {arr.shape}, expected {template.shape}") from exc
            if arr.dtype != template.dtype:
                arr = arr.astype(template.dtype)
        return np.array(arr, copy=True)

    def _get_input_desc(self):
        return getattr(self, "_input_desc", {}) or {}

    def _resize_to_expected(self, input_name, img_hwc):
        desc = self._get_input_desc().get(input_name)
        if desc is None:
            return np.asarray(img_hwc)
        shape = getattr(desc, "shape", None)
        if not shape or len(shape) < 4:
            return np.asarray(img_hwc)
        try:
            H = int(shape[-3])
            W = int(shape[-2])
        except Exception:
            return np.asarray(img_hwc)
        arr = np.asarray(img_hwc)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{input_name}: expected HWC image, got {arr.shape}")
        try:
            import cv2
            arr = cv2.resize(arr, (W, H))
        except Exception:
            arr = np.asarray(Image.fromarray(arr).resize((W, H)))
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def _expand_for_input(self, input_name, array):
        desc = self._get_input_desc().get(input_name)
        a = np.asarray(array)
        target_rank = len(getattr(desc, "shape", ())) if desc else a.ndim
        while a.ndim < target_rank:
            a = np.expand_dims(a, 0)
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        return a

    def _coerce_stage(self, stage):
        s = np.asarray(stage, np.float32).reshape(-1)
        if s.shape[0] == 2:
            s = np.concatenate([s, [0.0]]).astype(np.float32)
        else:
            s = s[:3].astype(np.float32)
        return s

    def _crop_center(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[top:top + new_h, left:left + new_w]

    def prepare_image(self, image) -> np.ndarray:
        if self.img_size is None:
            raise AttributeError("img_size must be set before calling prepare_image")
        if isinstance(image, Image.Image):
            arr = np.asarray(image)
        else:
            arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if getattr(self, "center_crop", False):
            arr = self._crop_center(arr)
        try:
            import cv2
            arr = cv2.resize(arr, (self.img_size, self.img_size))
        except Exception:
            arr = np.asarray(Image.fromarray(arr).resize((self.img_size, self.img_size)))
        arr = arr.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

    def _prepare_inputs(self, inputs, h0, c0):
        """
        Must be overridden by subclasses.  The default implementation
        raises an error to remind you to provide the mapping.
        """
        raise NotImplementedError(
            "Subclasses must implement `_prepare_inputs` to supply model inputs."
        )
# -------------------------------------------------------------------------

class CellHunter(AutoPatcher):
    """
    Cell‑hunting policy.

    Accepts a single live observation from `observe()` per call in the form:
        [pipette(3,), stage(2 or 3,), image(H,W,[3]), resistance(scalar)]
    Internally accumulates a 16‑step history and maps it to the ONNX feeds:
        camera_image      → (1,16,3,H,W)
        pipette_positions → (1,16,3)
        stage_positions   → (1,16,3)
        resistance        → (1,16)

    Also supports new wrapper‑style models that expose inputs as:
        obs::<key>  and optionally goal::<key>
    In that case, it switches to single‑step raw HWC images (normalisation inside model)
    and fills missing goal keys by mirroring the current observation.
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

        # rolling history
        self._img_q   = collections.deque(maxlen=self.seq_len)
        self._pip_q   = collections.deque(maxlen=self.seq_len)
        self._stage_q = collections.deque(maxlen=self.seq_len)
        self._res_q   = collections.deque(maxlen=self.seq_len)

    def _prepare_inputs(self, model_input, h0, c0):
        """
        Supports:
          • Legacy sequence models (builds 16‑step stacks, CHW normalized images)
          • Wrapper models with inputs "obs::..." and optional "goal::..."
            - Single-step, raw HWC images (normalization handled inside the model)
            - Partial goal dict allowed (missing keys mirrored from obs)

        model_input may be:
          (pip, stage, image, resistance)
          {"obs": (pip, stage, image, resistance), "goal": (optional)}
          {"pipette_positions": ..., "stage_positions": ..., "camera_image": ..., "resistance": ...}
        """
        in_desc = self._get_input_desc()
        input_names = set(in_desc.keys())
        uses_obs_prefix = any(n.startswith("obs::") for n in input_names)
        # print(f"_prepare_inputs called with model_input type: {type(model_input)}")

        if uses_obs_prefix:
            if isinstance(model_input, dict):
                obs_pkg = model_input.get("obs", None)
                if obs_pkg is None and all(k in model_input for k in ("pipette_positions","stage_positions","camera_image","resistance")):
                    obs_pkg = (model_input["pipette_positions"],
                               model_input["stage_positions"],
                               model_input["camera_image"],
                               model_input["resistance"])
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
                            gimg = np.asarray(goal_pkg["camera_image"])
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg] * 3, axis=-1)
                    else:
                        gpip, gstage, gimg, gres = goal_pkg
                        gpip = np.asarray(gpip, np.float32).reshape(-1)
                        gstage = self._coerce_stage(gstage)
                        gres = np.asarray(gres, np.float32).reshape(-1)
                        if gimg is None:
                            gimg = img
                        else:
                            gimg = np.asarray(gimg)
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg] * 3, axis=-1)

                for nm in input_names:
                    if not nm.startswith("goal::"):
                        continue
                    key = nm.split("goal::", 1)[1]
                    if key == "camera_image":
                        gimg_r = self._resize_to_expected(nm, gimg)
                        inputs[nm] = self._prepare_wrapper_sequence(nm, gimg_r)
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
            "camera_image":      np.stack(list(self._img_q), 0)[None],
            "pipette_positions": np.stack(list(self._pip_q), 0)[None],
            "stage_positions":   np.stack(list(self._stage_q), 0)[None],
            "resistance":        np.stack(list(self._res_q), 0).reshape(1, -1),
        }

        inputs, h0, c0 = self._ensure_state_inputs(inputs, h0, c0, input_names=input_names)
        return inputs
class GigaSealer(AutoPatcher):
    """
    Gigasealing policy – override to provide the mapping this model expects.
    """
    def _prepare_inputs(self, inputs, h0, c0):
        # Accept the legacy history tuple: (img_q, pip_q, stage_q, res_q)
        try:
            img_q, pip_q, stage_q, res_q = inputs
        except Exception:
            raise ValueError("GigaSealer._prepare_inputs expects a 4-tuple history: (img_q, pip_q, stage_q, res_q)")
        # Example: only pressure (reuse res_q as a stand-in) & resistance
        feed = {
            "pressure":   np.array(res_q, dtype=np.float32)[None, None],  # (1,1,16)
            "resistance": np.stack(res_q, 0).reshape(1, -1),              # (1,16)
        }
        feed, _, _ = self._ensure_state_inputs(feed, h0, c0)
        return feed

class Burglar(AutoPatcher):
    """
    Break-in policy – provide its own mapping.
    """
    def _prepare_inputs(self, inputs, h0, c0):
        # Accept the legacy history tuple: (img_q, pip_q, stage_q, res_q)
        try:
            img_q, pip_q, stage_q, res_q = inputs
        except Exception:
            raise ValueError("Burglar._prepare_inputs expects a 4-tuple history: (img_q, pip_q, stage_q, res_q)")
        feed = {
            "capacitance":     np.stack(res_q, 0).reshape(1, -1),  # (1,16)
            "stage_positions": np.stack(stage_q, 0)[None],         # (1,16,3)
        }
        feed, _, _ = self._ensure_state_inputs(feed, h0, c0)
        return feed

class PipetteFinder(AutoPatcher):
    """
    Pipette finding policy.

    Accepts a single live observation from `observe()` per call in the form:
        [pipette(3,), stage(2 or 3,), image(H,W,[3])]
    Internally accumulates a 16-step history and maps it to the ONNX feeds:
        camera_image      -> (1,16,3,H,W)
        pipette_positions -> (1,16,3)
        stage_positions   -> (1,16,3)

    Also supports new wrapper-style models that expose inputs as:
        obs::<key> and optionally goal::<key>
    In that case, it switches to single-step raw HWC images (normalisation inside the model)
    and fills missing goal keys by mirroring the current observation.
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

        # rolling history
        self._img_q = collections.deque(maxlen=self.seq_len)
        self._pip_q = collections.deque(maxlen=self.seq_len)
        self._stage_q = collections.deque(maxlen=self.seq_len)

    def _prepare_inputs(self, model_input, h0, c0):
        """
        Supports:
          - Legacy sequence models (builds 16-step stacks, CHW normalized images)
          - Wrapper models with inputs "obs::..." and optional "goal::..."
            * Single-step, raw HWC images (normalization handled inside the model)
            * Partial goal dict allowed (missing keys mirrored from obs)

        model_input may be:
          (pip, stage, image)
          {"obs": (pip, stage, image), "goal": (optional)}
          {"pipette_positions": ..., "stage_positions": ..., "camera_image": ...}
        """
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
                arr = np.asarray(img)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
            return pip, stage, arr

        if uses_obs_prefix:
            if isinstance(model_input, dict):
                obs_pkg = model_input.get("obs", None)
                if obs_pkg is None and all(k in model_input for k in ("pipette_positions", "stage_positions", "camera_image")):
                    obs_pkg = (model_input["pipette_positions"],
                               model_input["stage_positions"],
                               model_input["camera_image"])
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
                    inputs[nm] = self._prepare_wrapper_sequence(nm, self._resize_to_expected(nm, img))
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
                            gimg = np.asarray(goal_pkg["camera_image"])
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg] * 3, axis=-1)
                    else:
                        seq = list(goal_pkg)
                        if len(seq) >= 1 and seq[0] is not None:
                            gpip = np.asarray(seq[0], np.float32).reshape(-1)
                        if len(seq) >= 2 and seq[1] is not None:
                            gstage = self._coerce_stage(seq[1])
                        if len(seq) >= 3 and seq[2] is not None:
                            gimg = np.asarray(seq[2])
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg] * 3, axis=-1)
                for nm in input_names:
                    if not nm.startswith("goal::"):
                        continue
                    key = nm.split("goal::", 1)[1]
                    if key == "camera_image":
                        inputs[nm] = self._prepare_wrapper_sequence(nm, self._resize_to_expected(nm, gimg))
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
