
import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque


def _default_providers():
    """
    Determine the preferred ONNX Runtime execution providers in order of priority.
    
    Returns:
        List[str]: Ordered list of available providers (GPU preferred, CPU last).
    """
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
    """Helper class for ONNX model input/output processing and state management."""
    @staticmethod
    def matches(name: str, logical: str) -> bool:
        """
        Check if a name matches a logical key (exact or prefix match).

        Args:
            name (str): Actual input/output name.
            logical (str): Logical name to match.

        Returns:
            bool: True if matches, False otherwise.
        """
        return name == logical or name.startswith(f"{logical}.")

    @staticmethod
    def shape_from_desc(desc):
        """
        Extract shape tuple from ONNX input/output descriptor.

        Args:
            desc: ONNX descriptor with `shape` attribute.

        Returns:
            tuple[int] | None: Shape as a tuple or None if unavailable.
        """
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
        """
        Convert ONNX descriptor type string to NumPy dtype.

        Args:
            desc: ONNX descriptor with `type` attribute.

        Returns:
            np.dtype | None: Corresponding NumPy dtype or None if unknown.
        """
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
        """
        Infer the action horizon length from descriptor shape.

        Args:
            desc: ONNX input descriptor.

        Returns:
            int | None: Length of action sequence dimension, or None.
        """
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
        """
        Infer the action horizon length from descriptor shape.

        Args:
            desc: ONNX input descriptor.

        Returns:
            int | None: Length of action sequence dimension, or None.
        """
        arr = np.asarray(array)
        target_rank = len(getattr(desc, "shape", ())) if desc else arr.ndim
        while arr.ndim < target_rank:
            arr = np.expand_dims(arr, 0)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def resize_for_desc(desc, img_hwc):
        """
        Expand an array to match descriptor rank and convert dtype if necessary.

        Args:
            desc: ONNX input descriptor.
            array: Input array.

        Returns:
            np.ndarray: Expanded array compatible with descriptor.
        """
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
        """
        Ensure an image array has three channels.

        Args:
            arr: Input 2D or 3D array.

        Returns:
            np.ndarray: 3-channel HWC array.
        """
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return arr

    @staticmethod
    def center_crop(arr: np.ndarray) -> np.ndarray:
        """
        Center-crop an image array to half its height and width.

        Args:
            arr: Input HWC or 2D array.

        Returns:
            np.ndarray: Center-cropped array.
        """
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[top:top + new_h, left:left + new_w]

    @staticmethod
    def prepare_image(image, img_size: int, center_crop: bool) -> np.ndarray:
        """
        Convert PIL or array image to normalized CxHxW tensor.

        Args:
            image: PIL.Image or ndarray.
            img_size (int): Target image size.
            center_crop (bool): Whether to center crop before resize.

        Returns:
            np.ndarray: Transposed (C,H,W) float32 array in [0,1].
        """
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
        """
        Convert PIL or array image to normalized CxHxW tensor.

        Args:
            image: PIL.Image or ndarray.
            img_size (int): Target image size.
            center_crop (bool): Whether to center crop before resize.

        Returns:
            np.ndarray: Transposed (C,H,W) float32 array in [0,1].
        """
        s = np.asarray(stage, np.float32).reshape(-1)
        if s.shape[0] == 2:
            s = np.concatenate([s, [0.0]]).astype(np.float32)
        else:
            s = s[:3].astype(np.float32)
        return s

    @staticmethod
    def default_state_value(name, desc, num_layers, hidden_size, action_horizon):
        """
        Generate default state value for RNN or cached input.

        Args:
            name (str): Input name.
            desc: Descriptor with shape and type.
            num_layers (int): Number of RNN layers.
            hidden_size (int): Hidden state size.
            action_horizon (int): Sequence length.

        Returns:
            np.ndarray: Default initialized state array.
        """
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
        """
        Make a copy of a NumPy array or None.

        Args:
            value: Input array or None.

        Returns:
            np.ndarray | None: Copied array or None.
        """
        if value is None:
            return None
        return np.array(value, copy=True)

    @staticmethod
    def coerce_state_value(name, value, template):
        """
        Make a copy of a NumPy array or None.

        Args:
            value: Input array or None.

        Returns:
            np.ndarray | None: Copied array or None.
        """
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
    """Manages RNN-style state inputs and outputs for ONNX models."""
    def __init__(self, num_layers: int, hidden_size: int):
        """
        Initialize the state manager with network parameters and empty state structures.

        Args:
            num_layers (int): Number of recurrent layers.
            hidden_size (int): Size of the hidden state per layer.
        """
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
        """
        Configure state manager with ONNX input/output descriptors.

        Args:
            input_desc (dict): Mapping of name -> ONNX input descriptor.
            input_names (iterable): Names of model inputs.
            output_names (iterable): Names of model outputs.

        Returns:
            _StateManager: self
        """
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
        """
        Reset all state buffers to default values.

        Returns:
            _StateManager: self
        """
        self.buffers = {name: _ModelIO.copy_value(template) for name, template in self.templates.items()}
        self.wrapper_history = {}
        return self

    def prepare(self, feed_dict, *, h_override=None, c_override=None, input_names=None):
        """
        Prepare feed dict with coerced state values.

        Args:
            feed_dict (dict): Initial inputs.
            h_override, c_override: Optional RNN overrides.
            input_names (iterable, optional): Subset of names to prepare.

        Returns:
            tuple[dict, np.ndarray|None, np.ndarray|None]: Prepared feed, h0, c0.
        """
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
        """
        Update internal state buffers from model outputs.

        Args:
            output_dict (dict): Output name -> array
        """
        for out_name, in_name in self.output_map.items():
            if out_name not in output_dict:
                continue
            template = self.templates.get(in_name)
            self.buffers[in_name] = _ModelIO.coerce_state_value(
                in_name, output_dict[out_name], template
            )

    def get(self, logical):
        """
        Retrieve a stored buffer value by matching its logical name.

        Args:
            logical (str): Logical identifier to search for in buffers.

        Returns:
            Any: The corresponding buffer value if found, else None.
        """
        for name, value in self.buffers.items():
            if _ModelIO.matches(name, logical):
                return value
        return None

    def snapshot(self):
        """
        Return a copy of all state buffers.

        Returns:
            dict: Name -> array copy
        """
        return {name: _ModelIO.copy_value(value) for name, value in self.buffers.items()}

    def restore(self, state_dict):
        """
        Restore state buffers from snapshot.

        Args:
            state_dict (dict): Name -> array snapshot
        """
        if not state_dict:
            return
        for name, value in state_dict.items():
            if name in self.templates:
                template = self.templates[name]
                self.buffers[name] = _ModelIO.coerce_state_value(name, value, template)

    def sequence(self, input_name, array):
        """
        Maintain a fixed-length history for a sequential input.

        Args:
            input_name (str): Name of input to sequence.
            array (np.ndarray): New value to append.

        Returns:
            np.ndarray: Stacked sequence array.
        """
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
        """
        Initialize the base AutoPatcher model wrapper.

        Args:
            onnx_path (str | Path, optional): Path to a .onnx model file to load.
            providers (list[str], optional): ONNX Runtime execution providers.
            num_layers (int, optional): Number of LSTM/hidden layers in state.
            hidden_size (int, optional): Size of hidden layers in state.
        """
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
        """
        Load ONNX model with optional execution providers.

        Args:
            onnx_path (str|Path, optional): Path to .onnx file.
            providers (list[str], optional): Execution providers.

        Returns:
            Tuple[ort.InferenceSession, List[str], List[str]]:
                - ONNX Runtime session
                - List of input names
                - List of output names

        Raises:
            FileNotFoundError:
                If no valid `.onnx` model file is provided or found in the
                default directory.
            RuntimeError:
                If the action output key cannot be determined after loading
                the model.
        """
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
        """
        Cache and report basic metadata about the currently loaded ONNX model.
        Returns: 
            Dict[str, object]:
                Dictionary containing model metadata, including:
                - "uses_wrapper" (bool): Whether model uses `obs::`-style inputs.
                - "has_goal" (bool): Whether model expects `goal::` inputs.
                - "input_names" (Tuple[str, ...]): Names of model inputs.
                - "output_names" (Tuple[str, ...]): Names of model outputs.
                - "providers" (Tuple[str, ...]): ONNX Runtime execution providers.
                - "action_shape" (Optional[Tuple[int, ...]]): Shape of action output.
                - "action_dtype" (Optional[Any]): Data type of action output.

        Raises:
            RuntimeError:
                If the model has not been loaded (i.e., `input_names` is None).
        """
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
        """
        Run a forward pass of the ONNX model and update internal state.

        Args:
            inputs (Any, optional): Model-specific input structure. Must be compatible
                with `_prepare_inputs` implementation in subclasses.
            h0 (np.ndarray, optional): Initial hidden state override.
            c0 (np.ndarray, optional): Initial cell state override.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - action (np.ndarray): Model output action of shape (1, action_dim).
                - new_h0 (np.ndarray): Updated hidden state.
                - new_c0 (np.ndarray): Updated cell state.

        Raises:
            RuntimeError: If the ONNX model does not produce the expected action output.
        """
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
        """Reset the internal recurrent state buffers."""
        self.state.reset()
        self._prefilled = False

    def get_state_snapshot(self):
        """Retrieve a copy of the current internal state buffers."""
        return self.state.snapshot()

    def set_state_snapshot(self, state_dict):
        """
        Restore internal state buffers from a snapshot.

        Args:
            state_dict (Dict[str, np.ndarray]): Previously saved state dictionary.
        """
        self.state.restore(state_dict)
    def prepare_image(self, image) -> np.ndarray:
        """
        Preprocess an image according to model expectations.

        Args:
            image (Any): Input image (PIL image or NumPy array).

        Returns:
            np.ndarray: Preprocessed image tensor.

        Raises:
            AttributeError: If `img_size` is not set before calling.
        """
        if self.img_size is None:
            raise AttributeError("img_size must be set before calling prepare_image")
        return _ModelIO.prepare_image(image, self.img_size, self.center_crop)

    def _prepare_inputs(self, inputs, h0, c0):
        """
        Construct model input dictionary from raw inputs.

        This method must be implemented by subclasses.

        Args:
            inputs (Any): Raw input data.
            h0 (np.ndarray): Initial hidden state.
            c0 (np.ndarray): Initial cell state.

        Returns:
            Dict[str, np.ndarray]: Prepared model input dictionary.

        Raises:
            NotImplementedError: Always raised in base class.
        """
        raise NotImplementedError("Subclasses must implement `_prepare_inputs` to supply model inputs.")

    def _matches_state_name(self, name, logical):
        """
        Check whether a name matches a logical state identifier.

        Args:
            name (str): Actual state name.
            logical (str): Logical name to match against.

        Returns:
            bool: True if names match.
        """
        return _ModelIO.matches(name, logical)

    def _configure_state_management(self):
        """Configure state manager using current model input/output descriptors."""
        self.state.configure(self._input_desc, self.input_names, self.output_names)

    def _initialize_state_buffers(self):
        """Initialize or reset internal state buffers."""
        self.state.reset()

    def _ensure_state_inputs(self, inputs, h0, c0, *, input_names=None):
        """
        Ensure required state inputs are present in the model feed.

        Args:
            inputs (Dict[str, np.ndarray]): Input dictionary.
            h0 (np.ndarray): Hidden state override.
            c0 (np.ndarray): Cell state override.
            input_names (Iterable[str], optional): Subset of input names to include.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
                Updated inputs, h0, and c0.
        """
        return self.state.prepare(inputs, h_override=h0, c_override=c0, input_names=input_names)

    def _update_state_from_outputs(self, output_dict):
        """
        Update internal state buffers using model outputs.

        Args:
            output_dict (Dict[str, np.ndarray]): Model outputs.
        """
        self.state.update(output_dict)

    def _shape_from_desc(self, desc):
        """
        Extract tensor shape from an ONNX descriptor.

        Args:
            desc: ONNX input/output descriptor.

        Returns:
            Tuple[int, ...] | None: Parsed shape or None if unavailable.
        """
        return _ModelIO.shape_from_desc(desc)

    def _dtype_from_desc(self, desc):
        """
        Extract NumPy dtype from an ONNX descriptor.

        Args:
            desc: ONNX input/output descriptor.

        Returns:
            np.dtype | None: Corresponding NumPy dtype.
        """
        return _ModelIO.dtype_from_desc(desc)

    def _infer_action_horizon(self, desc=None):
        """
        Infer action sequence length (horizon) from descriptor.

        Args:
            desc: Optional ONNX descriptor.

        Returns:
            int | None: Inferred horizon length.
        """
        if desc is not None:
            return _ModelIO.infer_action_horizon(desc)
        return self.state.action_horizon

    def _default_state_value(self, name):
        """
        Generate a default value for a state input.

        Args:
            name (str): State variable name.

        Returns:
            np.ndarray: Default-initialized state array.
        """
        desc = self._input_desc.get(name)
        return _ModelIO.default_state_value(
            name, desc, self.state.num_layers, self.state.hidden_size, self.state.action_horizon
        )

    def _copy_state_value(self, value):
        """
        Create a copy of a state value.

        Args:
            value (np.ndarray): Input state value.

        Returns:
            np.ndarray: Copied state value.
        """
        return _ModelIO.copy_value(value)

    def _coerce_state_value(self, name, value):
        """
        Coerce a state value to match expected template shape and dtype.

        Args:
            name (str): State variable name.
            value (Any): Input value.

        Returns:
            np.ndarray: Coerced state value.
        """
        template = self.state.templates.get(name)
        return _ModelIO.coerce_state_value(name, value, template)

    def _get_state_value(self, logical):
        """
        Retrieve a state value by logical name.

        Args:
            logical (str): Logical state identifier.

        Returns:
            np.ndarray | None: State value if found.
        """
        return self.state.get(logical)

    def _get_input_desc(self):
        """
        Get ONNX input descriptors.

        Returns:
            Dict[str, Any]: Input descriptor mapping.
        """
        return self._input_desc

    def _resize_to_expected(self, input_name, img_hwc):
        """
        Resize an image to match expected ONNX input dimensions.

        Args:
            input_name (str): Input tensor name.
            img_hwc (np.ndarray): Image in HWC format.

        Returns:
            np.ndarray: Resized image.
        """
        return _ModelIO.resize_for_desc(self._input_desc.get(input_name), img_hwc)

    def _expand_for_input(self, input_name, array):
        """
        Expand array dimensions to match ONNX input rank.

        Args:
            input_name (str): Input tensor name.
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: Expanded array.
        """
        return _ModelIO.expand_for_desc(self._input_desc.get(input_name), array)

    def _coerce_stage(self, stage):
        """
        Normalize stage position to expected shape.

        Args:
            stage (Any): Stage position input.

        Returns:
            np.ndarray: 3-element stage vector.
        """
        return _ModelIO.coerce_stage(stage)

    def _crop_center(self, arr: np.ndarray) -> np.ndarray:
        """
        Perform center crop on an image.

        Args:
            arr (np.ndarray): Input image.

        Returns:
            np.ndarray: Center-cropped image.
        """
        return _ModelIO.center_crop(arr)

    def _prepare_wrapper_sequence(self, input_name, array):
        """
        Convert input into a temporal sequence buffer for wrapper models.

        Args:
            input_name (str): Input tensor name.
            array (np.ndarray): Input data.

        Returns:
            np.ndarray: Sequence-formatted input.
        """
        return self.state.sequence(input_name, array)

    def _axis_dims(self):
        """
        Infer dimensionality of pipette, stage, and action spaces.

        Returns:
            Tuple[int, int, int | None]:
                (pipette_dim, stage_dim, action_dim)
        """
        action_dim = None
        if self.session and self.state.action_key:
            for out in self.session.get_outputs():
                if _ModelIO.matches(out.name, self.state.action_key):
                    shape = _ModelIO.shape_from_desc(out)
                    action_dim = shape[-1] if shape and isinstance(shape[-1], int) else None
                    break
        pip_dim = stage_dim = None
        for name, desc in self._input_desc.items():
            shape = _ModelIO.shape_from_desc(desc)
            if shape and isinstance(shape[-1], int):
                if "pipette_position" in name:
                    pip_dim = shape[-1] if pip_dim is None else min(pip_dim, shape[-1])
                if "stage_position" in name:
                    stage_dim = shape[-1] if stage_dim is None else min(stage_dim, shape[-1])
        pip_dim = pip_dim or (2 if action_dim == 2 else 3)
        stage_dim = stage_dim if stage_dim is not None else {2: 0, 3: 0, 4: 1, 6: 3}.get(action_dim, 0)
        return pip_dim, stage_dim, action_dim

    def _trim_axes(self, pip, stage):
        """
        Trim pipette and stage vectors to expected dimensionality.

        Args:
            pip (Any): Pipette position.
            stage (Any): Stage position.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Trimmed pipette and stage arrays.
        """
        pip_dim, stage_dim, _ = self._axis_dims()
        pip_arr = np.asarray(pip, np.float32).reshape(-1)[:pip_dim]
        stage_arr = np.asarray(stage, np.float32).reshape(-1)
        if stage_dim <= 0:
            stage_arr = stage_arr[:0]
        elif stage_dim == 1:
            stage_arr = np.asarray(stage_arr[2] if stage_arr.size >= 3 else stage_arr[:1], np.float32).reshape(1)
        else:
            stage_arr = stage_arr[:stage_dim]
        return pip_arr, stage_arr
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
        """
        Initialize the CellHunter policy.

        Args:
            onnx_path (str or Path, optional):
                Path to ONNX model.
            providers (list[str], optional):
                ONNX Runtime execution providers.
            num_layers (int):
                Number of recurrent layers.
            hidden_size (int):
                Hidden state size.
            seq_len (int):
                Length of temporal observation history.
            img_size (int):
                Image resize dimension.
            prefill_init (bool):
                Whether to prefill history buffers on first input.
            center_crop (bool):
                Whether to center crop images before resizing.
        """
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
        """
        Convert observations into model-ready inputs for CellHunter.

        Args:
            model_input (tuple or dict):
                Observation input. Either:
                - Tuple: (pipette, stage, image, resistance)
                - Dict with 'obs' and optionally 'goal'
            h0 (np.ndarray or None):
                Hidden state override.
            c0 (np.ndarray or None):
                Cell state override.

        Returns:
            dict:
                Model-ready input dictionary.
        """
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
            pip, stage = self._trim_axes(pip, stage)
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
                gpip, gstage = self._trim_axes(gpip, gstage)

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
        pip = np.asarray(pip, dtype=np.float32).reshape(-1)
        stage = self._coerce_stage(stage)
        pip, stage = self._trim_axes(pip, stage)
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
        """
        Prepare inputs for the gigasealing policy.

        Args:
            inputs (tuple):
                Tuple of (img_q, pip_q, stage_q, res_q).
            h0 (np.ndarray or None):
                Hidden state override.
            c0 (np.ndarray or None):
                Cell state override.

        Returns:
            dict:
                Model-ready input dictionary.

        Raises:
            ValueError:
                If inputs are not a valid 4-tuple.
        """
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
        """
        Prepare inputs for the break-in policy.

        Args:
            inputs (tuple):
                Tuple of (img_q, pip_q, stage_q, res_q).
            h0 (np.ndarray or None):
                Hidden state override.
            c0 (np.ndarray or None):
                Cell state override.

        Returns:
            dict:
                Model-ready input dictionary.

        Raises:
            ValueError:
                If inputs are not a valid 4-tuple.
        """
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
        """
        Initialize the PipetteFinder policy.

        Args:
            onnx_path (str or Path, optional):
                Path to ONNX model.
            providers (list[str], optional):
                ONNX Runtime execution providers.
            num_layers (int):
                Number of recurrent layers.
            hidden_size (int):
                Hidden state size.
            seq_len (int):
                Length of temporal observation history.
            img_size (int):
                Image resize dimension.
            prefill_init (bool):
                Whether to prefill history buffers on first input.
            center_crop (bool):
                Whether to center crop images before resizing.
        """
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
        """
        Prepare model inputs for ONNX inference, supporting both wrapper-style
        (`obs::` / `goal::`) inputs and sequence-based inputs.

        Args:
            model_input (Union[tuple, dict]):
                Input observation. Supported formats:
                - Tuple: (pipette_positions, stage_positions, camera_image)
                - Dict: keys include "pipette_positions", "stage_positions", "camera_image"
                - Wrapper dict: {"obs": ..., "goal": ...}
            h0 (Optional[np.ndarray]):
                Initial hidden state override.
            c0 (Optional[np.ndarray]):
                Initial cell state override.

        Returns:
            Dict[str, np.ndarray]:
                Dictionary mapping model input names to properly shaped numpy arrays
                ready for ONNX inference.

        Raises:
            ValueError:
                - If `model_input` is missing required keys.
                - If tuple input has insufficient elements.
                - If `camera_image` is missing when required.
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
                arr = _ModelIO.ensure_three_channel(np.asarray(img))
            pip_trim, stage_trim = self._trim_axes(pip, stage)
            return pip_trim, stage_trim, arr

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
                gpip, gstage = self._trim_axes(gpip, gstage)
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
        pip, stage = self._trim_axes(pip, stage)

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
