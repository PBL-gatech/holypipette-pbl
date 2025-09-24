import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image


class AutoPatcher:
    """
    
    """
    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400):
        self.session = None
        self.input_names = None
        self.output_names = None
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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
        self.input_names  = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"Loaded model {onnx_path} with inputs {self.input_names} → outputs {self.output_names}")
        return self.session, self.input_names, self.output_names

    def inference(self, inputs=None, h0=None, c0=None):
        """
        Run inference using the mapping supplied by `_prepare_inputs`.
        Adds robust diagnostics: prints expected vs provided shapes/dtypes if ORT fails.
        """
        # print(f"inference called with inputs type: {type(inputs)}")

        # Map caller-provided structure to ONNX feeds
        feed_dict = self._prepare_inputs(inputs, h0, c0)
        # print(f"model inputs prepared with keys: {list(feed_dict.keys())}")

        # Only pass names that the model actually takes
        filtered_inputs = {k: v for k, v in feed_dict.items() if k in self.input_names}

        # --- Optional: print expected vs provided for quick shape/dtype diff ---
        try:
            in_desc = {i.name: i for i in self.session.get_inputs()}
            for k, v in filtered_inputs.items():
                arr = np.asarray(v)
                exp_shape = tuple(in_desc[k].shape) if k in in_desc else None
                exp_type = getattr(in_desc.get(k, None), "type", None)
                # print(f"[ORT-FEED] {k}: provided shape={arr.shape} dtype={arr.dtype} | expected shape={exp_shape} dtype={exp_type}")
        except Exception:
            pass

        # --- Execute ORT, surfacing detailed info if it fails ---
        try:
            outputs = self.session.run(None, filtered_inputs)
        except Exception as e:
            print("[ERROR] onnxruntime session.run failed. Provided inputs summary:")
            try:
                for k, v in filtered_inputs.items():
                    a = np.asarray(v)
                    amin = a.min() if a.size else "NA"
                    amax = a.max() if a.size else "NA"
                    # print(f"  {k}: shape={a.shape}, dtype={a.dtype}, min={amin}, max={amax}")
            except Exception:
                print("  (could not print input summaries)")
            raise

        # Package outputs
        output_dict = dict(zip(self.output_names, outputs))
        new_h0 = output_dict.get("h1", h0)
        new_c0 = output_dict.get("c1", c0)
        # print(f"inference done, h0 shape: {None if h0 is None else getattr(h0, 'shape', None)}, "
            #   f"new_h0 shape: {None if new_h0 is None else new_h0.shape}")

        # Robust action extraction (supports (1,6) or (1,T,6))
        actions = output_dict["actions"]
        if actions.ndim == 3:
            action = actions[:, -1, :]
        elif actions.ndim == 2:
            action = actions
        else:
            raise RuntimeError(f"Unexpected actions shape: {actions.shape}")
        return action, new_h0, new_c0


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
        in_desc = {i.name: i for i in self.session.get_inputs()}
        input_names = set(in_desc.keys())
        uses_obs_prefix = any(n.startswith("obs::") for n in input_names)
        # print(f"_prepare_inputs called with model_input type: {type(model_input)}")

        # Helper: resize HWC image to the exact HxW declared by ONNX for a given input name
        def _resize_to_expected(name, img_hwc):
            # Expected shapes: obs::camera_image → (B=1, S=1, H, W, 3), goal::camera_image → (B=1, H, W, 3)
            shape = in_desc[name].shape
            # Defensive: pull H,W from the last 3 spatial dims before channel
            if len(shape) >= 4:
                H = int(shape[-3])
                W = int(shape[-2])
            else:
                return img_hwc  # unexpected; avoid altering
            arr = np.asarray(img_hwc)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{name}: expected HWC image, got {arr.shape}")
            try:
                import cv2
                arr = cv2.resize(arr, (W, H))
            except Exception:
                from PIL import Image
                arr = np.asarray(Image.fromarray(arr).resize((W, H)))
            # Keep wrapper behavior: float32 with 0..255 range (no /255.0 here)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            return arr

        def _expand(name, arr):
            exp_rank = len(in_desc[name].shape)
            a = np.asarray(arr)
            while a.ndim < exp_rank:
                a = np.expand_dims(a, 0)
            if a.dtype != np.float32:
                a = a.astype(np.float32)
            return a
        # Helper to coerce stage to 3D (pad Z=0 if needed)
        def _coerce_stage(x):
            s = np.asarray(x, np.float32).reshape(-1)
            if s.shape[0] == 2:
                s = np.concatenate([s, [0.0]]).astype(np.float32)
            else:
                s = s[:3].astype(np.float32)
            return s


        # ─────────────────────────────────────────────
        # Wrapper branch (obs::..., goal::...) — single step, raw HWC
        # ─────────────────────────────────────────────
        if uses_obs_prefix:
            # Unpack observation from tuple or mapping
            if isinstance(model_input, dict):
                obs_pkg = model_input.get("obs", None)
                if obs_pkg is None and all(k in model_input for k in ("pipette_positions","stage_positions","camera_image","resistance")):
                    obs_pkg = (model_input["pipette_positions"],
                               model_input["stage_positions"],
                               model_input["camera_image"],
                               model_input["resistance"])
                goal_pkg = model_input.get("goal", None)  # may be None or partial dict/tuple
            else:
                obs_pkg = model_input
                goal_pkg = None

            # Resolve obs fields
            if isinstance(obs_pkg, dict):
                pip   = np.asarray(obs_pkg["pipette_positions"], np.float32).reshape(-1)
                stage = _coerce_stage(obs_pkg["stage_positions"])
                img   = np.asarray(obs_pkg["camera_image"])
                res   = np.asarray(obs_pkg["resistance"], np.float32).reshape(-1)
            else:
                pip, stage, img, res = obs_pkg
                pip   = np.asarray(pip,   np.float32).reshape(-1)
                stage = _coerce_stage(stage)
                img   = np.asarray(img)
                res   = np.asarray(res,   np.float32).reshape(-1)
            if img.ndim == 2:  # gray → 3‑chan
                img = np.stack([img]*3, axis=-1)

            inputs = {}
            # Observations
            for nm in input_names:
                if not nm.startswith("obs::"):
                    continue
                key = nm.split("obs::", 1)[1]
                if key == "camera_image":
                    img_r = _resize_to_expected(nm, img)
                    inputs[nm] = _expand(nm, img_r)

                elif key == "pipette_positions":
                    inputs[nm] = _expand(nm, pip)
                elif key == "stage_positions":
                    inputs[nm] = _expand(nm, stage)
                elif key == "resistance":
                    inputs[nm] = _expand(nm, res)

            # Goals (mirror obs by default; allow partial goal dict/tuple)
            if any(nm.startswith("goal::") for nm in input_names):
                # Defaults = obs
                gpip, gstage, gimg, gres = pip, stage, img, res

                if goal_pkg is not None:
                    if isinstance(goal_pkg, dict):
                        if "pipette_positions" in goal_pkg:
                            gpip = np.asarray(goal_pkg["pipette_positions"], np.float32).reshape(-1)
                        if "stage_positions" in goal_pkg:
                            gstage = _coerce_stage(goal_pkg["stage_positions"])
                        if "resistance" in goal_pkg:
                            gres = np.asarray(goal_pkg["resistance"], np.float32).reshape(-1)
                        if "camera_image" in goal_pkg and goal_pkg["camera_image"] is not None:
                            gimg = np.asarray(goal_pkg["camera_image"])
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg]*3, axis=-1)
                    else:
                        # 4‑tuple (gpip, gstage, gimg, gres); allow None for image
                        gpip, gstage, gimg, gres = goal_pkg
                        gpip = np.asarray(gpip, np.float32).reshape(-1)
                        gstage = _coerce_stage(gstage)
                        gres = np.asarray(gres, np.float32).reshape(-1)
                        if gimg is None:
                            gimg = img
                        else:
                            gimg = np.asarray(gimg)
                            if gimg.ndim == 2:
                                gimg = np.stack([gimg]*3, axis=-1)

                for nm in input_names:
                    if not nm.startswith("goal::"):
                        continue
                    key = nm.split("goal::", 1)[1]
                    if key == "camera_image":
                        gimg_r = _resize_to_expected(nm, gimg)
                        inputs[nm] = _expand(nm, gimg_r)

                    elif key == "pipette_positions":
                        inputs[nm] = _expand(nm, gpip)
                    elif key == "stage_positions":
                        inputs[nm] = _expand(nm, gstage)
                    elif key == "resistance":
                        inputs[nm] = _expand(nm, gres)

            # RNN/GRU state if requested by the model
            if "h0" in input_names and (h0 is None or (isinstance(h0, (int,float)) and h0 == 0)):
                try:
                    hshape = tuple(int(d) for d in in_desc["h0"].shape)
                    h0 = np.zeros(hshape, np.float32)
                except Exception:
                    h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
            if "c0" in input_names and (c0 is None or (isinstance(c0, (int,float)) and c0 == 0)):
                c0 = np.zeros_like(h0) if h0 is not None else np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
            if "h0" in input_names:
                inputs["h0"] = h0
            if "c0" in input_names:
                inputs["c0"] = c0
            # print(f"model inputs prepared with keys: {list(inputs.keys())}")
            return inputs

        # ─────────────────────────────────────────────
        # Legacy branch — 16‑step stacks, CHW normalized images
        # ─────────────────────────────────────────────
        pip, stage, img, res = model_input

        # ---- coerce types & pad stage to 3D if needed ----
        pip   = np.asarray(pip,   dtype=np.float32).reshape(3)
        stage = _coerce_stage(stage)
        res   = np.float32(res)

        # ---- image → CHW float32 [0,1] with center crop + resize ----
        img = self.prepare_image(img)                   # (3, img_size, img_size)

        # ---- push into history ----
        self._img_q.append(img)
        self._pip_q.append(pip)
        self._stage_q.append(stage)
        self._res_q.append(res)

        # ---- optional prefill on very first frame ----
        if self.prefill_init and not self._prefilled and len(self._img_q) == 1:
            for _ in range(self.seq_len - 1):
                self._img_q.append(self._img_q[0].copy())
                self._pip_q.append(self._pip_q[0].copy())
                self._stage_q.append(self._stage_q[0].copy())
                self._res_q.append(np.float32(self._res_q[0]))
            self._prefilled = True

        # pad short sequences by repeating last
        while len(self._img_q) < self.seq_len:
            self._img_q.append(self._img_q[-1].copy())
            self._pip_q.append(self._pip_q[-1].copy())
            self._stage_q.append(self._stage_q[-1].copy())
            self._res_q.append(np.float32(self._res_q[-1]))

        # ---- build ONNX feed dict (matches HuntTester legacy path) ----
        inputs = {
            "camera_image":      np.stack(list(self._img_q),   0)[None],  # (1,16,3,H,W)
            "pipette_positions": np.stack(list(self._pip_q),   0)[None],  # (1,16,3)
            "stage_positions":   np.stack(list(self._stage_q), 0)[None],  # (1,16,3)
            "resistance":        np.stack(list(self._res_q),   0).reshape(1, -1),  # (1,16)
        }

        # ---- LSTM state tensors (init zeros only if model expects them) ----
        if "h0" in self.input_names and (h0 is None or (isinstance(h0, (int, float)) and h0 == 0)):
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and (c0 is None or (isinstance(c0, (int, float)) and c0 == 0)):
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            inputs["h0"] = h0
        if "c0" in self.input_names:
            inputs["c0"] = c0

        return inputs

    # ---------------- image helpers ----------------
    def _crop_center(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        h, w = arr.shape[:2]
        new_h, new_w = h // 2, w // 2
        top  = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[top:top+new_h, left:left+new_w]

    def prepare_image(self, image) -> np.ndarray:
        """
        → center‑crop (optional) → resize to (img_size,img_size)
        → float32 normalise to [0,1] → CHW
        """
        if isinstance(image, Image.Image):
            arr = np.asarray(image)
        else:
            arr = np.asarray(image)

        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)

        if self.center_crop:
            arr = self._crop_center(arr)

        try:
            import cv2
            arr = cv2.resize(arr, (self.img_size, self.img_size))
        except Exception:
            arr = np.asarray(Image.fromarray(arr).resize((self.img_size, self.img_size)))

        arr = arr.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))            # HWC → CHW

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
        # LSTM state handling
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            feed["h0"] = h0
        if "c0" in self.input_names:
            feed["c0"] = c0
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
        # LSTM state handling
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            feed["h0"] = h0
        if "c0" in self.input_names:
            feed["c0"] = c0
        return feed
