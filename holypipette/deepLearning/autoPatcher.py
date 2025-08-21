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
        """
        inputs = self._prepare_inputs(inputs, h0, c0)
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.input_names}
        outputs = self.session.run(None, filtered_inputs)
        output_dict = dict(zip(self.output_names, outputs))
        new_h0 = output_dict.get("h1", h0)
        new_c0 = output_dict.get("c1", c0)
        action = output_dict["actions"][:, -1, :]
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
        model_input: [pip(3,), stage(2|3,), image, res]
        """
        pip, stage, img, res = model_input

        # ---- coerce types & pad stage to 3D if needed ----
        pip   = np.asarray(pip,   dtype=np.float32).reshape(3)
        stage = np.asarray(stage, dtype=np.float32).reshape(-1)
        if stage.shape[0] == 2:                         # live stage is XY — add Z=0
            stage = np.concatenate([stage, [0.0]], dtype=np.float32)
        else:
            stage = stage[:3]
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

        # ---- build ONNX feed dict (matches HuntTester) ----
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
    def _prepare_inputs(self, img_q, pip_q, stage_q, res_q, h0, c0):
        # Example: only pressure (here we reuse res_q as a stand‑in) & resistance
        inputs = {
            "pressure":      np.array(res_q, dtype=np.float32)[None, None],  # (1,1,16)
            "resistance":    np.stack(res_q, 0).reshape(1, -1),              # (1,16)
        }
        # LSTM state handling
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            inputs["h0"] = h0
        if "c0" in self.input_names:
            inputs["c0"] = c0
        return inputs

class Burglar(AutoPatcher):
    """
    Break‑in policy – provide its own mapping.
    """
    def _prepare_inputs(self, img_q, pip_q, stage_q, res_q, h0, c0):
        inputs = {
            "capacitance":    np.stack(res_q, 0).reshape(1, -1),  # (1,16)
            "stage_positions": np.stack(stage_q, 0)[None],       # (1,16,3)
        }
        # LSTM state handling
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            inputs["h0"] = h0
        if "c0" in self.input_names:
            inputs["c0"] = c0
        return inputs