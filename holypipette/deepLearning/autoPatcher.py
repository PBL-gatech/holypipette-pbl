import onnxruntime as ort
from pathlib import Path
import numpy as np

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
            model_dir = Path(__file__).parent / "patchModel"
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
    The cell‑hunting imitation policy.
    Provides the default input mapping that uses all four deques.
    """
    def _prepare_inputs(self, img_q, pip_q, stage_q, res_q, h0, c0):
        inputs = {
            "camera_image":      np.stack(img_q, 0)[None],   # (1,16,3,H,W)
            "pipette_positions": np.stack(pip_q, 0)[None],  # (1,16,3)
            "stage_positions":   np.stack(stage_q, 0)[None], # (1,16,3)
            "resistance":        np.stack(res_q, 0).reshape(1, -1),  # (1,16)
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