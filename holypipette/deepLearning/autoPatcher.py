#autoPatcher.py


import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime
import h5py
import time, statistics, collections, cv2, numpy as np
class AutoPatcher:
    def __init__(self):
        # Locate model directory
        curFile = str(Path(__file__).parent.absolute())

    def load_model(self, onnx_path=None):
        """
        General loader for ONNX models. If onnx_path is None, use the default path set in __init__.
        Returns the loaded cv2.dnn.Net object.
        """

        if not Path(onnx_path).exists():
            model_dir = Path(__file__).parent / 'patchModel'
            onnx_files = list(model_dir.glob('*.onnx'))
            if onnx_files:
                onnx_path = str(onnx_files[0])
            else:
                raise FileNotFoundError(f"No ONNX model at {onnx_path} or in {model_dir}")
        # load the network
        net = cv2.dnn.readNetFromONNX(onnx_path)
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        print(f"Loaded model from {onnx_path} with output layers: {output_layers}")
        return net,layer_names,output_layers

class ModelTester:
    """
    ModelTester sets up an example test inference environment based on the model type fed in from the AutoPatcher class.
    This class loads an ONNX model using OpenCV's DNN module. It retrieves the model's layer names and identifies the output layers by checking for unconnected output nodes.
    Attributes:
        model: The neural network model loaded from the provided ONNX file.
        layer_names: List of all layer names in the model.
        output_layers: List of names corresponding to the unconnected output layers.
    Args:
        model_path (str): File path to the ONNX model file.
    """
    def __init__(self, net, layerNames, outputLayers, input_path=None):
        self.model = net
        self.layer_names = layerNames
        self.output_layers = outputLayers
        self.input = None
        self.images = np.array([])
        self.resistance = np.array([])
        self.demo_keys = []
        self.current_demo_idx = 0
        self.current_frame = 0
        # optionally auto-load if path supplied
        if input_path:
            self.load_sample_input(input_path)

    def load_sample_input(self, input_path):
        """
        Load a sample input for testing the model.
        Args:
            input_path (str): File path to the sample input hdf5.
        """
        try:
            self.input = h5py.File(input_path, 'r')
        except OSError as e:
            print(self, "File Error", f"Could not open file {input_path}:\n{e}")
            return False

        if 'data' not in self.input:
            print(self, "File Error", f"No 'data' group in {input_path}")
            return False

        self.demo_keys = sorted(self.input['data'].keys())
        if not self.demo_keys:
            print(self, "Data Error", "No demos in HDF5 file")
            return False

        demo_key = self.demo_keys[0]
        demo_path = f'data/{demo_key}/obs'
        try:
            self.images = self.input[f'{demo_path}/camera_image'][:]
            self.resistance = self.input[f'{demo_path}/resistance'][:]
            self.pipette_positions = self.input[f'{demo_path}/pipette_positions'][:]
            self.stage_positions = self.input[f'{demo_path}/stage_positions'][:]
        except KeyError as e:
            print(self, "Data Error", f"Missing dataset in {demo_path}:\n{e}")
            self.images = np.array([])
            self.resistance = np.array([])
            self.pipette_positions = np.array([])
            self.stage_positions = np.array([])
            return False

        print(f"Loaded demo '{demo_key}': images {self.images.shape}, resistance {self.resistance.shape}, pipette positions {self.pipette_positions.shape}, stage positions {self.stage_positions.shape}")
        return True

    def get_demo_frame(self, demo_idx=0, frame_idx=0):
        """
        Return a single (image, resistance) tuple.
        """
        img = self.images[frame_idx]
        res = self.resistance[frame_idx]
        pip = self.pipette_positions[frame_idx]
        stage = self.stage_positions[frame_idx]

        return img, res, pip, stage
    
    def run_inference(self, frame_idx: int):
        """
        Inference on a 16-frame sequence using deques.
        Returns None until the first 16 frames have been queued.
        """

        # -------- lazy buffer init --------
        if not hasattr(self, "buf_init"):
            H, W = self.images.shape[1:3]         # taken from HDF5
            self.im_size = (W, H)                 # (W, H) for cv2.resize
            self.seq_len = 15                    # window length

            # rolling windows
            self.img_q   = collections.deque(maxlen=self.seq_len)
            self.pip_q   = collections.deque(maxlen=self.seq_len)
            self.stage_q = collections.deque(maxlen=self.seq_len)
            self.res_q   = collections.deque(maxlen=self.seq_len)

            # RNN hidden-state tensors
            self.h0 = None
            self.c0 = None

            # >>> ADAPT these two numbers to match your training config <<<
            self.num_layers  = 2      # e.g. config["algo"]["rnn"]["num_layers"]
            self.hidden_size = 256    # e.g. config["algo"]["rnn"]["hidden_dim"]

            self.buf_init = True

        # -------- push one time-step --------
        img, res, pip, stage = self.get_demo_frame(self.current_demo_idx, frame_idx)

        img = cv2.resize(img, self.im_size).astype(np.float32) / 255.0   # HWC 0-1
        img = img.transpose(2, 0, 1)                                     # CHW

        self.img_q.append(img)
        self.pip_q.append(pip.astype(np.float32))
        self.stage_q.append(stage.astype(np.float32))
        self.res_q.append(np.array([res], np.float32))

        if len(self.img_q) < self.seq_len:          # need 16 frames first
            return None

        # stack oldest→newest and prepend batch dim
        stack = lambda q: np.stack(q, axis=0)[None]   # (1,16,…)

        inputs = {
            "camera_image"      : stack(self.img_q),
            "pipette_positions" : stack(self.pip_q),
            "stage_positions"   : stack(self.stage_q),
            "resistance"        : stack(self.res_q),
        }

        # ------ ensure hidden state is ALWAYS present ------
        if self.h0 is None:   # cold start → zeros
            self.h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
            self.c0 = np.zeros_like(self.h0)

        inputs["h0"] = self.h0
        inputs["c0"] = self.c0
        # ----------------------------------------------------

        # feed inputs
        for name, tensor in inputs.items():
            self.model.setInput(tensor, name)

        # forward pass
        outs = self.model.forward(self.output_layers)
        out_dict = dict(zip(self.output_layers, outs))

        # carry hidden state to next step
        if "h1" in out_dict and "c1" in out_dict:
            self.h0, self.c0 = out_dict["h1"], out_dict["c1"]

        return out_dict


if __name__ == "__main__":
    patcher = AutoPatcher()
    hunterNetPath  = Path(__file__).parent / 'patchModel' / 'models' / 'HEKHUNTERv0_050.onnx'
    hunterDataPath = Path(__file__).parent / 'patchModel' / 'test_data' / 'HEKHUNTER_inference_set.hdf5'

    net, layerNames, outputLayers = patcher.load_model(str(hunterNetPath))
    tester = ModelTester(net, layerNames, outputLayers, str(hunterDataPath))

    times_ms = []

    total_frames = len(tester.images)
    for idx in range(total_frames):
        start = time.perf_counter()
        result = tester.run_inference(idx)
        if result is not None:                       # only after warm-up
            times_ms.append((time.perf_counter() - start) * 1_000)

    mean_t = statistics.mean(times_ms)
    sd_t   = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print(f"Inference timing over {len(times_ms)} steps:")
    print(f"  mean = {mean_t:.2f} ms,  sd = {sd_t:.2f} ms")

