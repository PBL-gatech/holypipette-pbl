#autoPatcher.py


import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime
import h5py

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
            self.pipette_positions = self.input[f'{demo_path}/pipette_pos'][:]
            self.stage_positions = self.input[f'{demo_path}/stage_pos'][:]
        except KeyError as e:
            print(self, "Data Error", f"Missing dataset in {demo_path}:\n{e}")
            self.images = np.array([])
            self.resistance = np.array([])
            return False

        print(f"Loaded demo '{demo_key}': images {self.images.shape}, resistance {self.resistance.shape}, pipette {self.pipette_positions.shape}, stage {self.stage_positions.shape}")
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

    # def run_inference(self, frame_idx=0):
    #     """
    #     Prepare blob and run through cv2.dnn.
    #     """
    #     img, _ = self.get_demo_frame(self.current_demo_idx, frame_idx)
    #     blob = cv2.dnn.blobFromImage(img)
    #     self.model.setInput(blob)
    #     outputs = self.model.forward(self.output_layers)
    #     return dict(zip(self.output_layers, outputs))


if __name__ == '__main__':
    patcher = AutoPatcher()
    hunterNetPath = str(Path(__file__).parent / 'patchModel' / 'models'/'HEKHUNTERv0_050.onnx')
    hunterNet, layerNames, outputLayer = patcher.load_model(hunterNetPath)
    hunterDataPath = str(Path(__file__).parent / 'patchModel' / 'test_data' /'HEKHUNTER_inference_set.hdf5')
    hunterTester = ModelTester(hunterNet,layerNames,outputLayer,hunterDataPath)
    hunterTester.load_sample_input(hunterDataPath)

