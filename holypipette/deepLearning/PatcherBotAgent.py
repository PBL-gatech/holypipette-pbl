# necessary imports
import onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque

# class Model Importer. takes in .onnx model, input and output normalization .npz, and model desc .json
# Builds model and sends to Model Inferencer, to perform inference.
#
# Model inferencer take ins in the observation, applies normalization, image crop and channel switch to the data, and passes to inference method
# it then passes the output of such to the action unnormalizer and then passes the action to the controller


class ModelImporter:

    def __init__(self, onnx_model_path: str, input_normalization_npz_path: str, output_normalization_npz_path: str, model_desc_json_path: str) -> None:
        pass


    def load(self):
        '''
        Load the model 
        '''
        pass


class ModelInferencer:

    def __init__(self, model_importer: ModelImporter) -> None:
        pass

    def obs_norm(self, observation: np.ndarray) -> np.ndarray:
        pass

    def action_unnorm(self, action: np.ndarray) -> np.ndarray:
        pass
    
    def process_obs(self,observation):
        pass

    def process_action(self,action):
        pass
    
    def set_goal(self, goal: float) -> None:
        pass

    def get_goal(self):
        pass

    def inference(self,observation:np.ndarray)->np.ndarray:
        
        # get observation -> process observation -> calls obs_norm, passes
        # into model and then performs process action, which calls action_unnorm
        # and returns action
        # if goal observation model, extract goal from observation array and call set goal


        pass



class PipetteFinder(ModelInferencer):
    def __init__(self, model_path: str = None):
        pass


class CellHunter(ModelInferencer):

    def __init__(self, model_path: str = None):
        pass

class GigaSealer(ModelInferencer):
    def __init__(self, model_path: str = None):
        pass

class Burglar(ModelInferencer):
    def __init__(self, model_path: str = None):
        pass




