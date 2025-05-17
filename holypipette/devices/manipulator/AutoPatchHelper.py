import time
import cv2
import numpy as np
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.manipulator import Manipulator
from holypipette.devices.camera import Camera
from holypipette.deepLearning.autoPatcher import AutoPatcher


class AutoPatchHelper():
    """
    A helper class to aid with different stages of the auto patching process.
    Neuron Hunting
    Gigasealing 
    Break in
    """
    def __init__(self, pipette: Manipulator, microscope: Microscope, camera: Camera, autoPatcher: AutoPatcher):
        self.pipette: Manipulator = pipette
        self.microscope: Microscope = microscope
        self.camera = camera
        self.autoPatcher: AutoPatcher = autoPatcher