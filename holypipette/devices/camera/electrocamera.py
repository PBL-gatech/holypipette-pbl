#electrocamera.py
'''
Camera class for a Teledyne Photometrics Qimaging Retiga Electro Camera
'''
import numpy as np

from . import *
import warnings
import logging
from pyvcam import pvc
# from pyvcam.camera import Camera
import pyvcam.camera




try:
    import cv2
except ImportError:
    warnings.warn('OpenCV is not installed.')




class ElectroCamera(Camera):
    '''A camera class for the Teledyne Photometrics Qimaging Retiga Electro Camera
       more info on the camera can be found here: https://github.com/Photometrics/PyVCAM/blob/master/docs/PyVCAM%20Wrapper.md#camerapy
    '''


    def __init__(self, width: int = 720, height: int = 720):
        super().__init__()

        self.width = width # update superclass img width / height vars
        self.height = height
        self.auto_normalize = False

        #setup the electro for continuous streaming
        pvc.init_pvcam()                   # Initialize PVCAM 
        self.cam = next(pyvcam.camera.Camera.detect_camera()) # Use generator to find first camera. 
        self.cam.open()                         # Open the camera.
        print(f"CAMERA {self.cam}")
        self.cam.set_roi(0,0,width,height)

        self.cam.start_live(exp_time=20,buffer_frame_count=16)

        self.frameno = None

        self.currExposure = 0

        self.upperBound = 255
        self.lowerBound = 0

        self.last_frame_time = None
        self.fps = 0


        # self.normalize() #normalize image on startup

        self.start_acquisition() #start thread that updates camera gui

    def set_exposure(self, value: float) -> None:
        '''
        set exposure time of camera in ms
        '''
        self.currExposure = (value/1000)
        logging.debug(f"exposure changed: {self.currExposure}")
        return self.currExposure


    def get_exposure(self):
        '''return the exposure time of the camera in ms
        '''
        logging.debug(f"current exposure value: {self.currExposure}")
        return self.currExposure * 1000 #convert to ms

    def close(self):
        if self.cam:
            self.cam.close()

    def reset(self) -> None:
        pass
        self.cam.close()
        pvc.init_pvcam()                   # Initialize PVCAM 
        self.cam = next(pyvcam.camera.Camera.detect_camera()) # Use generator to find first camera. 
        self.cam.open()                         # Open the camera.
        print(f"CAMERA {self.cam}")
        self.cam.set_roi(0,0,self.width,self.height)


    def normalize(self, img = None) -> None:

        if not self.auto_normalize:
            print("NORMALIZING")   


        if img is None:
            img = self.get_16bit_image()
        self.lowerBound = img.min()
        self.upperBound = img.max()

    def autonormalize(self,flag = None):
        self.auto_normalize = flag
        return self.auto_normalize

    def get_frame_no(self) -> int:
        return self.frameno
        
    def get_16bit_image(self) -> np.ndarray:
        '''get a 16 bit color image from the camera (no normalization)
           this compares to raw_snap which returns a 8 bit image with normalization
        '''
        try:
            out = self.cam.poll_frame()
            test = out[0]
            img = test['pixel_data']
            self.lastFrame = img
            self.frameno = out[2]
        except Exception as e:
            print(f"ERROR in get_16bit_image: {e}")
            self.frameno = 0
            return None # there was an error grabbing the most recent frame
        return img

    def raw_snap(self):
        '''
        Returns the current image (8 bit color, with normalization).
        This is a blocking call (wait until next frame is available)
        '''
        img = self.get_16bit_image()

        if self.auto_normalize:
            self.normalize(img)

        if img is None:
            return None

        # apply upper / lower bounds (normalization)
        span = np.maximum(self.upperBound - self.lowerBound, 1)  # Avoid division by zero

        img = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255).astype(np.uint8)

        # resize if needed
        if self.width != None and self.height != None:
            img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)

        return img