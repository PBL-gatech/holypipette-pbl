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

from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser, FocusLevels


try:
    import cv2
except ImportError:
    warnings.warn('OpenCV is not installed.')

# ? See __init__.py for the following line
# __all__ = ['PcoCamera']


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

        # self.cam.record(number_of_images = 10, mode='ring buffer') #use "ring buffer" mode for continuous streaming from camera
        # self.cam.wait_for_first_image()

        self.frameno = None

        self.currExposure = 0

        self.upperBound = 255
        self.lowerBound = 0

        self.last_frame_time = None
        self.fps = 0

        # self.pipetteFinder = PipetteFinder()
        self.pipetteFocuser = PipetteFocuser()

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
            # self.cam.stop()
            self.cam.close()

    def reset(self) -> None:
        pass
        self.cam.close()
        pvc.init_pvcam()                   # Initialize PVCAM 
        self.cam = next(pyvcam.camera.Camera.detect_camera()) # Use generator to find first camera. 
        self.cam.open()                         # Open the camera.
        print(f"CAMERA {self.cam}")
        self.cam.set_roi(0,0,self.width,self.height)

        # self.cam = pco.Camera()
        
        # config = {'exposure time': 10e-3,
        #             'roi': (0, 0, 2048, 2048),
        #             'timestamp': 'off',
        #             'pixel rate': 500_000_000,
        #             'trigger': 'auto sequence',
        #             'acquire': 'auto',
        #             'metadata': 'on',
        #             'binning': (1, 1)}
        # self.cam.configuration = config

        # self.cam.record(number_of_images=10, mode='ring buffer')
        # self.cam.wait_for_first_image()

    def normalize(self, img = None) -> None:

        if not self.auto_normalize:
            print("NORMALIZING")   

        # print(f"BEFORE IMAGE: {img}")
        if img is None:
            img = self.get_16bit_image()
            # print(f"IMAGE after get_16bit_image: {img}")
            # print(type(img))
        # print(f"AFTER IMAGE: {img}")
        #is there a better way to do this?
        #maybe 2 stdevs instead?
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
            # logging.debug(f"Got image from camera")
            # print(meta)
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
            # print("AutoNormalizing")
            self.normalize(img)

        if img is None:
            return None

        # if img is not None:
        #     focusLvl = self.pipetteFocuser.get_pipette_focus(img)
        #     print(focusLvl)

        # apply upper / lower bounds (normalization)
        span = np.maximum(self.upperBound - self.lowerBound, 1)  # Avoid division by zero

        img = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255).astype(np.uint8)

        # resize if needed
        if self.width != None and self.height != None:
            img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)

        # if img is not None:
        #     out = self.pipetteFinder.find_pipette(img)
        #     if out is not None:
        #         img = cv2.circle(img, out, 2, 0, 2)

        # find good points to track
        # corners = cv2.goodFeaturesToTrack(img, 250, 0.005, 10)
        # corners = np.int0(corners)
        
        # draw points on image

        # we only want the top 100 corners

        # if self.lastFrame is not None:
        #     flow = cv2.calcOpticalFlowFarneback(self.lastFrame, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return img