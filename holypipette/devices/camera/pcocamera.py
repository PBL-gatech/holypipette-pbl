'''
Camera for a PCO Panda Camera
'''
import numpy as np
import time

from . import *
import warnings
import pco
from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser, FocusLevels
import logging


try:
    import cv2
except ImportError:
    warnings.warn('OpenCV is not installed.')

__all__ = ['PcoCamera']


class PcoCamera(Camera):
    '''A camera class for the PCO Panda microscope camera.
       more info on the camera can be found here: https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.panda_manual.pdf
    '''

    PCO_RECORDER_LATEST_IMAGE = 0xFFFFFFFF

    def __init__(self, width: int = 1280, height: int = 1280):
        super(PcoCamera, self).__init__()

        self.width = width #update superclass img width / height vars
        self.height = height

        #setup the pco camera for continuous streaming
        self.cam = pco.Camera()

        print(f"CAMERA {self.cam}")

        # self.ca .sdk.set_timestamp_mode('binary & ascii')
        config = {'exposure time': 5e-3,
                    'roi': (385, 385, 1664, 1664),
                    'timestamp': 'off',
                    'trigger': 'auto sequence',
                    'acquire': 'auto',
                    'metadata': 'on',
                    'binning': (1, 1)}
        self.cam.configuration = config

        self.cam.record(number_of_images=10, mode='ring buffer') #use "ring buffer" mode for continuous streaming from camera
        self.cam.wait_for_first_image()

        self.frameno = None

        self.currExposure = 0

        self.upperBound = 255
        self.lowerBound = 0

        self.last_frame_time = None
        self.fps = 0

        # self.pipetteFinder = PipetteFinder()
        self.pipetteFocuser = PipetteFocuser()

        self.normalize() #normalize image on startup

        self.start_acquisition() #start thread that updates camera gui

    def set_exposure(self, value) -> None:
        self.cam.set_exposure_time(value / 1000)

    def get_exposure(self):
        '''return the exposure time of the camera in ms
        '''
        exposure = self.cam.get_exposure_time() # this is in seconds
        self.currExposure = exposure
        return exposure * 1000 #convert to ms

    def close(self):
        if self.cam:
            self.cam.stop()
            self.cam.close()

    def reset(self) -> None:
        self.cam.close()
        self.cam = pco.Camera()
        
        config = {'exposure time': 10e-3,
                    'roi': (0, 0, 2048, 2048),
                    'timestamp': 'off',
                    'pixel rate': 500_000_000,
                    'trigger': 'auto sequence',
                    'acquire': 'auto',
                    'metadata': 'on',
                    'binning': (1, 1)}
        self.cam.configuration = config

        self.cam.record(number_of_images=10, mode='ring buffer')
        self.cam.wait_for_first_image()

    def normalize(self, img = None) -> None:
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

    def get_frame_no(self) -> int:
        return self.frameno
        
    def get_16bit_image(self) -> np.ndarray:
        '''get a 16 bit color image from the camera (no normalization)
           this compares to raw_snap which returns a 8 bit image with normalization
        '''
        # if self.frameno == self.cam.rec.get_status()['dwProcImgCount'] and self.lastFrame is not None:
        #     return self.lastFrame
        # else:
        # print('-----get 16 bit image----- PcoCamera.py')
        self.frameno = self.cam.rec.get_status()['dwProcImgCount']
        self.get_frame_rate()
        
        try:
            # print(f"IMAGE NUMBER: {PcoCamera.PCO_RECORDER_LATEST_IMAGE}")
            # img, meta = self.cam.image(PcoCamera.PCO_RECORDER_LATEST_IMAGE)
            # this is the line that is causing an error if pco <= 2.1.2
            img, meta = self.cam.image(image_number=PcoCamera.PCO_RECORDER_LATEST_IMAGE)
            self.lastFrame = img
            # logging.debug(f"Got image from camera: {datetime.now()}")
            # print(meta)
        except Exception as e:
            print(f"ERROR in get_16bit_image: {e}")
            return self.last_frame # there was an error grabbing the most recent frame

        return img

    def raw_snap(self):
        '''
        Returns the current image (8 bit color, with normalization).
        This is a blocking call (wait until next frame is available)
        '''
        img = self.get_16bit_image()

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