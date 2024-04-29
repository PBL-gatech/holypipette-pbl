'''
Camera for the Moscow rig's Qimaging Rolera Bolt camera.
'''
import numpy as np
import time

from . import *
import warnings
from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser, FocusLevels
from collections import deque
import pymmcore
import copy
import threading

try:
    import cv2
except ImportError:
    warnings.warn('OpenCV is not installed.')

__all__ = ['moscow_qcam_camera.py']


class moscowQCamera(Camera):
    '''A camera class for the moscow rig's Qimaging Rolera Bolt camera.
    '''

    def __init__(self, width : int = 2048, height : int = 2048):
        super(moscowQCamera, self).__init__()

        self.width = width #update superclass img width / height vars
        self.height = height
        self.lock = threading.RLock()
        self.lastFrame = None

        #setup the moscow camera for continuous streaming
        self.cam = pymmcore.CMMCore()
        print('-----setup cam-----')
        #mm_dir = 'C:\\Program Files\\Micro-Manager_2.0.1_20230720\\' # locate your micromanager folder
        mm_dir = 'C:\\Program Files\\Micro-Manager-2.0\\'  # Update this path
        print('-----load cam-----')
        self.cam.setDeviceAdapterSearchPaths([mm_dir])
        print(self.cam.getVersionInfo())
        print(self.cam.getAPIVersionInfo())
        self.cam.loadSystemConfiguration(mm_dir + 'MMConfig_QCam.cfg')
        self.cam.setExposure(33)
        self.cam.snapImage()
        self.prev_frame = self.cam.getImage().copy()
        # cv2.imwrite('test.png', self.prev_frame)

        self.cam.startContinuousSequenceAcquisition(1)
        print('-----start cam-----')


        self.frameno = None
        self.normalizeFlag = False  # Add this line to initialize the flag
        self.currExposure = 0
        self.upperBound = 255
        self.lowerBound = 0
        # self.pipetteFinder = PipetteFinder()
        self.pipetteFocuser = PipetteFocuser()
        self.lastFrame = self.prev_frame
        self.normalize() #normalize image on startup


        self.start_acquisition() #start thread that updates camera gui

    def set_exposure(self, value):
        pass

    def get_exposure(self):
        '''return the exposure time of the camera in ms
        '''
        return 0


    def close(self):
        pass

    def get_frame_rate(self):
        return 0

    def reset(self):
        pass

    def normalize(self, img = None):
        pass
                  

    def get_frame_no(self):
        return 0
        
    def get_16bit_image(self):
        '''get a 16 bit color image from the camera (no normalization)
           this compares to raw_snap which returns a 8 bit image with normalization
        '''
        self.lock.acquire()
        if self.cam.getRemainingImageCount() > 0:
            self.prev_frame = copy.deepcopy(self.cam.popNextImage()).astype(np.uint8)
        self.lock.release()
        return self.prev_frame
        

    def raw_snap(self):
        '''
        Returns the current image (8 bit color, with normalization).
        This is a blocking call (wait until next frame is available)
        '''
        img = self.get_16bit_image()

        # if img is not None:
        #     focusLvl = self.pipetteFocuser.get_pipette_focus(img)
        #     print(focusLvl)

        #apply upper / lower bounds (normalization)
        span = self.upperBound - self.lowerBound

        if span == 0:
            span = 1 #prevent divide by 0 for blank images

        img = img.astype(np.float32)
        img = img - self.lowerBound
        img = img / span

        #convert to 8 bit color
        img = img * 255
        img[np.where(img < 0)] = 0
        img[np.where(img > 255)] = 255
        img = img.astype(np.uint8)

        #resize if needed
        if self.width != None and self.height != None:
            img = cv2.resize(img, (self.width, self.height), interpolation= cv2.INTER_LINEAR)

        return img