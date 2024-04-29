'''
Camera for a Qimaging Rolera Bolt Camera.
'''
import numpy as np
import time

from . import *
import warnings
from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser, FocusLevels
from collections import deque
import cv2
import pymmcore
import sys
import threading
import copy
import logging
logging.basicConfig(level=logging.INFO)
__all__ = ['QImagingCam']

sys.path.append('C:\\Program Files\\Micro-Manager-2.0\\')
class QImagingCam(Camera):
    '''A camera class for the Qimaging Rolera Bolt camera.
  
    '''

    def __init__(self, width : int = 1280, height : int = 1024):
        super(QImagingCam, self).__init__()

        self.width = width #update superclass img width / height vars
        self.height = height
      

        # self.mmc = pymmcore.CMMCore()
        # print('-----setup cam-----')
        # mm_dir = 'C:\\Program Files\\Micro-Manager_2.0\\' # locate your micromanager folder
        # # mm_dir = '/mnt/c/Users/sa-forest/Desktop/' # locate your micromanager folder
        # self.mmc.setDeviceAdapterSearchPaths([mm_dir])
        # print(self.mmc.getVersionInfo())
        # print(self.mmc.getAPIVersionInfo())

        # print('-----load cam-----')
        # # print(os.path.join(mm_dir, 'MMConfig_1.cfg'))
        # self.mmc.loadSystemConfiguration(mm_dir +'MMConfig_QCam.cfg') # load your micromanager camera configuration file
        # self.mmc.setExposure(33)
        # self.mmc.snapImage()
        # self.prev_frame = self.mmc.getImage().copy()
        # self.mmc.startContinuousSequenceAcquisition(1)

        # cv2.imwrite('test.png', self.prev_frame)



        self.mmc = pymmcore.CMMCore()
        logging.info('-----setup cam-----')
        # print('-----setup cam-----')
        mm_dir = 'C:\\Program Files\\Micro-Manager-2.0\\'  # Update this path
        logging.info('-----load cam-----')
        # print('-----load cam-----')
        self.mmc.setDeviceAdapterSearchPaths([mm_dir])

        # print(self.mmc.getVersionInfo())
        # print(self.mmc.getAPIVersionInfo())
        # print('-----start cam-----')
        self.mmc.loadSystemConfiguration(mm_dir + 'MMConfig_QCam.cfg')
        self.mmc.setExposure(33)
        self.mmc.snapImage()
        self.prev_frame = self.mmc.getImage().copy()
        # cv2.imwrite('test.png', self.prev_frame)
        logging.info('-----start cam-----')
        self.mmc.startContinuousSequenceAcquisition(1)
       
        self.frameno = None
        self.currExposure = 0
        self.upperBound = 255
        self.lowerBound = 0

        self.normalize() #normalize image on startup

        self.start_acquisition() #start thread that updates camera gui

    def set_exposure(self, value):
        pass

    def get_exposure(self):
        '''return the exposure time of the camera in ms
        '''
        return 0 #convert to ms

    def get_frame_rate(self):
        return 0

    def reset(self):
        pass 

    def normalize(self, img = None):
        if img is None:
            img = self.get_16bit_image()

        #is there a better way to do this?
        #maybe 2 stdevs instead?
        self.lowerBound = img.min()
        self.upperBound = img.max()

    def get_frame_no(self):
        return 0 #TODO: is this ok?
        
    def get_16bit_image(self):
        '''get a 16 bit color image from the camera (no normalization)
           this compares to raw_snap which returns a 8 bit image with normalization
        '''
        
        if self.mmc.getRemainingImageCount() > 0:
            self.prev_frame = copy.deepcopy(self.mmc.popNextImage()).astype(np.uint8)
            # self.prev_frame = self.mmc.popNextImage().copy().astype(np.uint16)
        

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
    
    def close(self):
        logging.info("closing camera in definition")
        self.stop_acquisition()
        # self.mmc.stopSequenceAcquisition()
        # print("stopped sequence acquisition")
        # print("reset mmc")
        self.mmc.unloadAllDevices()
        # print("unloaded all devices")
        return

    def __del__(self):
        self.mmc.stopSequenceAcquisition()
        self.mmc.unloadDevice('Camera')

    def new_frame(self):
        '''
        Returns True if a new frame is available
        '''
        return (self.mmc.getRemainingImageCount() > 0)

