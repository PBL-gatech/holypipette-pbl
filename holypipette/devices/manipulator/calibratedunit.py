# coding=utf-8
"""
A class to handle a manipulator unit with coordinates calibrated to the reference system of a camera.
It contains methods to calibrate the unit.

Should messages be issued?
Also ranges should be taken into account

Should this be in devices/ ? Maybe in a separate calibration folder
"""
from __future__ import print_function
from __future__ import absolute_import
from typing import List
from .manipulatorunit import *
from numpy import (array, zeros, dot, arange, vstack, sign, pi, arcsin,
                   mean, std, isnan)
import cv2
import numpy as np
import time
import math
from holypipette.devices.manipulator import *

from numpy.linalg import inv, pinv, norm
from threading import Thread
from .StageCalHelper import FocusHelper, StageCalHelper
from .PipetteCalHelper import PipetteCalHelper, PipetteFocusHelper

__all__ = ['CalibratedUnit', 'CalibrationError', 'CalibratedStage']

verbose = True

##### Calibration parameters #####
from holypipette.config import Config, NumberWithUnit, Number, Boolean


class CalibrationConfig(Config):
    position_update = NumberWithUnit(1000, unit='ms',
                                     doc='dt for updating displayed pos.',
                                     bounds=(0, 10000))
    
    autofocus_dist = NumberWithUnit(500, unit='um',
                                     doc='z dist to scan for autofocusing.',
                                     bounds=(100, 5000))
    
    stage_diag_move = NumberWithUnit(500, unit='um',
                                     doc='x, y dist to move for stage cal.',
                                     bounds=(0, 10000))
    
    frame_lag = NumberWithUnit(4, unit='frames',
                                     doc='number of frames between for computing change with optical flow',
                                     bounds=(1, 20))
    
    pipette_diag_move = NumberWithUnit(200, unit='um',
                                     doc='x, y dist to move for pipette cal.',
                                     bounds=(50, 10000))
    

    categories = [('Stage Calibration', ['autofocus_dist', 'stage_diag_move', 'frame_lag']),
                  ('Pipette Calibration', ['pipette_diag_move']),
                  ('Display', ['position_update'])]


class CalibrationError(Exception):
    def __init__(self, message='Device is not calibrated'):
        self.message = message

    def __str__(self):
        return self.message


class CalibratedUnit(ManipulatorUnit):
    def __init__(self, unit, stage=None, microscope=None, camera=None,
                 config=None):
        '''
        A manipulator unit calibrated to a fixed reference coordinate system.
        The stage refers to a platform on which the unit is mounted, which can
        be None.

        Parameters
        ----------
        unit : ManipulatorUnit for the (XYZ) unit
        stage : CalibratedUnit for the stage
        microscope : ManipulatorUnit for the microscope (single axis)
        camera : a camera, ie, object with a snap() method (optional, for visual calibration)
        '''
        ManipulatorUnit.__init__(self, unit.dev, unit.axes)
        self.saved_state_question = ('Move manipulator and stage back to '
                                     'initial position?')
        if config is None:
            config = CalibrationConfig(name='Calibration config')
        self.config = config
        if stage is None: # In this case we assume the unit is on a fixed element.
            self.stage = FixedStage()
            self.fixed = True
        else:
            self.stage = stage
            self.fixed = False
        self.microscope = microscope
        self.camera = camera

        self.calibrated = False
        self.must_be_recalibrated = False
        self.up_direction = [-1 for _ in range(len(unit.axes))] # Default up direction, determined during calibration

        self.pipette_position = None
        self.photos = None
        self.photo_x0 = None
        self.photo_y0 = None

        # Matrices for passing to the camera/microscope system
        self.M = zeros((3,len(unit.axes))) # stage units (in micron) to camera
        self.Minv = zeros((len(unit.axes),3)) # Inverse of M
        self.r0 = zeros(3) # offset for px -> um conversion
        self.r0_inv = zeros(3) # offset for um -> px conversion
        self.unit = unit

        self.emperical_offset = np.zeros(3) # offset for pipette position in px based on deep learning model

        #setup pipette calibration helper class
        self.pipetteCalHelper = PipetteCalHelper(unit, self.microscope, camera, stage)
        self.pipetteFocusHelper = PipetteFocusHelper(unit, camera)

    def save_state(self):
        if self.stage is not None:
            self.stage.save_state()
        if self.microscope is not None:
            self.microscope.save_state()
        self.saved_state = self.position()

    def delete_state(self):
        if self.stage is not None:
            self.stage.delete_state()
        if self.microscope is not None:
            self.microscope.delete_state()
        self.saved_state = None

    def recover_state(self):
        if self.stage is not None:
            self.stage.recover_state()
        if self.microscope is not None:
            self.microscope.recover_state()
        self.absolute_move(self.saved_state)

    def pixels_to_um(self, pos_pixels):
        '''
        Converts pixel coordinates to pipette um.
        '''
        if self.Minv.shape[1] == 2: #2x2 stage movement
            xy = dot(self.Minv, pos_pixels[0:2]) + self.r0_inv[0:2]
            return np.array([xy[0], xy[1], 0])
        else: #3x3 pipette movement
            return dot(self.Minv, pos_pixels) + self.r0_inv
    
    def pixels_to_um_relative(self, pos_pixels):
        '''
        Converts pixel coordinates to pipette um.
        '''
        if self.Minv.shape[1] == 2: #2x2 stage movement
            xy = dot(self.Minv, pos_pixels[0:2])
            return np.array([xy[0], xy[1], 0])
        else: #3x3 pipette movement
            return dot(self.Minv, pos_pixels)
    
    def um_to_pixels(self, pos_microns):
        '''
        Converts um to pixel coordinates.
        '''
        return dot(self.M, pos_microns) + self.r0 - self.emperical_offset
    
    def um_to_pixels_relative(self, pos_microns):
        '''
        Converts um to pixel coordinates.
        '''
        return dot(self.M, pos_microns)

    def reference_position(self, include_offset = True):
        '''
        Position of the pipette in pixels (camera coordinate frame)

        Returns
        -------
        The current position in um as an XYZ vector.
        '''
        # if not self.calibrated:
        #     raise CalibrationError
        pos_um = self.position() # position vector (um) in manipulator unit system
        if include_offset:
            pos_pixels = self.um_to_pixels(pos_um) + self.stage.reference_position() + self.emperical_offset
        else:
            pos_pixels = self.um_to_pixels(pos_um) + self.stage.reference_position()
        return pos_pixels # position vector (pixels) in camera system

    def reference_move(self, pos_pixels, yolo_correction=True):
        '''
        Moves the unit to position pos_pixels in reference camera system, without moving the stage.

        Parameters
        ----------
        r : XYZ position vector in um
        safe : if True, moves the Z axis first or last, so as to avoid touching the coverslip
        '''

        if np.isnan(np.array(pos_pixels)).any():
            raise RuntimeError("can not move to nan location.")
        
        print(f'Move position: {pos_pixels}')
        print(f'Reference position: {self.reference_position()}')
        pos_micron = self.pixels_to_um(pos_pixels - self.stage.reference_position()) # position vector (um) in manipulator unit system

        self.absolute_move(pos_micron)
        self.wait_until_still()

        if isinstance(self, CalibratedStage) or isinstance(self, FixedStage):
            return
        
        if not yolo_correction:
            return
        
        emperical_poses = []
        for i in range(10):
            _, _, _, frame = self.camera.raw_frame_queue[0]
            pos = self.pipetteCalHelper.pipetteFinder.find_pipette(frame)
            if pos != None:
                emperical_poses.append([pos[0], pos[1]])
        
        if len(emperical_poses) == 0:
            print('No pipette found in image, can\'t run correction')
            return
        
        pos_pixels_emperical = np.median(emperical_poses, axis=0)
        pos_pixels_theoretical = self.reference_position()[0:2]

        print('Theoretical position: ', pos_pixels_theoretical)
        print('Emperical position: ', pos_pixels_emperical)
        print('abs move completed!')

        #update offset
        self.emperical_offset[0:2] = pos_pixels_emperical - pos_pixels_theoretical
        print('Emperical offset (pix): ', self.emperical_offset)
              
        correction = self.pixels_to_um_relative(self.emperical_offset)

        #recalculate setpoint after offset correction
        pos_micron = self.pixels_to_um(pos_pixels - self.stage.reference_position()) - correction

        #move to emperical position
        self.absolute_move(pos_micron, blocking=True)

    def focus(self):
        '''
        Move the microscope so as to put the pipette tip in focus
        '''
        if not self.calibrated:
            raise CalibrationError('Pipette not calibrated')
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')
        
        self.microscope.absolute_move(self.reference_position()[2] + 200)
        self.microscope.wait_until_still()
        self.microscope.absolute_move(self.reference_position()[2])
        self.microscope.wait_until_still()

    def autofocus_pipette(self):
        '''Use the microscope image to put the pipette in focus
        '''
        print('Autofocusing pipette')
        self.pipetteFocusHelper.focus()

    def safe_move(self, r, yolo_correction=True):
        '''
        Moves the device to position x (an XYZ vector) in a way that minimizes
        interaction with tissue.

        If the movement is down, the manipulator is first moved horizontally,
        then along the pipette axis.
        If the movement is up, a direct move is done.

        Parameters
        ----------
        r : target position in um, an (X,Y,Z) vector
        withdraw : in um; if not 0, the pipette is withdrawn by this value from the target position x
        recalibrate : if True, pipette is recalibrated 1 mm before its target
        '''
        if not self.calibrated:
            raise CalibrationError
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')

        # r from pyQt has origin at the center of the image, move origin to the top left corner (as expected by calibration)
        r = np.array(r)
        r = r + np.array([self.camera.width // 2, self.camera.height // 2, 0])

        self.reference_move(r, yolo_correction) # Or relative move in manipulator coordinates, first axis (faster)


    def pixel_per_um(self, M=None):
        '''
        Returns the objective magnification in pixel per um, calculated for each manipulator axis.
        '''
        if M is None:
            M = self.M
        p = []
        for axis in range(len(self.axes)):
            # The third axis is in um, the first two in pixels, hence the odd formula
            p.append(((M[0,axis]**2 + M[1,axis]**2))**.5) #TODO: is this correct? 
        return p
    
    def record_cal_point(self):
        self.pipetteCalHelper.record_cal_point()

    def finish_calibration(self):
        '''
        Automatic calibration of the pipette manipulator.
        '''
        
        # move the pipette and create a calibration matrix (pix -> um)
        mat = self.pipetteCalHelper.calibrate()
        
        #make matrix 4x4 for inverse
        mat = np.vstack((mat, np.array([0,0,0,1])))

        self.stage.pipette_cal_position = self.stage.position() #update the position the pipette was calibrated at

        # *** Compute the (pseudo-)inverse ***
        mat_inv = pinv(mat)

        print(f'calibration matrix: {mat}')
        print('inv : ', mat_inv)

        # store r0 and r0_inv
        self.r0 = mat[0:3, 3] #um -> pixels offset
        self.r0_inv = mat_inv[0:3, 3] #pixels -> um offset

        #just 3x3 portion of M for self.M
        self.M = mat[0:3, 0:3]

        #just 3x3 portion of Minv
        self.Minv = mat_inv[0:3, 0:3]


        #check for nan values (invalid cal)
        if isnan(self.M).any() or isnan(self.Minv).any():
            raise CalibrationError('Matrix contains NaN values')

        print('Calibration Successful!')
        print('M: ', self.M)
        print('r0: ', self.r0)
        print()
        print('Minv: ', self.Minv)
        print('r0_inv: ', self.r0_inv)

        self.calibrated = True
        self.must_be_recalibrated = False
    

    def recalibrate_pipette(self):
        '''recalibrate pipette offset while keeping matrix
        '''
        if self.M is None or self.Minv is None:
            raise Exception("initial calibration required for single point recalibration!")
        
        print('recalculating piptte offsets...')
        emperical_poses = []
        for i in range(10):
            _, _, _, frame = self.camera.raw_frame_queue[0]
            pos = self.pipetteCalHelper.pipetteFinder.find_pipette(frame)
            if pos != None:
                emperical_poses.append([pos[0], pos[1]])
        
        if len(emperical_poses) == 0:
            print('No pipette found in image, can\'t run correction')
            return
        
        new_offset = np.zeros(3)
        pos_pixels_emperical = np.median(emperical_poses, axis=0)
        pos_pixels_emperical = np.append(pos_pixels_emperical, self.microscope.position())
        new_offset = pos_pixels_emperical - self.um_to_pixels_relative(self.dev.position()) - self.stage.reference_position()
        print('Old offsets: ', self.r0, self.r0_inv)

        self.r0 = np.array(new_offset)

        #create r0_inv
        #first create homogenous matrix

        homogenous_mat = np.zeros((4,4))
        homogenous_mat[0:3, 0:3] = self.M
        homogenous_mat[0:3, 3] = self.r0
        homogenous_mat[3, 3] = 1

        #invert
        homogenous_mat_inv = pinv(homogenous_mat)

        #get r0_inv
        self.r0_inv = homogenous_mat_inv[0:3, 3]
        self.calibrated = True
        self.must_be_recalibrated = False

        print('New offsets: ', self.r0, self.r0_inv)


    def save_configuration(self):
        '''
        Outputs configuration in a dictionary.
        '''
        config = {'up_direction' : self.up_direction,
                  'M' : self.M,
                  'r0' : self.r0}

        return config

    def load_configuration(self, config):
        '''
        Loads configuration from dictionary config.
        Variables not present in the dictionary are untouched.
        '''
        self.M = config.get('M', self.M)
        self.Minv = pinv(self.M)
        self.r0 = np.zeros(self.M.shape[0])
        self.r0_inv = np.zeros(self.M.shape[0])
        if self.M.shape[0] == 3:
            self.must_be_recalibrated = True #the pipette offsets need to be recalibrated upon reboot.
        self.calibrated = True

class CalibratedStage(CalibratedUnit):
    '''
    A horizontal stage calibrated to a fixed reference coordinate system.
    The optional stage refers to a platform on which the unit is mounted, which can
    be None.
    The stage is assumed to be parallel to the focal plane (no autofocus needed)

    Parameters
    ----------
    unit : ManipulatorUnit for this stage
    stage : CalibratedUnit for a stage on which this stage might be mounted
    microscope : ManipulatorUnit for the microscope (single axis)
    camera : a camera, ie, object with a ``snap()`` method (optional, for visual calibration)
    '''
    def __init__(self, unit, stage=None, microscope=None, camera=None,
                 config=None):
        CalibratedUnit.__init__(self, unit, stage, microscope, camera,
                                config=config)
        self.saved_state_question = 'Move stage back to initial position?'

        self.focusHelper = FocusHelper(microscope, camera)
        self.stageCalHelper = StageCalHelper(unit, camera, self.config.frame_lag)
        self.pipette_cal_position = np.zeros(2)
        self.unit = unit

        # It should be an XY stage, ie, two axes
        if len(self.axes) != 2:
            raise CalibrationError('The unit should have exactly two axes for horizontal calibration.')

    def reference_position(self):
        '''Returns the offset (in pixels) of the stage compared to where it was when calibrated
        '''
        #get delta in um
        posDelta = self.unit.position()

        #convert to pixels
        posDelta = dot(self.M, posDelta) + self.r0

        #just get x and y (only concerned with pixels)
        posDelta = posDelta[:2]

        #append 0 for z
        posDelta = np.append(posDelta, 0)

        return posDelta

    def reference_move(self, r, yolo_correction=None):
        if len(r)==2: # Third coordinate is actually not useful
            r3D = zeros(3)
            r3D[:2] = r
        else:
            r3D = r
        CalibratedUnit.reference_move(self, r3D) # Third coordinate is ignored

    def reference_relative_move(self, pos_pix):
        '''
        Moves the unit by vector r in reference camera system, without moving the stage.

        Parameters
        ----------
        pos_pix : position in pixels
        '''
        if not self.calibrated:
            raise CalibrationError
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')

            
        pos_microns = dot(self.Minv, pos_pix)
        self.relative_move(pos_microns)

    def calibrate(self):
        '''
        Automatic calibration for a horizontal XY stage

        '''
        if not self.stage.calibrated:
            self.stage.calibrate()

        self.info('Preparing stage calibration')
        # self.info("auto focusing microscope...")
        # self.focusHelper.autofocus(dist=self.config.autofocus_dist)
        # self.info("Finished focusing.")

        # use LK optical flow to determine transformation matrix
        mat = self.stageCalHelper.calibrate(dist=self.config.stage_diag_move)
        mat = np.append(mat, np.array([[0,0,1]]), axis=0)
        mat_inv = pinv(mat)

        # store r0 and r0_inv
        self.r0 = mat[0:2, 2] #um -> pixels offset
        self.r0_inv = mat_inv[0:2, 2] #pixels -> um offset

        #for M and Minv, we only want the upper 2x2 matrix (b/c assumption that z axis is equivilant), the rest of the matrix is just the identity
        self.M = mat[0:2, 0:2]
        self.Minv = mat_inv[0:2, 0:2]
        self.calibrated = True
        self.must_be_recalibrated = False

        self.info('Stage calibration done')



    def mosaic(self, width = None, height = None):
        '''
        Takes a photo mosaic. Current position corresponds to
        the top left corner of the collated image.
        Stops when the unit's position is out of range, unless
        width and height are specified.

        Parameters
        ----------
        width : total width in pixel (optional)
        height : total height in pixel (optional)

        Returns
        -------
        A large image of the mosaic.
        '''
        u0=self.position()
        if width == None:
            width = self.camera.width * 4
        if height == None:
            height = self.camera.height * 4

        dx, dy = self.camera.width, self.camera.height
        # Number of moves in each direction
        nx = 1+int(width/dx)
        ny = 1+int(height/dy)
        # Big image
        big_image = zeros((ny*dy,nx*dx))

        column = 0
        xdirection = 1 # moving direction along x axis

        try:
            for row in range(ny):
                img, _ = self.camera.snap()
                big_image[row*dy:(row+1)*dy, column*dx:(column+1)*dx] = img
                for _ in range(1,nx):
                    column+=xdirection
                    self.reference_relative_move([-dx*xdirection,0,0]) # sign: it's a compensatory move
                    self.wait_until_still()
                    self.sleep(0.1)
                    img, _ = self.camera.snap()
                    big_image[row * dy:(row + 1) * dy, column * dx:(column + 1) * dx] = img
                if row<ny-1:
                    xdirection = -xdirection
                    self.reference_relative_move([0,-dy,0])
                    self.wait_until_still()
        finally: # move back to initial position
            self.absolute_move(u0)

        cv2.imwrite('mosaic.png', big_image)

        return big_image

class FixedStage(CalibratedUnit):
    '''
    A stage that cannot move. This is used to simplify the code.
    '''
    def __init__(self):
        self.stage = None
        self.microscope = None
        self.r = array([0.,0.,0.]) # position in reference system
        self.u = array([0.,0.]) # position in stage system
        self.calibrated = True

    def position(self):
        return self.u

    def reference_position(self):
        return self.r

    def reference_move(self, r):
        # The fixed stage cannot move: maybe raise an error?
        pass

    def absolute_move(self, x, axis = None):
        pass