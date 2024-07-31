import time
import cv2
import numpy as np
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.manipulator import Manipulator
from holypipette.devices.camera import Camera
from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser, FocusLevels
from threading import Thread

class PipetteCalHelper():
    '''A helper class to aid with Pipette Calibration
    '''
    
    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 1000

    def __init__(self, pipette: Manipulator, microscope: Microscope, camera: Camera, calibrated_stage):
        self.pipette : Manipulator = pipette
        self.microscope: Microscope = microscope
        self.camera = camera
        self.pipetteFinder : PipetteFinder = PipetteFinder()
        self.calibrated_stage = calibrated_stage
        self.cal_points = [] #the points we've recorded for calibration

        
    def record_cal_point(self):
        '''Records a calibration point by moving the pipette to the center of the pipette in the current frame
        '''
        _, _, _, frame = self.camera.raw_frame_queue[0]
        pos_pix = self.pipetteFinder.find_pipette(frame)
        if pos_pix != None:
            frame = cv2.circle(frame, pos_pix, 10, 0, 2)
            stage_pos_pix = self.calibrated_stage.reference_position()
            self.cal_points.append((pos_pix[0] - stage_pos_pix[0], pos_pix[1] - stage_pos_pix[1], self.microscope.position(), self.pipette.position()[0], self.pipette.position()[1], self.pipette.position()[2]))

            self.camera.show_point(pos_pix)
            print('recorded: ', self.cal_points[-1])
        else:
            print('no pipette found')

    def calibrate(self):
        '''Calibrates the pipette using YOLO object detection and pipette encoders to create a um -> pixels transformation matrix
        '''
        print('calibrating...')
        #convert to int64 for cv2
        cal_points_int = np.array(self.cal_points, dtype=np.int64)
        print(f'cal points: {self.cal_points}')
        ret, mat, inliers = cv2.estimateAffine3D(np.array(cal_points_int)[:,3:6], np.array(cal_points_int)[:,0:3])


        #convert to float64 for cv2
        self.cal_points = np.array(self.cal_points, dtype=np.float64)

        #use least squares to find the 4x4 homogeneous transform matrix
        A = np.zeros((self.cal_points.shape[0] * 3, 12))

        #construct vectorized A matrix
        for i in range(0, self.cal_points.shape[0], 3):
            A[i] = [self.cal_points[i,3], self.cal_points[i,4], self.cal_points[i,5], 1, 0, 0, 0, 0, 0, 0, 0, 0]
            A[i + 1] = [0, 0, 0, 0, self.cal_points[i,3], self.cal_points[i,4], self.cal_points[i,5], 1, 0, 0, 0, 0]
            A[i + 2] = [0, 0, 0, 0, 0, 0, 0, 0, self.cal_points[i,3], self.cal_points[i,4], self.cal_points[i,5], 1]

        #construct vectorized b matrix
        b = np.zeros((self.cal_points.shape[0] * 3, 1))
        for i in range(0, self.cal_points.shape[0], 3):
            b[i] = self.cal_points[i,0]
            b[i + 1] = self.cal_points[i,1]
            b[i + 2] = self.cal_points[i,2]

        #solve least squares
        m2 = np.linalg.inv(A.T @ A) @ A.T @ b
        
        #reshape into 4x4 matrix
        m2 = np.reshape(m2, (3,4))
        print('least squares: ', m2)

        
        print(f'calibration matrix: {mat}')
        self.cal_points = []
        return m2

class PipetteFocusHelper():
    def __init__(self, pipette: Manipulator, camera: Camera):
        self.pipette = pipette
        self.camera = camera
        self.pipetteFocuser : PipetteFocuser = PipetteFocuser()
    
    def focus(self):
        '''Moves the pipette into focus, if it's in the current frame
        '''
        print('focusing...')
        frame = self.camera.get_16bit_image()
        focusLevel = self.pipetteFocuser.get_pipette_focus(frame)
        print(focusLevel)
        for i in range(10):
            if focusLevel == FocusLevels.OUT_OF_FOCUS_DOWN:
                print("moving pipette down")
                self.pipette.relative_move(-50, 2)
            elif focusLevel == FocusLevels.OUT_OF_FOCUS_UP:
                print("moving pipette up")
                self.pipette.relative_move(50, 2)
            elif focusLevel == FocusLevels.IN_FOCUS:
                print('in focus')
                break
            
            self.pipette.wait_until_still()

            frame = self.camera.get_16bit_image()
            focusLevel = self.pipetteFocuser.get_pipette_focus(frame)