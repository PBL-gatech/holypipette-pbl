import cv2
import numpy as np
import time

from holypipette.devices.camera.camera import Camera
from holypipette.devices.cellsorter import CellSorterManip
from holypipette.devices.cellsorter import CellSorterController
from holypipette.controller import TaskController

__all__ = ['CalibratedCellSorter']

class CalibratedCellSorter(TaskController):
    def __init__(self, cellsorterManip: CellSorterManip, cellSorterController: CellSorterController, stage, microscope, camera: Camera):
        self.cellsorterManip = cellsorterManip
        self.cellsorterController = cellSorterController
        self.stage = stage
        self.camera = camera
        self.microscope = microscope
        self.calibrated = False
        self.pipetteOffsetPix = None
        self.coverslipZPos = None
        self.slowMoveRegion = 0.5 #mm
        self.slowMoveSpeed = 0.2
        self.fastMoveSpeed = 5

    def pulse_suction(self, duration):
        self.cellsorterController.open_valve_for_time(1, duration)

    def pulse_pressure(self, duration):
        self.cellsorterController.open_valve_for_time(2, duration)
    
    def position(self):
        return self.cellsorterManip.position()
    
    def absolute_move(self, position, velocity=None):
        print('velocity', velocity)
        self.cellsorterManip.absolute_move(position, velocity=velocity)
    
    def relative_move(self, position, velocity=None):
        self.cellsorterManip.relative_move(position, velocity=velocity)

    def set_led_status(self, status, ring=None):
        self.cellsorterController.set_led(status)
        if ring != None:
            self.cellsorterController.set_led_ring(ring)

    def get_led_status(self):
        return self.cellsorterController.get_led()


    def calibrate(self):
        #make sure stage is calibrated
        if not self.stage.calibrated:
            raise Exception("Stage is not calibrated")
        
        #grab latest image
        img, _ = self.camera.snap()

        #convert to grayscale if needed
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #find the center of the tube via hough transform
        img = cv2.medianBlur(img, 19)
        #add channel dimension to make it 3D
        img = img[:, :, np.newaxis]
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=100, maxRadius=300)
        
        if circles is None or len(circles) == 0:
            raise Exception("No circles found in image")
        
        if len(circles) > 1:
            raise Exception("Multiple circles found in image")
        
        #get the center of the pipette
        x, y, r = circles[0][0]
        self.camera.show_point([int(x), int(y)], radius=int(r), color=(0, 0, 255), show_center=True)

        self.pipetteOffsetPix = np.array([x, y])
        self.coverslipZPos = self.position()

        print("Pipette offset: ", self.pipetteOffsetPix)
        print("Coverslip Z position: ", self.coverslipZPos)

        #draw a circle on the image
        self.calibrated = True

    def raise_pipette(self):
        self.cellsorterManip.set_max_speed(self.fastMoveSpeed)
        self.absolute_move(self.position() + 5)
        self.cellsorterManip.wait_until_still()
    
    def center_cellsorter_on_point(self, point): #x,y in pixels, z in stage units
        x, y, z = point
        if not self.stage.calibrated:
            raise Exception("Stage is not calibrated")
        if not self.calibrated or self.pipetteOffsetPix is None or self.coverslipZPos is None:
            raise Exception("Cell Sorter is not calibrated")
        if self.microscope.floor_Z is None:
            raise Exception("Cell Plane not set")

        self.raise_pipette()     

        #move the stage such that it's centered in x, y, put cells in focus
        self.microscope.absolute_move(self.microscope.floor_Z)
        self.stage.reference_move(np.array([x + self.pipetteOffsetPix[0], y + self.pipetteOffsetPix[1]]))
        self.stage.wait_until_reached(np.array([x + self.pipetteOffsetPix[0], y + self.pipetteOffsetPix[1]]))
        time.sleep(0.5)

        print("moving cellsorter to cell plane")
        #move the cellsorter to the cell plane
        currentPos = self.position()
        if abs(currentPos - self.coverslipZPos) < self.slowMoveRegion:
            #move slowly to setpoint
            self.absolute_move(self.coverslipZPos, self.slowMoveSpeed)
            self.cellsorterManip.wait_until_still()
        else:
            #move quickly to slowMoveRegion away from setpoint
            initSetpoint = self.coverslipZPos + np.sign(self.slowMoveRegion - self.coverslipZPos) * self.slowMoveRegion
            self.absolute_move(initSetpoint, self.fastMoveSpeed)
            self.cellsorterManip.wait_until_still()
            #move slowly to setpoint
            self.absolute_move(self.coverslipZPos, self.slowMoveSpeed)
            self.cellsorterManip.wait_until_still()

        self.absolute_move(self.coverslipZPos)

