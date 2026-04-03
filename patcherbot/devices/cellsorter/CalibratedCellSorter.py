import cv2
import numpy as np
import time

from patcherbot.devices.camera.camera import Camera
from patcherbot.devices.cellsorter import CellSorterManip
from patcherbot.devices.cellsorter import CellSorterController
from patcherbot.controller import TaskController

__all__ = ['CalibratedCellSorter']

class CalibratedCellSorter(TaskController):
    """
    Controller for a calibrated cell sorter system that integrates a manipulator,
    pressure/valve controller, stage, microscope, and camera.
    """
    def __init__(self, cellsorterManip: CellSorterManip, cellSorterController: CellSorterController, stage, microscope, camera: Camera):
        """
        Initialize the calibrated cell sorter controller.

        args:
            cellsorterManip (CellSorterManip): Manipulator controlling cell sorter position.
            cellSorterController (CellSorterController): Controller for valves and LEDs.
            stage: Motorized stage object used for positioning.
            microscope: Microscope controller for Z positioning.
            camera (Camera): Camera used for imaging and calibration.
        """
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
        """
        Activate suction for a specified duration.

        args:
            duration (float): Time in seconds to apply suction.
        """
        self.cellsorterController.open_valve_for_time(1, duration)

    def pulse_pressure(self, duration):
        """
        Activate pressure for a specified duration.

        args:
            duration (float): Time in seconds to apply pressure.
        """
        self.cellsorterController.open_valve_for_time(2, duration)
    
    def position(self):
        """
        Get the current position of the cell sorter.

        returns:
            float: Current position value from the manipulator.
        """
        return self.cellsorterManip.position()
    
    def absolute_move(self, position, velocity=None):
        """
        Move the cell sorter to an absolute position.

        args:
            position (float): Target position.
            velocity (float, optional): Movement speed.
        """
        print('velocity', velocity)
        self.cellsorterManip.absolute_move(position, velocity=velocity)
    
    def relative_move(self, position, velocity=None):
        """
        Move the cell sorter relative to its current position.

        args:
            position (float): Relative displacement.
            velocity (float, optional): Movement speed.
        """
        self.cellsorterManip.relative_move(position, velocity=velocity)

    def set_led_status(self, status, ring=None):
        """
        Set the LED status of the cell sorter.

        args:
            status: LED state (implementation-defined).
            ring (optional): LED ring identifier.
        """
        self.cellsorterController.set_led(status)
        if ring != None:
            self.cellsorterController.set_led_ring(ring)

    def get_led_status(self):
        """
        Get the current LED status.

        returns:
            Any: Current LED state from the controller.
        """
        return self.cellsorterController.get_led()

    def set_led_ring_enabled(self, status, ring=1):
        """
        Enable or disable the LED ring.

        args:
            status: Desired LED state.
            ring (int, optional): Ring identifier. Defaults to 1.
        """
        self.set_led_status(status, ring)


    def calibrate(self):
        """
        Calibrate the cell sorter using image-based detection.

        Detects the pipette location using a Hough Circle transform,
        determines its pixel offset, and records the Z position of
        the coverslip.

        raises:
            Exception: If the stage is not calibrated.
            Exception: If no circles or multiple circles are detected.
        """
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
        self.camera.show_circle([int(x), int(y)], radius=int(r), color=(0, 0, 255), show_center=True)

        self.pipetteOffsetPix = np.array([x, y])
        self.coverslipZPos = self.position()

        print("Pipette offset: ", self.pipetteOffsetPix)
        print("Coverslip Z position: ", self.coverslipZPos)

        #draw a circle on the image
        self.calibrated = True

    def raise_pipette(self):
        """Raise the pipette away from the sample."""
        self.cellsorterManip.set_max_speed(self.fastMoveSpeed)
        self.absolute_move(self.position() + 5)
        self.cellsorterManip.wait_until_still()
    
    def center_cellsorter_on_point(self, point): #x,y in pixels, z in stage units
        """
        Center the cell sorter on a given image point.

        args:
            point (tuple): (x, y, z) target where x and y are pixel coordinates,
                        and z is in stage units.

        raises:
            Exception: If stage or cell sorter is not calibrated.
            Exception: If microscope focal plane is not set.
        """
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

