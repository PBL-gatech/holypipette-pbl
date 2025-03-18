import time
import cv2
import numpy as np
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.manipulator import Manipulator
from holypipette.devices.camera import Camera
from holypipette.deepLearning.pipetteFinder import PipetteFinder
from holypipette.deepLearning.pipetteFocuser import PipetteFocuser
from threading import Thread
import logging

class PipetteCalHelper():
    """
    A helper class to aid with 2D pipette calibration.

    Overview:
      - Calibration points are collected relative to the stage's base reference.
      - Only the (x, y) components are used, ignoring the z-axis.
      - Ten calibration points are gathered so that the field of view is well‐sampled.
      - A 2D affine transformation is computed from pipette encoder (x, y) positions to
        image (x, y) positions (with stage reference subtracted), using OpenCV.
      - The 2×3 matrix is then embedded in a 3×4 homogeneous transformation matrix.
    
    Summary:
      1. Record calibration points as tuples: (image_x, image_y, encoder_x, encoder_y).
      2. Subtract the stage's reference position from the pipette’s detected image location.
      3. Use cv2.estimateAffine2D on the collected points.
      4. Convert the resulting 2×3 matrix into a 3×4 matrix for homogeneous representation.
    """

    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 1000

    def __init__(self, pipette: Manipulator, microscope: Microscope, camera: Camera, calibrated_stage):
        self.pipette: Manipulator = pipette
        self.microscope: Microscope = microscope
        self.camera = camera
        self.pipetteFinder: PipetteFinder = PipetteFinder()
        self.calibrated_stage = calibrated_stage
        # Each calibration point will be a tuple:
        #   (image_x, image_y, encoder_x, encoder_y)
        self.cal_points = []

    def collect_cal_points(self, num_points=10, xy_step=3, max_retries=5):
        """
        Collects 'num_points' calibration points.
        
        For each point:
         - The pipette’s current (x, y) image position (with the stage's reference subtracted)
           is paired with the pipette’s encoder (x, y) coordinates.
         - A small move (with added randomness) is commanded between points so that the
           calibration data covers a larger area.
        """
        self.cal_points = []
        for i in range(num_points):
            self._record_point_with_retries(max_retries)
            if i < num_points - 1:
                # Move pipette slightly in the plane to spread out calibration data.
                dx = xy_step + np.random.uniform(-1, 1)
                dy = xy_step + np.random.uniform(-1, 1)
                self.pipette.relative_move([dx, dy, 0])
                self.pipette.wait_until_still()
        return len(self.cal_points) >= num_points

    def _record_point_with_retries(self, max_retries=5):
        """
        Attempt to record a calibration point. If no pipette is detected in the image,
        jitter the pipette and retry.
        """
        for _ in range(max_retries):
            before = len(self.cal_points)
            self.record_cal_point()
            if len(self.cal_points) > before:
                return True
            # Jitter the pipette if detection failed.
            self.pipette.relative_move([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                0
            ])
            self.pipette.wait_until_still()
            time.sleep(0.5)
        return False

    def record_cal_point(self):
        """
        Records a calibration point as follows:
         - Retrieves the current camera frame.
         - Uses the pipetteFinder to detect the pipette’s (x, y) position.
         - Subtracts the stage’s reference position so that the result is in the stage’s coordinate system.
         - Pairs the (x, y) image position with the pipette’s encoder (x, y) coordinates.
        """
        # Get the latest frame from the camera.
        _, _, _, frame = self.camera.raw_frame_queue[0]
        pos_pix = self.pipetteFinder.find_pipette(frame)
        if pos_pix is not None:
            # Optionally, display the detected pipette on the frame.
            frame = cv2.circle(frame, pos_pix, 10, 0, 2)
            # add the stage's reference position (assumed to be in pixels).
            stage_pos_pix = self.calibrated_stage.reference_position()
            image_x = pos_pix[0] - stage_pos_pix[0]
            image_y = pos_pix[1] - stage_pos_pix[1]
            # Retrieve the pipette's encoder (x, y) positions; ignore z.
            encoder_pos = self.pipette.position()  # assumed format: [x, y, z]
            encoder_x = encoder_pos[0]
            encoder_y = encoder_pos[1]
            self.cal_points.append((image_x, image_y, encoder_x, encoder_y))
            self.camera.show_point(pos_pix)
            print("Recorded calibration point:", self.cal_points[-1])
        else:
            print("No pipette found in current frame.")

    def calibrate(self):
        """
        Computes a 2D affine transformation that maps pipette encoder (x, y)
        positions to image (x, y) positions (already relative to the stage's base).
        
        It then embeds the resulting 2×3 matrix into a 3×4 homogeneous transformation matrix.
        
        Returns:
            A 3×4 transformation matrix.
            (The calibrated unit's finish_calibration routine may then convert this
             into a full 4×4 matrix as needed.)
        """
        if len(self.cal_points) < 3:
            print("Not enough calibration points for affine transformation.")
            return None

        # Prepare the data arrays (each is N×2):
        # encoder_points: [ [encoder_x, encoder_y], ... ]
        # image_points:   [ [image_x, image_y], ... ]
        encoder_points = np.array([[pt[2], pt[3]] for pt in self.cal_points], dtype=np.float64)
        image_points   = np.array([[pt[0], pt[1]] for pt in self.cal_points], dtype=np.float64)

        # Compute the 2D affine transformation (a 2×3 matrix)
        M2x3, inliers = cv2.estimateAffine2D(encoder_points, image_points)
        if M2x3 is None:
            print("Failed to compute affine transformation.")
            return None

        print("Raw 2D affine transformation matrix (2×3):")
        print(M2x3)

        # Convert the 2×3 matrix to a 3×4 homogeneous transformation matrix.
        # We assume the pipette moves only in x and y (z=0).
        # The resulting matrix has the form:
        #   [ A  B  0  t_x ]
        #   [ C  D  0  t_y ]
        #   [ 0  0  1   0  ]
        mat3x4 = np.zeros((3, 4), dtype=np.float64)
        mat3x4[0, 0:2] = M2x3[0, 0:2]
        mat3x4[0, 3]   = M2x3[0, 2]
        mat3x4[1, 0:2] = M2x3[1, 0:2]
        mat3x4[1, 3]   = M2x3[1, 2]
        mat3x4[2, 2]   = 1.0

        print("Converted 3×4 homogeneous transformation matrix:")
        print(mat3x4)

        # Save the calibration matrix for later use (e.g., for centering the pipette).
        self.calibration_matrix = mat3x4
        # Clear the calibration points after computing the matrix.
        self.cal_points = []
        return mat3x4

class PipetteFocusHelper():
    def __init__(self, pipette: Manipulator, camera: Camera):
        self.pipette = pipette
        self.camera = camera
        self.pipetteFocuser: PipetteFocuser = PipetteFocuser()
    
    def focus(self):
        """
        Adjusts the pipette focus by capturing an image,
        predicting the defocus value, and commanding a relative move
        using that value.
        """
        # logging.info("Focusing pipette...")
        frame = self.camera.get_16bit_image()
        # convert to 8-bit for display
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        defocus_value = self.pipetteFocuser.get_pipette_focus_value(frame)
        # print(f"Defocus value: {defocus_value:.2f} µm")
        
        # Use the defocus value directly in the relative move command.
        self.pipette.relative_move([0, 0, -defocus_value])
        self.pipette.wait_until_still()