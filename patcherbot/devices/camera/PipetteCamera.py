'''
Camera class for the Pipette USB camera using OpenCV

This camera class is designed to work with USB cameras that are compatible with OpenCV's VideoCapture.

Usage:
    camera = PipetteCamera(device_index=0, width=640, height=480)
    camera.start_acquisition()
    while True:
        raw, frame = camera.snap()
        # process the frame
    camera.stop_acquisition()
'''

import cv2
import time
import numpy as np

from .camera import Camera


class PipetteCamera(Camera):
    """
    Camera class for a USB pipette camera using OpenCV.

    This class provides continuous acquisition, frame normalization, and
    exposure control for USB cameras compatible with OpenCV's VideoCapture.
    """
    def __init__(self, device_index=0, width=480, height=480):
        """
        Initialize a USB pipette camera using OpenCV.

        Args:
            device_index (int): Index of the camera device (default 0).
            width (int): Desired frame width in pixels (default 480).
            height (int): Desired frame height in pixels (default 480).

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        super().__init__()
        self.device_index = device_index
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Couldn't open camera.")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = actual_width if actual_width else width
        self.height = actual_height if actual_height else height

        self.auto_normalize = False
        self.frameno = 0
        self.currExposure = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        self.upperBound = 255
        self.lowerBound = 0
        self.lastFrame = None
        self._last_bgr_frame = None
        self.last_frame_time = None
        self.fps = 0.0

        try:
            self.normalize()
        except RuntimeError:
            # Camera may need a moment before delivering frames
            self.upperBound = 255
            self.lowerBound = 0

        self.start_acquisition()

    def set_exposure(self, value: float) -> None:
        """
        Set the camera's exposure time.

        Args:
            value (float): Exposure time in camera units (OpenCV-specific).

        Raises:
            RuntimeError: If the camera has not been initialized.
        """
        if self.cap is None:
            raise RuntimeError("Camera has not been initialized.")
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        self.currExposure = float(value)

    def get_exposure(self) -> float:
        """
        Get the current exposure time of the camera.

        Returns:
            float: Current exposure time.

        Raises:
            RuntimeError: If the camera has not been initialized.
        """
        if self.cap is None:
            raise RuntimeError("Camera has not been initialized.")
        exposure = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        self.currExposure = exposure
        return exposure

    def close(self) -> None:
        """
        Release the camera resources and stop acquisition.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        super().close()

    def reset(self) -> None:
        """
        Reset the camera by releasing and reopening it with the same settings.

        Raises:
            RuntimeError: If the camera cannot be reopened.
        """
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Couldn't reopen camera.")

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.frameno = 0
        self.lastFrame = None
        self._last_bgr_frame = None
        self.last_frame_time = None
        self.fps = 0.0
        self.currExposure = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))

        try:
            self.normalize()
        except RuntimeError:
            self.upperBound = 255
            self.lowerBound = 0

    def normalize(self, img=None) -> None:
        """
        Normalize the camera image by setting lower and upper bounds.

        Args:
            img (np.ndarray, optional): Image to normalize. If None, the latest 16-bit frame is used.

        Notes:
            Updates `self.lowerBound` and `self.upperBound`.
            Skips normalization if `img` is None and no previous frame is available.
        """
        if img is None:
            img = self.get_16bit_image()
        if img is None:
            return
        if not self.auto_normalize:
            print("NORMALIZING")
        self.lowerBound = int(img.min())
        self.upperBound = int(img.max())

    def autonormalize(self, flag=None):
        """
        Enable or toggle automatic normalization.

        Args:
            flag (bool, optional): True to enable, False to disable. If None, toggles the current state.

        Returns:
            bool: The new state of `auto_normalize`.
        """
        if flag is None:
            flag = not self.auto_normalize
        self.auto_normalize = bool(flag)
        return self.auto_normalize

    def get_frame_no(self) -> int:
        """
        Get the current frame number.

        Returns:
            int: Frame counter since camera acquisition started.
        """
        return self.frameno

    def get_16bit_image(self):
        """
        Retrieve the latest 16-bit image from the camera.

        Returns:
            np.ndarray: Latest image as 16-bit array.

        Raises:
            RuntimeError: If the camera is not initialized or frame capture fails.
        """
        if self.cap is None:
            raise RuntimeError("Camera has not been initialized.")
        ret, frame = self.cap.read()
        if not ret:
            if self.lastFrame is not None:
                return self.lastFrame
            raise RuntimeError("Failed to capture frame")

        self._last_bgr_frame = frame
        self.frameno += 1

        now = time.time()
        if self.last_frame_time is not None:
            elapsed = max(now - self.last_frame_time, 1e-6)
            self.fps = 1.0 / elapsed
        self.last_frame_time = now

        frame_16 = frame.astype(np.uint16) * 257
        self.lastFrame = frame_16
        return frame_16

    def raw_snap(self):
        """
        Retrieve the current image as 8-bit color with normalization.

        Returns:
            np.ndarray: Normalized 8-bit image.

        Notes:
            Applies `auto_normalize` if enabled.
        """
        img = self.get_16bit_image()
        if img is None:
            return None

        if self.auto_normalize:
            self.normalize(img)

        span = max(self.upperBound - self.lowerBound, 1)
        normalized = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255)
        return normalized.astype(np.uint8)

    def get_frame_rate(self):
        """
        Get the current frame rate of the camera.

        Returns:
            float: Frames per second (FPS). Returns 0 if the camera is not initialized.
        """
        if self.fps:
            return self.fps
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)


