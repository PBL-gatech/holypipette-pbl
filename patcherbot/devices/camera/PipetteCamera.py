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
    def __init__(self, device_index=0, width=480, height=480):
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
        if self.cap is None:
            raise RuntimeError("Camera has not been initialized.")
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        self.currExposure = float(value)

    def get_exposure(self) -> float:
        if self.cap is None:
            raise RuntimeError("Camera has not been initialized.")
        exposure = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        self.currExposure = exposure
        return exposure

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        super().close()

    def reset(self) -> None:
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
        if img is None:
            img = self.get_16bit_image()
        if img is None:
            return
        if not self.auto_normalize:
            print("NORMALIZING")
        self.lowerBound = int(img.min())
        self.upperBound = int(img.max())

    def autonormalize(self, flag=None):
        if flag is None:
            flag = not self.auto_normalize
        self.auto_normalize = bool(flag)
        return self.auto_normalize

    def get_frame_no(self) -> int:
        return self.frameno

    def get_16bit_image(self):
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
        img = self.get_16bit_image()
        if img is None:
            return None

        if self.auto_normalize:
            self.normalize(img)

        span = max(self.upperBound - self.lowerBound, 1)
        normalized = np.clip((img.astype(np.float32) - self.lowerBound) / span * 255, 0, 255)
        return normalized.astype(np.uint8)

    def get_frame_rate(self):
        if self.fps:
            return self.fps
        if self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)


