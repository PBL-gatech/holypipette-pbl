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
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Couldn't open camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.start_acquisition()

    def raw_snap(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def set_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def get_exposure(self):
        return self.cap.get(cv2.CAP_PROP_EXPOSURE)

    def get_frame_rate(self):
        # Return the camera frame rate if available.
        return self.cap.get(cv2.CAP_PROP_FPS)

    def close(self):
        if self.cap:
            self.cap.release()
        super().close()

def main():
    cam = PipetteCamera()
    WIN_NAME = "Pipette Camera (press q to quit)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    while True:
        # snap() returns (raw, processed) image; we display the processed image.
        try:
            _, frame = cam.snap()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            break
        cv2.imshow(WIN_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()