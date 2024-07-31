from __future__ import print_function
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import traceback
import numpy as np

import logging
import time

from holypipette.utils.FileLogger import FileLogger
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.devices.camera.camera import Camera


__all__ = ['LiveFeedQt']


class LiveFeedQt(QtWidgets.QLabel):
    def __init__(self, camera: Camera, recording_state_manager: RecordingStateManager, image_edit=None, display_edit=None, mouse_handler=None, parent=None):

        super(LiveFeedQt, self).__init__(parent=parent)
        # The image_edit function (does nothing by default) gets the raw
        # unscaled image (i.e. a numpy array), while the display_edit
        # function gets a QPixmap and is meant to draw GUI elements in
        # "display space" (by default, a red cross in the middle of the
        # screen).
        if image_edit is None:
            image_edit = lambda frame: frame
        self.image_edit = image_edit

        if display_edit is None:
            display_edit = lambda img: img
        self.display_edit = display_edit

        self.mouse_handler = mouse_handler
        self.camera = camera
        self.width, self.height = self.camera.width, self.camera.height

        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)

        self.recording_state_manager = recording_state_manager
        self.recorder = FileLogger(recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", isVideo=True, filetype="csv", recorder_filename="camera_frames")

        # Remember the last frame that we displayed, to not unnecessarily
        # process/show the same frame for slow input sources
        self._last_frameno = None
        self._last_edited_frame = None
        
        self.last_frame_time = None
        self.fps = 0

        self.update_image()

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(28) # 30 fps -> 31.5fps
        # timer.start(16) # 60 fps --> but actually 64 fps
        # timer.start(33) # 30 fps --> but avctually 21.5 fps

    def mousePressEvent(self, event):
        # Ignore clicks that are not on the image
        xs = event.x() - self.size().width() * 0.5
        ys = event.y() - self.size().height() * 0.5
        pixmap = self.pixmap()
        if abs(xs) > pixmap.width() * 0.5 or abs(ys) > pixmap.height() * 0.5:
            self.setFocus()
            return

        if self.mouse_handler is not None:
            self.mouse_handler(event)

    def log_frame_rate(self):
    # Calculate and log the frame rate at which images are processed
        current_time = time.time()
        if self.last_frame_time is not None:
            time_diff = current_time - self.last_frame_time
            self.fps = 1.0 / time_diff
            logging.info(f"FPS in LIVEFEED: {self.fps:.2f}")
        self.last_frame_time = current_time

    @QtCore.pyqtSlot()
    def update_image(self):
        try:
            # get last frame from camera
            frameno, frame_time, frame = self.camera.last_frame_data()

            if frame is None:
                return  # Frame acquisition thread has stopped
            
            if self._last_frameno is None or self._last_frameno != frameno:
                frame = self.image_edit(frame)
            
                self._last_edited_frame = frame
                self._last_frameno = frameno
            else:
                # No need to preprocess a frame again if it has not changed
                frame = self._last_edited_frame

            # * Where you place tihs function is important, relative to repeated frames and such. Either you check in this file 
            # * or in the FileLogger file
            self.recorder.write_camera_frames(frame_time.timestamp(), frame, frameno)
            # self.log_frame_rate()
            # print(f"FRAME SHAPE: {frame.shape}")

            if len(frame.shape) == 2:
                # Grayscale image via MicroManager
                if frame.dtype == np.dtype('uint32'):
                    bytesPerLine = self.width*4
                    format = QtGui.QImage.Format_RGB32
                else:
                    bytesPerLine = self.width
                    format = QtGui.QImage.Format_Indexed8
            else:
                # Color image via OpenCV
                bytesPerLine = 3 * self.width
                format = QtGui.QImage.Format_RGB888
            
            # ? So why frame.data and not frame? frame.data points to a memory location. I think both work, but check https://doc.qt.io/qt-6/qimage.html#QImage-5
            q_image = QtGui.QImage(frame.data, self.width, self.height,
                                   bytesPerLine, format)
            # q_image = QtGui.QImage(frame, self.width, self.height,
            #                        bytesPerLine, format)
            
            
            if format == QtGui.QImage.Format_RGB888:
                # OpenCV returns images as 24bit BGR (and not RGB), but there is no
                # direct support for this format in QImage
                q_image = q_image.rgbSwapped()

            pixmap = QtGui.QPixmap.fromImage(q_image)
            size = self.size()
            width, height = size.width(), size.height()
            scaled_pixmap = pixmap.scaled(width, height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            if self.display_edit is not None:
                self.display_edit(scaled_pixmap)
            self.setPixmap(scaled_pixmap)

        except Exception:
            print(traceback.format_exc())
