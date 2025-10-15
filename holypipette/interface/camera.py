from __future__ import print_function
import numpy as np
from PyQt5 import QtCore, QtWidgets
import warnings
import cv2
from numpy import *
from holypipette.interface import TaskInterface, command, blocking_command
import logging
from holypipette.utils.RecordingStateManager import RecordingStateManager


class CameraInterface(TaskInterface):
    updated_exposure = QtCore.pyqtSignal('QString', 'QString')

    def __init__(self, camera, with_tracking=False, status_category='Camera'):
        super().__init__()
        self.camera = camera
        self.with_tracking = with_tracking
        self.recording_state_manager = RecordingStateManager()
        self.status_category = status_category
        self._is_active = False

    def connect(self, main_gui):
        self.updated_exposure.connect(main_gui.set_status_message)
        self.signal_updated_exposure()
        if self.with_tracking:
            main_gui.image_edit_funcs.append(self.show_tracked_objects)
            #main_gui.image_edit_funcs.append(self.show_tracked_paramecium)
            #main_gui.image_edit_funcs.append(self.pipette_contact_detection)

    def signal_updated_exposure(self):
        # Should be called by subclasses that actually support setting the exposure
        if not self._is_active:
            return
        exposure = self.camera.get_exposure()
        if exposure > 0:
            self.updated_exposure.emit(self.status_category, 'Exposure: %.1f ms' % exposure)

    def set_active(self, is_active: bool):
        was_active = self._is_active
        self._is_active = bool(is_active)
        if self._is_active:
            self.signal_updated_exposure()
        elif was_active:
            self.updated_exposure.emit(self.status_category, None)

    def set_status_category(self, category: str) -> None:
        self.status_category = category
        if self._is_active:
            self.signal_updated_exposure()

    def set_camera(self, camera) -> None:
        self.camera = camera
        if self._is_active:
            self.signal_updated_exposure()

    @blocking_command(category='Camera',
                      description='Auto exposure',
                      task_description='Adjusting exposure')
    def auto_exposure(self,args):
        self.camera.auto_exposure()
        self.signal_updated_exposure()

    @command(category='Camera',
             description='Increase exposure time by {:.1f}ms',
             default_arg=2.5)
    def increase_exposure(self, increase):
        self.camera.change_exposure(increase)
        self.signal_updated_exposure()
    @command(category='Camera',
                description='Set exposure time to {:.1f}ms',
                default_arg=2.5)
    def set_exposure(self, exposure):     
        currexpos = self.camera.get_exposure()
        change = exposure - currexpos
        if change > 0:
            self.increase_exposure(change)
        elif change < 0:
            self.decrease_exposure(-change)
        # logging.info('Current exposure time is: {}'.format(currexpos))
        # logging.info("difference is: {}".format(change))
        logging.info('New exposure time is: {}'.format(exposure))
        # self.camera.set_exposure(exposure)
        self.signal_updated_exposure()
    
    @command(category='Camera',
             description='Normalize the image',
             )
    def normalize(self, param=None):
        self.camera.normalize()

    @command(category='Camera',
             description='AutoNormalize the image',
             )
    def autonormalize(self, state):
        # if state: 
        #     print("AutoNormalizing")
        # else:
        #     print("Not AutoNormalizing")
        self.camera.autonormalize(state)

    @command(category='Camera',
             description='Decrease exposure time by {:.1f}ms',
             default_arg=2.5)
    def decrease_exposure(self, decrease):
        self.camera.change_exposure(-decrease)
        self.signal_updated_exposure()

