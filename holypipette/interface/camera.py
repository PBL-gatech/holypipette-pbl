from __future__ import print_function
import numpy as np
from PyQt5 import QtCore, QtWidgets
import warnings
import cv2
from numpy import *
from holypipette.interface import TaskInterface, command, blocking_command


class CameraInterface(TaskInterface):
    updated_exposure = QtCore.pyqtSignal('QString', 'QString')

    def __init__(self, camera, with_tracking=False):
        super(CameraInterface, self).__init__()
        self.camera = camera
        self.with_tracking = with_tracking

    def connect(self, main_gui):
        self.updated_exposure.connect(main_gui.set_status_message)
        self.signal_updated_exposure()
        if self.with_tracking:
            main_gui.image_edit_funcs.append(self.show_tracked_objects)
            #main_gui.image_edit_funcs.append(self.show_tracked_paramecium)
            #main_gui.image_edit_funcs.append(self.pipette_contact_detection)

    def signal_updated_exposure(self):
        # Should be called by subclasses that actually support setting the exposure
        exposure = self.camera.get_exposure()
        if exposure > 0:
            self.updated_exposure.emit('Camera', 'Exposure: %.1fms' % exposure)

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
             description='Normalize the image',
             )
    def normalize(self, param=None):
        self.camera.normalize()

    @command(category='Camera',
             description='Decrease exposure time by {:.1f}ms',
             default_arg=2.5)
    def decrease_exposure(self, decrease):
        self.camera.change_exposure(-decrease)
        self.signal_updated_exposure()

    # @command(category='Camera',
    #          description='Save the current image to a file')
    # def save_image(self):
    #     # try:
    #     #     from PIL import Image
    #     # except ImportError:
    #     #     self.error('Saving images needs the PIL or Pillow module')
    #     #     return
    #     frame, _ = self.camera.snap()
    #     fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Save image',
    #                                                      filter='Images (*.png, *.tiff)',
    #                                                      options=QtWidgets.QFileDialog.DontUseNativeDialog)
    #     if len(fname):
    #         img = Image.fromarray(frame)
    #         try:
    #             img.save(fname)
    #         except (KeyError, IOError):
    #             self.exception('Saving image as "%s" failed.' % fname)