# coding=utf-8
'''
Access and control of Acquisition threads for the DAQ, and Pressure controller
'''
import numpy as np

from holypipette.interface import TaskInterface, command, blocking_command
from holypipette.devices.pressurecontroller.BasePressureController import PressureController
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import NiDAQ
from .patchConfig import PatchConfig
from PyQt5 import QtCore
import time


__all__ = ['GraphInterface']

class GraphInterface(TaskInterface):
    def __init__(self, amplifier: Amplifier, daq: NiDAQ, pressure: PressureController, recording_state_manager):
        super().__init__()
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.recording_state_manager = recording_state_manager

    @command(category='Pressure', 
              description='obtain current pressure value')
    def get_last_pressure(self):
        return self.pressure.get_last_acquisition()
    @command(category='Pressure',
              description='change pressure setpoint',
              default_arg=0)
    def set_pressure(self, pressure):
        self.execute(self.pressure.set_pressure(pressure))
    @command(category='Pressure',
                description='switch pressure on or off',
                default_arg=False) 
    def set_ATM(self, atm):
        self.execute(self.pressure.set_ATM(atm))
    @command(category='DAQ',
                description='get last Data from DAQ')
    def get_last_data(self):
        if self.daq.get_last_acquisition() is not None:
            return self.daq.get_last_acquisition()
        else:
            return None
    @command(category = 'Amplifier',
                      description='set the Zap duration')
    def set_zap_duration(self, duration):
        self.execute(self.amplifier.set_zap_duration(duration))

    @command(category = 'Amplifier',
                      description='Zap the cell',
                      success_message ='Zap done')
    def zap(self):
        self.execute(self.amplifier.zap())

    



