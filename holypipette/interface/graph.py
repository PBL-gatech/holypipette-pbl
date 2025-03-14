# coding=utf-8
'''
Access and control of Acquisition threads for the DAQ, and Pressure controller
'''
import numpy as np

from holypipette.interface import TaskInterface, command, blocking_command
from holypipette.devices.pressurecontroller.BasePressureController import PressureController
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import DAQ
from PyQt5 import QtCore
import time


__all__ = ['GraphInterface']

class GraphInterface(TaskInterface):
    def __init__(self, amplifier: Amplifier, daq: DAQ, pressure: PressureController, recording_state_manager):
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
        self.execute(self.pressure.set_pressure, argument=pressure)

    @command(category='Pressure',
              description='get pressure setpoint')
    def get_pressure(self):
        return self.pressure.get_pressure()
    @command(category='Pressure',
                description='switch pressure on or off',
                default_arg=False) 
    def set_ATM(self, atm):
        self.execute(self.pressure.set_ATM,argument=atm)
    @command(category='Pressure',
                    description='obtain current pressure state') 
    def get_ATM(self):
            return self.pressure.get_ATM()
        
    @command(category='DAQ',
                description='get last Data from DAQ')
    def get_last_data(self):
        if self.daq.get_last_acquisition() is not None:
            return self.daq.get_last_acquisition()
        else:
            return None
        
    @command(category='DAQ',
                description='obtain acquision mode')
    def getCellMode(self):
        return self.daq.getCellMode()
    
    @command(category='DAQ',
                description=' set the acquisition mode',
                default_arg=False)
    def setCellMode(self, cellMode):
        self.execute(self.daq.setCellMode, argument=cellMode)

    @command(category = 'Amplifier',
                      description='set the Zap duration',
                      default_arg=0.5)
    def set_zap_duration(self, duration):
        self.execute(self.amplifier.set_zap_duration, argument=duration)

    @command(category = 'Amplifier',
                      description='Zap the cell',
                      success_message ='Zap done')

    def zap(self):
        self.execute(self.amplifier.zap)

    



