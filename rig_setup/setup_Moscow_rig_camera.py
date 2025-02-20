'''
This script sets up the Moscow rig. It is used to set up the devices and their connections. The devices are then used in the main script.
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.DAQ import  DAQ, FakeDAQ
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.devices.pressurecontroller import MoscowPressureController, FakePressureController
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


# set up Camera
camera = PcoCamera()

# set up Pressure Controller
pressure = FakePressureController()


# set up Ephys
daq = FakeDAQ()
amplifier = FakeAmplifier()


# set up movement controllers

controller = FakeManipulator(min=[-240000, 50000, 280000],
                             max=[-230000, 60000, 290000])
pipetteManip = FakeManipulator(min=[0, 0, 0],
                                      max=[4000, 20000, 20000])
stage = ManipulatorUnit(controller, [1, 2])

microscope = Microscope(controller, 3)
microscope.up_direction = 1.0
unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

# set up cell sorter
cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()





