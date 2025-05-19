'''
This script sets up the Moscow rig. It is used to set up the devices and their connections. The devices are then used in the main script.
'''
import serial
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.DAQ import FakeDAQ, NiDAQ
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.devices.pressurecontroller import MoscowPressureController, FakePressureController
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


# set up Camera
camera = PcoCamera()

# set up Pressure Controller
# pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
# pressureReaderSerial = serial.Serial(port='COM9', baudrate=9600, timeout=0)
# pressure = MoscowPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)
pressure = FakePressureController()

# set up Ephys
# amplifier = MultiClampChannel(channel=1)
# daq = NiDAQ('cDAQ1Mod1', 'ai0', 'cDaq1Mod4', 'ao0', 'cDaq1Mod1', 'ai3')
daq = FakeDAQ()
amplifier = FakeAmplifier()

# set up movement controllers

controllerSerial = serial.Serial('COM9')
controller = ScientificaSerialNoEncoder(controllerSerial)
microscope = Microscope(controller, 3)
microscope.up_direction = 1.0

pipetteSerial = serial.Serial('COM12')
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial)
stage = ManipulatorUnit(controller, [1, 2])
unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

# set up cell sorter
cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()





