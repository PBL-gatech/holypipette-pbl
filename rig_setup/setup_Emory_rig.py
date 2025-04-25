'''
"Fake setup" for GUI development on a computer without access to a rig
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.DAQ import  NiDAQ
from holypipette.devices.camera.electrocamera import ElectroCamera
from holypipette.devices.pressurecontroller import EmoryPressureController
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


# set up Camera
camera = ElectroCamera()

# set up Pressure Controller
pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM3', baudrate=9600, timeout=0)
pressure = EmoryPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)


# set up Ephys
amplifier = MultiClampChannel(channel=1)
daq = NiDAQ('Dev2', 'ai0', 'Dev2', 'ao0', 'Dev2', 'ai3')

# set up movement controllers
controllerSerial = serial.Serial('COM20',baudrate=38400,timeout=3)
controller = ScientificaSerialNoEncoder(controllerSerial,speed = 200, accel = 10000)
microscope = Microscope(controller, 3,speed=1000,accel=19500)
microscope.up_direction = 1.0

pipetteSerial = serial.Serial('COM19',baudrate=38400,timeout=3)
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial,speed = 1000, accel = 10000)
stage = ManipulatorUnit(controller, [1, 2])
unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

# set up cell sorter
cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()






