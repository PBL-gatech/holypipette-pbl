'''
"Fake setup" for GUI development on a computer without access to a rig
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.DAQ import FakeDAQ, NiDAQ
from holypipette.devices.camera.pcocamera import PcoCamera

from holypipette.devices.pressurecontroller import  FakePressureController,  MoscowPressureController
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


controllerSerial = serial.Serial('COM6')
controller = ScientificaSerialNoEncoder(controllerSerial)

pipetteSerial = serial.Serial('COM3')
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial)
stage = ManipulatorUnit(controller, [1, 2])



cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()


camera = PcoCamera()

microscope = Microscope(controller, 3)
microscope.up_direction = 1.0

unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

daq = NiDAQ('Dev1', 'ai0', 'Dev1', 'ao0', 'Dev1', 'ai3')
# daq = DAQ('cDAQ1Mod1', 'ai0', 'cDaq1Mod4', 'ao0')

pressure = FakePressureController()
amplifier = MultiClampChannel(channel=1)

pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM9', baudrate=9600, timeout=0)
# pressureReaderSerial = serial.Serial(port='COM8', baudrate=9600, timeout=0)
# pressure = IBBPressureController(channel=1, arduinoSerial=pressureControllerSerial)
# pressure = TestPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)
pressure = MoscowPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)