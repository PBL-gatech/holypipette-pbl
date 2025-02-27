'''
"Fake setup" for GUI development on a computer without access to a rig
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.DAQ import FakeDAQ, DAQ
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.devices.camera.electrocamera import ElectroCamera

from holypipette.devices.pressurecontroller import FakePressureController, EmoryPressureController,MoscowPressureController
from holypipette.devices.camera.camera import FakeCamera
from holypipette.devices.camera import FakeCalCamera, FakePipetteManipulator
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


controllerSerial = serial.Serial('COM20',baudrate=38400,timeout=3)
controller = ScientificaSerialNoEncoder(controllerSerial,speed = 4000, accel = 19500)

pipetteSerial = serial.Serial('COM19',baudrate=38400,timeout=3)
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial,speed = 1000, accel = 19500)
stage = ManipulatorUnit(controller, [1, 2])

# controller = FakeManipulator(min=[-240000, 50000, 280000],
#                              max=[-230000, 60000, 290000])
# pipetteManip = FakeManipulator(min=[0, 0, 0],
#                                       max=[4000, 20000, 20000])
# stage = ManipulatorUnit(controller, [1, 2])

cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()

# pipetteManip.x = [200, 300, 400] # start with pipette in frame
# controller.x = [-235000, 55000, 285000]
camera = ElectroCamera()

# camera = FakeCalCamera(stageManip=controller, pipetteManip=pipetteManip, image_z=100, cellSorterManip=cellSorterManip)
microscope = Microscope(controller, 3,speed=10000,accel=19500)
microscope.up_direction = 1.0

unit = ManipulatorUnit(pipetteManip, [1, 2, 3])


daq = DAQ('Dev2', 'ai0', 'Dev2', 'ao0', 'Dev2', 'ai3')

# daq = FakeDAQ()
# amplifier = FakeAmplifier()
amplifier = MultiClampChannel(channel=1)

pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM3', baudrate=9600, timeout=0)
pressure = EmoryPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)
# pressure = FakePressureController()