'''
"Fake setup" for GUI development on a computer without access to a rig
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.DAQ import FakeDAQ, DAQ
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.devices.camera.qimagingcam import QImagingCam
from holypipette.devices.pressurecontroller import IBBPressureController, FakePressureController, TestPressureController,MoscowPressureController
from holypipette.devices.camera.camera import FakeCamera
from holypipette.devices.camera import FakeCalCamera, FakePipetteManipulator
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


controllerSerial = serial.Serial('COM6')
controller = ScientificaSerialNoEncoder(controllerSerial)

pipetteSerial = serial.Serial('COM3')
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial)
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
camera = PcoCamera()
# camera = moscowQCamera()
# camera = QImagingCam()
# camera = FakeCalCamera(stageManip=controller, pipetteManip=pipetteManip, image_z=100, cellSorterManip=cellSorterManip)
microscope = Microscope(controller, 3)
microscope.up_direction = 1.0

unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

daq = DAQ('cDAQ1Mod1', 'ai0', 'cDaq1Mod4', 'ao0')

# daq = FakeDAQ()
# amplifier = FakeAmplifier()
# pressure = FakePressureController()
amplifier = MultiClampChannel(channel=1)


pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM8', baudrate=9600, timeout=0)
# pressure = IBBPressureController(channel=1, arduinoSerial=pressureControllerSerial)
# pressure = TestPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)
pressure = MoscowPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)