'''
"Fake setup" for GUI development on a computer without access to a rig
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.DAQ import FakeDAQ, DAQ
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.devices.camera.qimagingcam import QImagingCam
from holypipette.devices.pressurecontroller import IBBPressureController, FakePressureController
from holypipette.devices.pressurecontroller.MoscowPressureController import MoscowPressureController
from holypipette.devices.camera.camera import FakeCamera
from holypipette.devices.camera import FakeCalCamera, FakePipetteManipulator
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


ser8 = serial.Serial('COM8')
if ser8.isOpen():
    print("CLOSING 8")
    ser8.close()
ser5 = serial.Serial('COM5') 
if ser5.isOpen():
    print("CLOSING 5")
    ser5.close()
ser3 = serial.Serial('COM3')
if ser3.isOpen():
    print("CLOSING 3")
    ser3.close()
ser6 = serial.Serial('COM6') 
if ser6.isOpen():
    print("CLOSING 6")
    ser6.close()

# controllerSerial = serial.Serial('COM6')
# controller = ScientificaSerialNoEncoder(controllerSerial)

# pipetteSerial = serial.Serial('COM3')
# pipetteManip = ScientificaSerialNoEncoder(pipetteSerial)
# stage = ManipulatorUnit(controller, [1, 2])

controller = FakeManipulator(min=[-240000, 50000, 280000],
                             max=[-230000, 60000, 290000])
pipetteManip = FakeManipulator(min=[0, 0, 0],
                                      max=[4000, 20000, 20000])
stage = ManipulatorUnit(controller, [1, 2])

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
pressure = FakePressureController()
amplifier = MultiClampChannel(channel=1)
# pressureSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
# pressure = IBBPressureController(channel=1, arduinoSerial=pressureSerial)
# pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
# pressureReaderSerial = serial.Serial(port='COM8', baudrate=9600, timeout=0)
# pressure = MoscowPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)