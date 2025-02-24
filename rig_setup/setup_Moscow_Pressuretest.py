'''
Setup for Moscow Pressuretest
'''
import serial
from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.amplifier import FakeAmplifier
from holypipette.devices.amplifier.DAQ import FakeDAQ, DAQ
from holypipette.devices.camera.pcocamera import PcoCamera
# from holypipette.devices.camera.qimagingcam import QImagingCam
from holypipette.devices.pressurecontroller import IBBPressureController, FakePressureController, TestPressureController, MoscowPressureController
from holypipette.devices.camera.camera import FakeCamera
from holypipette.devices.camera import FakeCalCamera, FakePipetteManipulator
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip


controller = FakeManipulator(min=[-240000, 50000, 280000],
                             max=[-230000, 60000, 290000])
pipetteManip = FakeManipulator(min=[0, 0, 0],
                                      max=[4000, 20000, 20000])
stage = ManipulatorUnit(controller, [1, 2])

cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()


camera = FakeCalCamera(stageManip=controller, pipetteManip=pipetteManip, image_z=100, cellSorterManip=cellSorterManip)
microscope = Microscope(controller, 3,speed=10000,accel=19500)
microscope.up_direction = 1.0

unit = ManipulatorUnit(pipetteManip, [1, 2, 3])


daq = FakeDAQ()
amplifier = FakeAmplifier()


pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM3', baudrate=9600, timeout=0)
pressure = MoscowPressureController(channel=1, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)