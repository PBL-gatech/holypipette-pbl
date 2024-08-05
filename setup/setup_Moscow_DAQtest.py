'''
"Fake setup" for GUI development on a computer without access to a rig specifically for the DAQ
'''

from holypipette.devices.amplifier.multiclamp import MultiClampChannel
from holypipette.devices.amplifier.DAQ import  DAQ
from holypipette.devices.pressurecontroller import FakePressureController
from holypipette.devices.camera import FakeCalCamera
from holypipette.devices.manipulator import *
from holypipette.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip



'''
Manipulator,Camera, and stage setup
'''
controller = FakeManipulator(min=[-240000, 50000, 280000],
                             max=[-230000, 60000, 290000])
pipetteManip = FakeManipulator(min=[0, 0, 0],
                                      max=[4000, 20000, 20000])
stage = ManipulatorUnit(controller, [1, 2])

cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()
camera = FakeCalCamera(stageManip=controller, pipetteManip=pipetteManip, image_z=100, cellSorterManip=cellSorterManip)
microscope = Microscope(controller, 3)
microscope.up_direction = 1.0
unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

'''
DAQ  and Pressure setup
'''

daq = DAQ('cDAQ1Mod1', 'ai0', 'cDaq1Mod4', 'ao0', 'cDaq1Mod1', 'ai3')
amplifier = MultiClampChannel(channel=1)
pressure = FakePressureController()
