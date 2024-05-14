import sys

import PyQt5.QtWidgets as QtWidgets
import traceback
# from PyQt5 import QtWidgets

from holypipette.log_utils import console_logger
from holypipette.interface import AutoPatchInterface
from holypipette.interface.pipettes import PipetteInterface
from holypipette.gui import PatchGui, EPhysGraph, CurrentProtocolGraph


# from setup_IBB_rig import *
# from setup_fake_rig import *
from setup_Moscow_rig import *

console_logger()  # Log to the standard console as well

app = QtWidgets.QApplication(sys.argv)

pipette_controller = PipetteInterface(stage, microscope, camera, unit, cellSorterManip, cellSorterController)
patch_controller = AutoPatchInterface(amplifier, daq, pressure, pipette_controller)
gui = PatchGui(camera, pipette_controller, patch_controller)
graphs = EPhysGraph(daq, pressure)
graphs.show()
currentProtocolGraph = CurrentProtocolGraph(daq)


gui.initialize()
gui.show()
ret = app.exec_()

sys.exit(ret)
