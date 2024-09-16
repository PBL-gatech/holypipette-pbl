import sys
from PyQt5.QtWidgets import QApplication
import traceback

from holypipette.log_utils import setup_logging
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.interface import AutoPatchInterface
from holypipette.interface.pipettes import PipetteInterface
from holypipette.gui.graph import EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, HoldingProtocolGraph
from holypipette.gui.patch import PatchGui

# from setup_IBB_rig import *
# from setup_fake_rig import *
from setup.setup_Moscow_rig import *

setup_logging()  # Log to the standard console as well

def main():
    app = QApplication(sys.argv)

    recording_state_manager = RecordingStateManager()

    pipette_controller = PipetteInterface(stage, microscope, camera, unit, cellSorterManip, cellSorterController)
    patch_controller = AutoPatchInterface(amplifier, daq, pressure, pipette_controller)
    gui = PatchGui(camera, pipette_controller, patch_controller, recording_state_manager)
    graphs = EPhysGraph(daq, pressure, recording_state_manager)
    # graphs.location_on_the_screen()
    graphs.show()

    currentProtocolGraph = CurrentProtocolGraph(daq, recording_state_manager)
    voltageProtocolGraph = VoltageProtocolGraph(daq, recording_state_manager)
    holdingProtocolGraph = HoldingProtocolGraph(daq, recording_state_manager)

    gui.initialize()
    # gui.location_on_the_screen()
    gui.show()
    ret = app.exec_()
    sys.exit(ret)

if __name__ == "__main__":
    main()