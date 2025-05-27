# patch_gui.py
import faulthandler
faulthandler.enable()
# faulthandler.dump_traceback_later(5)

import atexit
import sys
from PyQt5.QtWidgets import QApplication
import traceback
from holypipette.utils.exception_handler import set_global_exception_hook

# Set the global exception hook
set_global_exception_hook()


from holypipette.utils.log_utils import setup_logging
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.interface import AutoPatchInterface
from holypipette.interface.pipettes import PipetteInterface
from holypipette.interface.graph import GraphInterface
from holypipette.gui.graph import EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, HoldingProtocolGraph
from holypipette.gui.patch import PatchGui



from rig_setup.setup_Moscow_rig import *  
# from rig_setup.setup_fake_rig import * 
# from rig_setup.setup_Moscow_rig_camera import *    
# from rig_setup.setup_Moscow_Pressuretest import *

setup_logging()  # Log to the standard console as well

def main():
    app = QApplication(sys.argv)

    recording_state_manager = RecordingStateManager()

    pipette_controller = PipetteInterface(stage, microscope, camera, unit, cellSorterManip, cellSorterController)
    patch_controller = AutoPatchInterface(amplifier, daq, pressure, pipette_controller, recording_state_manager)
    graph_interface = GraphInterface(amplifier, daq, pressure, recording_state_manager)
    gui = PatchGui(camera, pipette_controller, patch_controller, recording_state_manager)
    graphs = EPhysGraph(graph_interface, recording_state_manager)
    # graphs.location_on_the_screen()
    graphs.show()

    currentProtocolGraph = CurrentProtocolGraph(graph_interface, recording_state_manager)
    voltageProtocolGraph = VoltageProtocolGraph(graph_interface, recording_state_manager)
    holdingProtocolGraph = HoldingProtocolGraph(graph_interface, recording_state_manager)


    gui.initialize()
    gui.show()
    ret = app.exec_()
    sys.exit(ret)

if __name__ == "__main__":
    main()