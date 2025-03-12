# patch_gui.py
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(5)

import atexit
import sys
from PyQt5.QtWidgets import QApplication
import traceback
from holypipette.exception_handler import set_global_exception_hook

# Set the global exception hook
set_global_exception_hook()


from holypipette.log_utils import setup_logging
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.interface import AutoPatchInterface
from holypipette.interface.pipettes import PipetteInterface
from holypipette.gui.graph import EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, HoldingProtocolGraph
from holypipette.gui.patch import PatchGui
# from experiments.Analysis.DatasetBuilder import DatasetBuilder



# from rig_setup.setup_Emory_rig import * 
from rig_setup.setup_fake_rig import *
# from setup.setup_Moscow_Pressuretest import *

setup_logging()  # Log to the standard console as well

def main():
    app = QApplication(sys.argv)

    recording_state_manager = RecordingStateManager()

    pipette_controller = PipetteInterface(stage, microscope, camera, unit, cellSorterManip, cellSorterController)
    patch_controller = AutoPatchInterface(amplifier, daq, pressure, pipette_controller)
    gui = PatchGui(camera, pipette_controller, patch_controller, recording_state_manager)
    graphs = EPhysGraph(amplifier,daq, pressure, recording_state_manager)
    # graphs.location_on_the_screen()
    graphs.show()

    currentProtocolGraph = CurrentProtocolGraph(daq, recording_state_manager)
    voltageProtocolGraph = VoltageProtocolGraph(daq, recording_state_manager)
    holdingProtocolGraph = HoldingProtocolGraph(daq, recording_state_manager)


    # datasetConverter = DatasetBuilder(dataset_name='dataset_1.hdf5')
    # datasetConverter.add_demo(demo_file_path='2025_02_14-12_57')

    gui.initialize()
    # gui.location_on_the_screen()
    gui.show()
    ret = app.exec_()
    sys.exit(ret)

if __name__ == "__main__":
    main()