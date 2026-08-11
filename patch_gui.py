# patch_gui.py
import faulthandler
faulthandler.enable()
# faulthandler.dump_traceback_later(5)

import sys
import atexit
from PyQt5.QtWidgets import QApplication, QMessageBox
import traceback
from patcherbot.utils.exception_handler import set_global_exception_hook

# Set the global exception hook
set_global_exception_hook()


from patcherbot.utils.log_utils import setup_logging
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.interface import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.interface.graph import GraphInterface
from patcherbot.gui.graph import EPhysGUI, EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, LeakSubtractionGraph, HoldingProtocolGraph, OptogeneticStimProtocolGraph, OptogeneticWavelengthProtocolGraph
from patcherbot.gui.patch import PatchGui
from rig_setup.rig_config import RigConfigError, RigConfigManager
from rig_setup.rig_selector import RigSelectorDialog
from patcherbot.devices.camera.FakeCalCamera import FakeCalCamera

setup_logging()  # Log to the standard console as well

def main():
    """
    Starts the Patch GUI application and prepares the rig for use.

    This function creates the graphical applications, prompts the user to select a rig configuration, 
    builds all required hardware device objects (stage, microscope, amplifier, etc.), and connects them to
    the appropriate controller and interface classes.

    It then creates the main window and signal graph displays, links all components together, and starts the Qt
    event loop so the program can respond to user input.

    If any configuration or hardware initialization step fails, the error is displayed and the program exists
    safely.

    Raises:
        RigConfigError: If the selected rig configuration is invalid or incomplete.
        Exception: If any other error occurs during initialization.
    """
    app = QApplication(sys.argv)
    manager = RigConfigManager()
    manager.ensure_default_config()

    selector = RigSelectorDialog(manager)
    if selector.exec_() != selector.Accepted:
        return

    config_path = selector.selected_path or manager.default_config_path()
    active_pipette_count = selector.selected_pipette_count

    try:
        config_data = manager.load_config(config_path)

        rig_devices = manager.build_devices(
            config_data,
            active_pipette_count=active_pipette_count,
        )
    except RigConfigError as exc:
        QMessageBox.critical(None, "Rig configuration error", str(exc))
        return
    except Exception:
        QMessageBox.critical(None, "Rig initialization failed", traceback.format_exc())
        return

    stage = rig_devices["stage"]
    microscope = rig_devices["microscope"]
    camera = rig_devices["camera"]
    # Ensure camera has refs if it needs them
    if isinstance(camera, FakeCalCamera):
        camera.stageManip = rig_devices["stage_controller"]
        camera.pipetteManip = rig_devices["pipette_controller"]
        camera.cellSorterManip = rig_devices["cell_sorter_manipulator"]
    pipette_camera = rig_devices["pipette_camera"]
    unit = rig_devices["pipette_unit"]
    cellSorterManip = rig_devices["cell_sorter_manipulator"]
    cellSorterController = rig_devices["cell_sorter_controller"]
    amplifier = rig_devices["amplifier"]
    daq = rig_devices["daq"]
    pressure = rig_devices["pressure"]
    lamp = rig_devices["lamp"]
    laser = rig_devices["laser"]

    recording_state_manager = RecordingStateManager()

    calibration_data = config_data.get("calibration") if isinstance(config_data, dict) else None
    patch_data = config_data.get("patch") if isinstance(config_data, dict) else None
    protocol_data = config_data.get("protocol") if isinstance(config_data, dict) else None


    if isinstance(unit, dict):
        patch_controllers = {}
        pipette_controllers = {}
        graph_interface = {}
        for i, id in enumerate(unit.keys()):

            curr_amplifier = list(amplifier.values())[i]
            curr_daq = list(daq.values())[i]
            curr_pressure = list(pressure.values())[i]

            pipette_controllers[id] = PipetteInterface(
                stage, microscope, camera, unit[id], cellSorterManip, cellSorterController,
                calibration_data=calibration_data,
            )

            patch_controllers[id] = AutoPatchInterface(
                curr_amplifier, curr_daq, curr_pressure, pipette_controllers[id], recording_state_manager, lamp, laser,
                config_data=patch_data,
                protocol_data=protocol_data,
            )

            graph_interface[id] = GraphInterface(curr_amplifier, curr_daq, curr_pressure, recording_state_manager, laser)

            currentProtocolGraph = CurrentProtocolGraph(graph_interface[id], recording_state_manager)
            voltageProtocolGraph = VoltageProtocolGraph(graph_interface[id], recording_state_manager)
            leakSubtractionGraph = LeakSubtractionGraph(graph_interface[id], recording_state_manager)
            holdingProtocolGraph = HoldingProtocolGraph(graph_interface[id], recording_state_manager)
            optogeneticStimProtocolGraph = OptogeneticStimProtocolGraph(graph_interface[id], recording_state_manager)
            optogeneticWavelengthProtocolGraph = OptogeneticWavelengthProtocolGraph(graph_interface[id], recording_state_manager)
    else:
        pipette_controllers = PipetteInterface(
            stage, microscope, camera, unit, cellSorterManip, cellSorterController,
            calibration_data=calibration_data,
            )
        
        patch_controllers = AutoPatchInterface(
                amplifier, daq, pressure, pipette_controllers, recording_state_manager, lamp, laser,
                config_data=patch_data,
                protocol_data=protocol_data,
            )
        
        graph_interface = GraphInterface(amplifier, daq, pressure, recording_state_manager, laser)

        currentProtocolGraph = CurrentProtocolGraph(graph_interface, recording_state_manager)
        voltageProtocolGraph = VoltageProtocolGraph(graph_interface, recording_state_manager)
        leakSubtractionGraph = LeakSubtractionGraph(graph_interface, recording_state_manager)
        holdingProtocolGraph = HoldingProtocolGraph(graph_interface, recording_state_manager)
        optogeneticStimProtocolGraph = OptogeneticStimProtocolGraph(graph_interface, recording_state_manager)
        optogeneticWavelengthProtocolGraph = OptogeneticWavelengthProtocolGraph(graph_interface, recording_state_manager)


    gui = PatchGui(camera, pipette_camera, pipette_controllers, patch_controllers, recording_state_manager)
    graphs = EPhysGUI(graph_interface, recording_state_manager)
    # graphs.location_on_the_screen()
    graphs.show()


    gui.initialize()
    gui.show()
    ret = app.exec_()
    sys.exit(ret)

if __name__ == "__main__":
    main()
