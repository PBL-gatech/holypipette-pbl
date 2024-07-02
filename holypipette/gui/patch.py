from __future__ import absolute_import

from types import MethodType

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import PyQt5.QtGui as QtGui
import numpy as np
import logging


from ..controller import TaskController
from holypipette.gui.manipulator import ManipulatorGui
from holypipette.interface.patch import AutoPatchInterface
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.interface.pipettes import PipetteInterface
from holypipette.devices.camera.pcocamera import PcoCamera
from holypipette.interface.base import command

from holypipette.utils.FileLogger import FileLogger
from datetime import datetime

class PatchGui(ManipulatorGui):

    patch_command_signal = QtCore.pyqtSignal(MethodType, object)
    patch_reset_signal = QtCore.pyqtSignal(TaskController)

    def __init__(self, camera: PcoCamera, pipette_interface: PipetteInterface, patch_interface: AutoPatchInterface, recording_state_manager:RecordingStateManager, with_tracking=False):
        super(PatchGui, self).__init__(camera, pipette_interface,with_tracking=with_tracking,recording_state_manager=recording_state_manager)

        self.setWindowTitle("Patch GUI")
        # Note that pipette interface already runs in a thread, we need to use
        # the same for the patch interface

        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface
        self.recording_state_manager = recording_state_manager

        self.patch_interface.moveToThread(pipette_interface.thread())
        self.interface_signals[self.patch_interface] = (self.patch_command_signal,
                                                        self.patch_reset_signal)
        
        #add patching button tab
        button_tab = PatchButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
        self.add_config_gui(self.patch_interface.config)
        self.add_tab(button_tab, 'Commands', index = 0)

        #add cell sorter button tab
        cellsorter_tab = CellSorterButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals)
        self.add_tab(cellsorter_tab, 'Cell Sorter', index = 0)

        # Update the pressure and information in the status bar every 16ms
        self.pressure_timer = QtCore.QTimer()
        self.pressure_timer.timeout.connect(self.display_pressure)
        self.pressure_timer.start(16)
        self.patch_interface.set_pressure_near()

    # this is heavily affecting performance. If we use lastVal it introduces a delay of of a few seconds
    # however this implementation means that we are reading the arduino meaure twice, and that delays are being stacked?
    def display_pressure(self):
        # current_pressure = self.patch_interface.pressure.get_pressure()
        # current_pressure = self.patch_interface.pressure.measure()
        current_pressure = self.patch_interface.pressure.getLastVal()
        self.set_status_message('pressure', 'Pressure: {:.0f} mbar'.format(current_pressure))

    def register_commands(self):
        super(PatchGui, self).register_commands()
        # self.register_mouse_action(Qt.LeftButton, Qt.ShiftModifier,
        #                            self.patch_interface.patch_with_move)
        self.register_mouse_action(Qt.LeftButton, Qt.NoModifier,
                                   self.patch_interface.add_cell)
        self.register_key_action(Qt.Key_B, None,
                                 self.patch_interface.break_in)
        self.register_key_action(Qt.Key_F2, None,
                                 self.patch_interface.store_cleaning_position)
        self.register_key_action(Qt.Key_F3, None,
                                 self.patch_interface.store_rinsing_position)
        self.register_key_action(Qt.Key_F4, None,
                                 self.patch_interface.clean_pipette)


class TrackingPatchGui(PatchGui):
    def __init__(self, camera, pipette_interface, patch_interface,
                 with_tracking=False):
        super(TrackingPatchGui, self).__init__(camera, pipette_interface,
                                               patch_interface,
                                               with_tracking=True)
        self.setWindowTitle("Patch GUI with tracking")

    def register_commands(self):
        super(TrackingPatchGui, self).register_commands()
        self.register_key_action(Qt.Key_F5, None,
                                 self.patch_interface.sequential_patching)
        self.register_key_action(Qt.Key_F8, None,
                                 self.patch_interface.contact_detection)

class ButtonTabWidget(QtWidgets.QWidget):
    def nothing(self):
        pass

    def __init__(self):
        super(ButtonTabWidget, self).__init__()

    def do_nothing(self):
        pass # a dummy function for buttons that aren't implemented yet

    def run_command(self, cmd):

        #check if the task_description exists
        if hasattr(cmd, 'task_description'):
            #we're dealing with a command
            self.start_task(cmd.task_description, cmd.__self__)
            if cmd.__self__ in self.interface_signals:
                command_signal, _ = self.interface_signals[cmd.__self__]
                command_signal.emit(cmd, None)
            else:
                cmd(None)
        else:
            #we're dealing with a function
            cmd()

    def addPositionBox(self, name: str, layout, update_func, tare_func=None, axes=['x', 'y', 'z']):
        #add a box for manipulator and stage positions
        box = QtWidgets.QGroupBox(name)
        row = QtWidgets.QHBoxLayout()
        indices = []
        #create a new row for each position
        for j, axis in enumerate(axes):
            #create a label for the position
            label = QtWidgets.QLabel(f'{axis}: TODO')
            row.addWidget(label)

            indices.append(len(self.pos_labels))
            self.pos_labels.append(label)
        box.setLayout(row)
        layout.addWidget(box)

        if tare_func is not None:
            #add a button to tare the manipulator
            tare_button = QtWidgets.QPushButton('Tare')
            tare_button.clicked.connect(lambda: tare_func())
            row.addWidget(tare_button)

        #periodically update the position labels
        pos_timer = QtCore.QTimer()
        pos_timer.timeout.connect(lambda: update_func(indices))
        pos_timer.start(16)
        self.pos_update_timers.append(pos_timer)
    
    def addButtonList(self, box_name: str, layout: QtWidgets.QVBoxLayout, buttonNames: list[list[str]], cmds):
        box = QtWidgets.QGroupBox(box_name)
        rows = QtWidgets.QVBoxLayout()
        # create a new row for each button
        for i, buttons_in_row in enumerate(buttonNames):
            new_row = QtWidgets.QHBoxLayout()
            new_row.setAlignment(Qt.AlignLeft)

            #for each button in the row, create a button
            for j, button in enumerate(buttons_in_row):
                button = QtWidgets.QPushButton(button)

                #make sure buttons fill the space in the x-axis
                button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

                #set the size of the button
                button.setMinimumWidth(50)
                button.setMinimumHeight(50)

                #connect the button to the command, run using the start_task method
                button.clicked.connect(lambda state, i=i, j=j: self.run_command(cmds[i][j]))

                #add the button to the frame
                new_row.addWidget(button)
            rows.addLayout(new_row)
        box.setLayout(rows)
        layout.addWidget(box)

    
class PatchButtons(ButtonTabWidget):
    def __init__(self, patch_interface : AutoPatchInterface, pipette_interface : PipetteInterface, start_task, interface_signals, recording_state_manager):
        super(PatchButtons, self).__init__()
        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface

        self.start_task = start_task
        self.interface_signals = interface_signals

        self.pos_update_timers = []
        self.pos_labels = []
        self.recording_state_manager = recording_state_manager

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.stage_xy = [0, 0]
        self.stage_z = 0
        self.pipette_xyz = [0,0,0]
        
        self.recorder = FileLogger(self.recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", recorder_filename="movement_recording")

        self.addPositionBox('stage position (um)', layout, self.update_stage_pos_labels, tare_func=self.tare_stage)
        self.addPositionBox('pipette position (um)', layout, self.update_pipette_pos_labels, tare_func=self.tare_pipette)
        self.init_stage_pos = None #used to store bootup positions so we can reset to them
        self.init_pipette_pos = None

       
        #add a box for cal
        buttonList = [['Calibrate Stage','Set Cell Plane'], ['Add Pipette Cal Point', 'Finish Pipette Cal'], ['Save Calibration', 'Recalibrate Pipette']]
        cmds = [[self.pipette_interface.calibrate_stage, self.pipette_interface.set_floor], [self.pipette_interface.record_cal_point, self.pipette_interface.finish_calibration], [self.pipette_interface.write_calibration, self.pipette_interface.recalibrate_manipulator]]
        self.addButtonList('calibration', layout, buttonList, cmds)

        #add a box for movement
        buttonList = [[ 'Focus Cell Plane', 'Focus Pipette Plane'], ['Fix Backlash', 'Center Pipette'], ['Run protocols']]
        cmds = [[self.pipette_interface.go_to_floor, self.pipette_interface.focus_pipette], [self.pipette_interface.fix_backlash, self.pipette_interface.center_pipette], [self.patch_interface.run_protocols]]
        self.addButtonList('movement', layout, buttonList, cmds)

        #add a box for patching cmds
        buttonList = [['Select Cell', 'Remove Last Cell'], ['Start Patch', 'Break In'], ['Store Cleaning Position'], ['Clean Pipette']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell], [self.patch_interface.patch, self.patch_interface.break_in], [self.patch_interface.store_cleaning_position], [self.patch_interface.clean_pipette]]
        self.addButtonList('patching', layout, buttonList, cmds)

        # #add a box for Lumencor LED control
        # buttonList = [['None'], ['Violet', 'Blue'], ['Cyan', 'Yellow'], ['Red', 'Near Infrared']]
        # cmds = [[self.do_nothing], [self.do_nothing, self.do_nothing], [self.do_nothing, self.do_nothing], [self.do_nothing, self.do_nothing]]
        # self.addButtonList('Lumencor LED', layout, buttonList, cmds)

        #add a box for Rig Recorder
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.record_button.setMinimumWidth(50)
        self.record_button.setMinimumHeight(50)
        layout.addWidget(self.record_button)
        # buttonList = [['Start Recording', 'Stop Recording'], ['Save Recording', 'Load Recording']]
        # self.addButtonList('Rig Recorder', layout, buttonList, cmds)
        # buttonList = [['Start Recording', 'Stop Recording']]
        # cmds = [[self.start_recording, self.stop_recording]]
        # # buttonList = [['Start Recording', 'Stop Recording'], ['Save Recording', 'Load Recording']]
        # self.addButtonList('Rig Recorder', layout, buttonList, cmds)
        
        self.setLayout(layout)

    def toggle_recording(self):
        if self.recording_state_manager.is_recording_enabled():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording_state_manager.set_recording(True)
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: red; color: white;")
        logging.info("Recording started")

    def stop_recording(self):
        self.recording_state_manager.set_recording(False)
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        logging.info("Recording stopped")


    def close(self):
        self.recorder.close()
        super(PatchButtons, self).close()

    def closeEvent(self):
        self.recorder.close()
        super(PatchButtons, self).closeEvent()

    def tare_pipette(self):
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        self.init_pipette_pos = currPos

    def update_pipette_pos_labels(self, indices):
        #update the position labels
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        if self.init_pipette_pos is None:
            self.init_pipette_pos = currPos
        currPos = currPos - self.init_pipette_pos

        self.recorder.setBatchMoves(True)
        self.recorder.write_movement_data_batch(datetime.now().timestamp(), self.stage_xy[0], self.stage_xy[1], self.stage_z, currPos[0], currPos[1], currPos[2])

        self.pipette_xyz = currPos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            label.setText(f'{label.text().split(":")[0]}: {currPos[i]:.2f}')

    def tare_stage(self):
        xyPos = self.pipette_interface.calibrated_stage.position()
        zPos = self.pipette_interface.microscope.position()
        self.init_stage_pos = np.array([xyPos[0], xyPos[1], zPos])


    def update_stage_pos_labels(self, indices):
        #update the position labels
        xyPos = self.pipette_interface.calibrated_stage.position()
        zPos = self.pipette_interface.microscope.position()

        if self.init_stage_pos is None:
            self.init_stage_pos = np.array([xyPos[0], xyPos[1], zPos])

        xyPos = xyPos - self.init_stage_pos[0:2]
        zPos = zPos - self.init_stage_pos[2]

        self.recorder.setBatchMoves(True)
        self.recorder.write_movement_data_batch(datetime.now().timestamp(), xyPos[0], xyPos[1], zPos, self.pipette_xyz[0], self.pipette_xyz[1], self.pipette_xyz[2])

        self.stage_xy = xyPos
        self.stage_z = zPos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            if i < 2:
                label.setText(f'{label.text().split(":")[0]}: {xyPos[i]:.2f}')
            else:
                #note: divide by 5 here to account for z-axis gear ratio
                label.setText(f'{label.text().split(":")[0]}: {zPos/5:.2f}') 


class CellSorterButtons(ButtonTabWidget):
    def __init__(self, patch_interface : AutoPatchInterface, pipette_interface : PipetteInterface, start_task, interface_signals):
        super(CellSorterButtons, self).__init__()
        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface

        self.start_task = start_task
        self.interface_signals = interface_signals

        self.pos_update_timers = []
        self.pos_labels = []

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.addPositionBox('Automated Movement', layout, self.update_cellsorter_pos_labels, axes=['Z'])

        #add a box for movement
        buttonList = [['Calibrate', 'Sorter to Cell']]
        cmds = [[self.pipette_interface.calibrate_cell_sorter, self.patch_interface.move_cellsorter_to_cell]]
        self.addButtonList('Cell Sorter Movement', layout, buttonList, cmds)
        self.addCellSorterControlBox('Cell Sorter Control', layout)

        self.setLayout(layout)

    def update_cellsorter_pos_labels(self, indices):
        #update the position labels
        currPos = self.pipette_interface.calibrated_cellsorter.position()
        for _, ind in enumerate(indices):
            label = self.pos_labels[ind]
            label.setText(f'{label.text().split(":")[0]}: {currPos:.2f}')

    def addCellSorterControlBox(self, name, layout):

        posLayout = QtWidgets.QGroupBox(name)
        rows = QtWidgets.QVBoxLayout()

        #add a label
        movement_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setText("Movement Control")
        label.setAlignment(Qt.AlignCenter)

        #add label to layout
        movement_row.addWidget(label)
        
        #add position text input
        posInput = QtWidgets.QLineEdit()
        posInput.setPlaceholderText('Relative Movement (um) Position')
        posInput.setValidator(QtGui.QDoubleValidator())
        posInput.returnPressed.connect(lambda: self.pipette_interface.calibrated_cellsorter.relative_move(float(posInput.text())))
        movement_row.addWidget(posInput)

        #add movement to layout
        rows.addLayout(movement_row)

        #add suction control row (label, input, button)
        suction_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setText("Suction")
        label.setAlignment(Qt.AlignCenter)

        #add label to layout
        suction_row.addWidget(label)

        #add pressure text input
        suctionInput = QtWidgets.QLineEdit()
        suctionInput.setPlaceholderText('Duration (ms)')
        suctionInput.setValidator(QtGui.QIntValidator())
        suction_row.addWidget(suctionInput)

        label = QtWidgets.QLabel(name)
        label.setText("ms")
        label.setAlignment(Qt.AlignCenter)
        suction_row.addWidget(label)

        #add button
        suctionButton = QtWidgets.QPushButton('Go')
        suctionButton.clicked.connect(lambda: self.pipette_interface.calibrated_cellsorter.pulse_suction(int(suctionInput.text())))
        suction_row.addWidget(suctionButton)
        rows.addLayout(suction_row)

        #add pressure control row (label, input, button)
        pressure_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setText("Pressure")
        label.setAlignment(Qt.AlignCenter)
        pressure_row.addWidget(label)

        #add pressure text input
        pressureInput = QtWidgets.QLineEdit()
        pressureInput.setPlaceholderText('Duration (ms)')
        pressureInput.setValidator(QtGui.QIntValidator())
        pressure_row.addWidget(pressureInput)

        label = QtWidgets.QLabel(name)
        label.setText("ms")
        label.setAlignment(Qt.AlignCenter)
        pressure_row.addWidget(label)

        #add button
        pressureButton = QtWidgets.QPushButton('Go')
        pressureButton.clicked.connect(lambda: self.pipette_interface.calibrated_cellsorter.pulse_pressure(int(pressureInput.text())))
        pressure_row.addWidget(pressureButton)
        rows.addLayout(pressure_row)

        #add radio button options for light (off, ring1, ring2)
        light_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setText("Light")
        label.setAlignment(Qt.AlignCenter)
        light_row.addWidget(label)

        #add radio buttons
        lightGroup = QtWidgets.QButtonGroup()
        lightGroup.setExclusive(True)
        lightOff = QtWidgets.QRadioButton('Off')
        lightGroup.addButton(lightOff)
        lightRing1 = QtWidgets.QRadioButton('Ring 1')
        lightGroup.addButton(lightRing1)
        lightRing1.setChecked(True)
        lightRing2 = QtWidgets.QRadioButton('Ring 2')
        lightGroup.addButton(lightRing2)
        #command cell sorter to turn off light when radio button is clicked
        lightOff.clicked.connect(lambda: self.pipette_interface.calibrated_cellsorter.set_led_status(False))
        lightRing1.clicked.connect(lambda: self.pipette_interface.calibrated_cellsorter.set_led_status(True, 1))
        lightRing2.clicked.connect(lambda: self.pipette_interface.calibrated_cellsorter.set_led_status(True, 2))
        
        light_row.addWidget(lightOff)
        light_row.addWidget(lightRing1)
        light_row.addWidget(lightRing2)
        rows.addLayout(light_row)




        #add rows to layout
        posLayout.setLayout(rows)
        layout.addWidget(posLayout)

        
        





