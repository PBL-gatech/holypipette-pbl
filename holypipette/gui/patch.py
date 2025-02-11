from __future__ import absolute_import

from types import MethodType

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
import PyQt5.QtGui as QtGui
import numpy as np
import logging

from PyQt5.QtWidgets import QDesktopWidget

from holypipette.controller import TaskController
from holypipette.gui.manipulator import ManipulatorGui
from holypipette.interface.patch import AutoPatchInterface
from holypipette.interface.pipettes import PipetteInterface
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.interface.base import command

from holypipette.utils.FileLogger import FileLogger
from datetime import datetime
import json
import os

class PatchGui(ManipulatorGui):

    patch_command_signal = QtCore.pyqtSignal(MethodType, object)
    patch_reset_signal = QtCore.pyqtSignal(TaskController)

    def __init__(self, camera, pipette_interface: PipetteInterface, patch_interface: AutoPatchInterface, recording_state_manager: RecordingStateManager, with_tracking=False):
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
    
        try:
            # Add patching button tab
            # button_tab = PatchButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
            self.add_config_gui(self.patch_interface.config)
            logging.debug("Added config GUI.")
            # self.add_tab(button_tab, 'Auto Patching', index=0)
            logging.debug("Added 'Auto Patching' tab.")
        except Exception as e:
            logging.error("Exception during PatchGui initialization: %s", e, exc_info=True)
            raise
        # #add cell sorter button tab
        # cellsorter_tab = CellSorterButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals)
        # self.add_tab(cellsorter_tab, 'Cell Sorter', index = 0)

        # add manual patching button tab
        # manual_patching_tab = ManualPatchButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
        # self.add_tab(manual_patching_tab, 'Manual Patching', index = 0)
        # add semi-auto patching button tab
        semi_auto_patching_tab = SemiAutoPatchButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
        self.add_tab(semi_auto_patching_tab, 'Semi-Auto Patching', index = 0)

        # Update the pressure and information in the status bar every 16ms
        self.pressure_timer = QtCore.QTimer()
        self.pressure_timer.timeout.connect(self.display_pressure)
        self.pressure_timer.start(16)
        self.patch_interface.set_pressure_near()

    def display_pressure(self):

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

class CollapsibleGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title="", parent=None):
        super(CollapsibleGroupBox, self).__init__(parent)
        self.setTitle("")  # Set the group box title to be blank to allow custom styling

        # Apply styles for rounded corners, grey borders, and consistent font
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid lightgray;  /* Light grey border */
                border-radius: 8px;           /* Rounded corners with 8px radius */
                margin-top: 10px;             /* Adjust top margin for visual separation */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for the group box */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-weight: bold;            /* Bold for the group box title */
            }
            QWidget {
                background-color: #f9f9f9;    /* Light grey background for the content area */
                border-radius: 8px;
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for content area */
            }
            QPushButton {
                background-color: #ffffff;     /* White background for buttons */
                border: 1px solid lightgray;   /* Light grey border for buttons */
                border-radius: 6px;            /* Slightly rounded corners for buttons */
                padding: 6px;                  /* Padding for a better button look */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;               /* Adjusted font size for buttons */
                outline: none;                 /* Remove default focus outline */
            }
            QPushButton:hover {
                background-color: rgba(173, 216, 230, 0.5);  /* Light blue with 50% transparency on hover */
                border: 1px solid #87CEEB;       /* Soft blue border on hover */
            }
            QPushButton:pressed {
                background-color: #d1e7ff;     /* Light blue when pressed for a subtle effect */
            }
            QPushButton:focus {
                border: 1px solid #87CEEB;      /* Consistent border color on focus (soft blue) */
                outline: none;                  /* Remove blue edge or highlight on focus */
            }
        """)

        # Create a toggle button (arrow) for expanding/collapsing
        self.toggle_button = QtWidgets.QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-family: Arial, Helvetica, sans-serif; font-size: 14px; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.on_toggle)

        # Layout for the toggle button
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.addWidget(self.toggle_button, alignment=Qt.AlignLeft)
        self.header_layout.addStretch()

        # Content area
        self.content_area = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        # Main layout of the collapsible group box
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.header_layout)
        self.main_layout.addWidget(self.content_area)
        self.main_layout.setContentsMargins(5, 5, 5, 5)  # Add some margin to create spacing inside
        self.setLayout(self.main_layout)

    def on_toggle(self):
        if self.toggle_button.isChecked():
            self.content_area.show()
            self.toggle_button.setArrowType(Qt.DownArrow)
        else:
            self.content_area.hide()
            self.toggle_button.setArrowType(Qt.RightArrow)

    def setContentLayout(self, layout):
        # Remove existing layout if any
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        self.content_layout.addLayout(layout)

class ButtonTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.pos_update_timers = []
        self.pos_labels = []
        self.interface_signals = {}
        self.start_task = None

    def do_nothing(self):
        pass  # a dummy function for buttons that aren't implemented yet

    def run_command(self, cmds):
        if isinstance(cmds, list):
            for cmd in cmds:
                if isinstance(cmd, list):
                    for sub_cmd in cmd:
                        self.execute_command(sub_cmd)
                else:
                    self.execute_command(cmd)
        else:
            self.execute_command(cmds)

    def execute_command(self, cmd):
        logging.info(f"Executing command: {cmd}")
        if hasattr(cmd, 'task_description'):
            self.start_task(cmd.task_description, cmd.__self__)
            if cmd.__self__ in self.interface_signals:
                command_signal, _ = self.interface_signals[cmd.__self__]
                command_signal.emit(cmd, None)
            else:
                cmd(None)
        else:
            cmd()

    def addPositionBox(self, name: str, layout, update_func, tare_func=None, axes=['x', 'y', 'z']):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(name)
        row = QtWidgets.QHBoxLayout()
        indices = []
        # Create a new row for each position
        for j, axis in enumerate(axes):
            # Create a label for the position
            label = QtWidgets.QLabel(f'{axis}: TODO')
            row.addWidget(label)

            indices.append(len(self.pos_labels))
            self.pos_labels.append(label)
        box.setContentLayout(row)
        layout.addWidget(box)

        if tare_func is not None:
            # Add a button to tare the manipulator
            tare_button = QtWidgets.QPushButton('Tare')
            tare_button.clicked.connect(lambda: tare_func())
            row.addWidget(tare_button)

        # Periodically update the position labels
        pos_timer = QtCore.QTimer()
        pos_timer.timeout.connect(lambda: update_func(indices))
        pos_timer.start(16)
        self.pos_update_timers.append(pos_timer)

    def positionAndTareBox(self, name: str, layout, update_func, tare_funcs, axes=['x', 'y', 'z']):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(name)
        main_layout = QtWidgets.QHBoxLayout()
        indices = []

        for j, axis in enumerate(axes):
            axis_layout = QtWidgets.QVBoxLayout()

            # Create a label for the position
            label = QtWidgets.QLabel(f'{axis}: 0.00')
            axis_layout.addWidget(label)
            indices.append(len(self.pos_labels))
            self.pos_labels.append(label)

            # Add a button to tare the manipulator
            tare_button = QtWidgets.QPushButton(f'Tare {axis}')
            tare_button.clicked.connect(tare_funcs[j])
            axis_layout.addWidget(tare_button)

            main_layout.addLayout(axis_layout)

        box.setContentLayout(main_layout)
        layout.addWidget(box)

        # Periodically update the position labels
        pos_timer = QtCore.QTimer()
        pos_timer.timeout.connect(lambda: update_func(indices))
        pos_timer.start(16)
        self.pos_update_timers.append(pos_timer)

    def addButtonList(self, box_name: str, layout: QtWidgets.QVBoxLayout, buttonNames: list[list[str]], cmds):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(box_name)
        rows = QtWidgets.QVBoxLayout()

        for i, buttons_in_row in enumerate(buttonNames):
            new_row = QtWidgets.QHBoxLayout()
            new_row.setAlignment(Qt.AlignLeft)

            for j, button_name in enumerate(buttons_in_row):
                button = QtWidgets.QPushButton(button_name)
                button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                button.setMinimumWidth(50)
                button.setMinimumHeight(50)

                # Use a lambda function with default arguments to correctly capture the command
                if i < len(cmds) and j < len(cmds[i]):
                    button_cmd = cmds[i][j]
                    button.clicked.connect(lambda state, cmd=button_cmd: self.run_command(cmd))
                else:
                    button.clicked.connect(self.do_nothing)

                new_row.addWidget(button)
            rows.addLayout(new_row)

        box.setContentLayout(rows)
        layout.addWidget(box)


class SemiAutoPatchButtons(ButtonTabWidget):
    def __init__(self, patch_interface: AutoPatchInterface, pipette_interface: PipetteInterface, start_task, interface_signals, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface

        self.start_task = start_task
        self.interface_signals = interface_signals

        self.recording_state_manager = recording_state_manager

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.stage_xy = [0, 0]
        self.stage_z = 0
        self.pipette_xyz = [0, 0, 0]
        self.tare_pipette_pos = [0, 0, 0]

        self.currx_stage_pos = [0, 0, 0]
        self.curry_stage_pos = [0, 0, 0]
        self.currz_stage_pos = [0, 0, 0]
   

        self.recorder = FileLogger(self.recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", recorder_filename="movement_recording")

        # Add position boxes using the updated methods (which use CollapsibleGroupBox)
        self.positionAndTareBox(
            'stage position (um)',
            layout,
            self.update_stage_pos_labels,
            tare_funcs=[self.tare_stage_x, self.tare_stage_y, self.tare_stage_z]
        )
        self.addPositionBox(
            'pipette position (um)',
            layout,
            self.update_pipette_pos_labels,
            tare_func=self.tare_pipette
        )

        # # Add box to emit patching states
        # buttonList = [['Cell Found', 'Gigaseal Reached', 'Whole Cell Achieved'], ['Patch Attempt Start', 'Patch Attempt Failed']]
        # cmds = [
        #     [self.emit_cell_found, self.emit_gigaseal, self.emit_whole_cell],
        #     [self.emit_patch_attempt_start, self.emit_patch_attempt_fail]
        # ]
        # self.addButtonList('patching states', layout, buttonList, cmds)

        # Add a box for calibration setup
        buttonList = [['Calibrate Stage','Calibrate Pipette'],['Store Cell Plane'],['Store Safe Position','Store Home Position'],['Store Cleaning Position']]
        cmds = [[self.pipette_interface.calibrate_stage, self.pipette_interface.calibrate_manipulator],
                [self.pipette_interface.set_floor],
                [self.patch_interface.store_safe_position,
                self.patch_interface.store_home_position],
                [self.patch_interface.store_cleaning_position]
        ]
        self.addButtonList('calibration', layout, buttonList, cmds)
        # Add a box for movement commands
        buttonList = [['move group down','move group up'],['Move to Safe Position','Move to Home Position'],['Move to cell plane'],['Center Pipette','Clean pipette']]
        cmds = [
            [self.patch_interface.move_group_down, self.patch_interface.move_group_up],
            [self.patch_interface.move_to_safe_space, self.patch_interface.move_to_home_space],
            [self.pipette_interface.go_to_floor],
            [self.pipette_interface.center_pipette,self.patch_interface.clean_pipette]
        ]
        self.addButtonList('movement', layout, buttonList, cmds)

        # Add a box for patching commands
        buttonList = [['Select Cell','Remove Last Cell'],['Hunt Cell','Gigaseal'],['Break-in','Run Protocols'],['Patch Cell']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell],
                [self.patch_interface.hunt_cell,self.patch_interface.break_in],
                [self.patch_interface.gigaseal,[self.patch_interface.run_protocols, self.recording_state_manager.increment_sample_number]],
                [self.patch_interface.patch]
]
            
    
        self.addButtonList('patching', layout, buttonList, cmds)

        # Add a box for Rig Recorder
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.record_button.setMinimumWidth(50)
        self.record_button.setMinimumHeight(50)
        layout.addWidget(self.record_button)

        self.setLayout(layout)
    


    def emit_cell_found(self):
        self.patch_interface.state_emitter("NH Success")

    def emit_gigaseal(self):
        self.patch_interface.state_emitter("GS success")

    def emit_whole_cell(self):
        self.patch_interface.state_emitter("WC success")

    def emit_patch_attempt_start(self):
        self.patch_interface.state_emitter("patching started")

    def emit_patch_attempt_fail(self):
        self.patch_interface.state_emitter("patching failed")

    def toggle_recording(self):
        if self.recording_state_manager.is_recording_enabled():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording_state_manager.set_recording(True)
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: red; color: white;border-radius: 5px; padding: 5px;")
        logging.info("Recording started")

    def stop_recording(self):
        self.recording_state_manager.set_recording(False)
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        logging.info("Recording stopped")

    def close(self):
        self.save_stage_pos()
        self.recorder.close()
        super(SemiAutoPatchButtons, self).close()

    def closeEvent(self, event):
        self.save_stage_pos()
        self.recorder.close()
        super(SemiAutoPatchButtons, self).closeEvent(event)

    def tare_pipette(self):
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        self.tare_pipette_pos = currPos

    def update_pipette_pos_labels(self, indices):
        # Update the position labels
        # start_time = time.perf_counter_ns()
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        currPos = currPos - self.tare_pipette_pos
        if self.recording_state_manager.is_recording_enabled():
            self.recorder.setBatchMoves(True)
            self.recorder.write_movement_data_batch(
                datetime.now().timestamp(),
                self.stage_xy[0],
                self.stage_xy[1],
                self.stage_z,
                currPos[0],
                currPos[1],
                currPos[2]
            )

        self.pipette_xyz = currPos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            label.setText(f'{label.text().split(":")[0]}: {currPos[i]:.2f}')


    def tare_stage_x(self):
        xPos = self.pipette_interface.calibrated_stage.position(0)
        self.currx_stage_pos = [xPos, 0, 0]
        print("Tare stage x: ", self.currx_stage_pos)

    def tare_stage_y(self):
        yPos = self.pipette_interface.calibrated_stage.position(1)
        self.curry_stage_pos = [0, yPos, 0]
        print("Tare stage y: ", self.curry_stage_pos)

    def tare_stage_z(self):
        zPos = self.pipette_interface.microscope.position()
        self.currz_stage_pos = [0, 0, zPos]
        print("Tare stage z: ", self.currz_stage_pos)

    def update_stage_pos_labels(self, indices):
        # Update the position labels
        # start_time = time.perf_counter_ns()
        xyPos = self.pipette_interface.calibrated_stage.position() - self.currx_stage_pos[0:2] - self.curry_stage_pos[0:2]
        zPos = self.pipette_interface.microscope.position() - self.currz_stage_pos[2]

        self.stage_xy = xyPos
        self.stage_z = zPos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            if i < 2:
                label.setText(f'{label.text().split(":")[0]}: {xyPos[i]:.2f}')
            else:
                # Note: divide by 5 here to account for z-axis gear ratio
                label.setText(f'{label.text().split(":")[0]}: {zPos/5:.2f}')

