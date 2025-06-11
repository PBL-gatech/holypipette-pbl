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
        self.add_config_gui(self.patch_interface.config)
        logging.debug("Added config GUI.")
        classic_patching_tab = ClassicPatchButtons(self.patch_interface, pipette_interface, self.start_task,self.interface_signals, self.recording_state_manager)
        self.add_tab(classic_patching_tab, 'Classic Auto Patching', index = 0)

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
        self.section_buttons = {}  # Dictionary to store buttons by section
        self.color_change_sections = []  # Sections that should change color on completion
        self.section_colors = {}  # Store custom colors for different sections


    def do_nothing(self):
        pass  # a dummy function for buttons that aren't implemented yet
    
    def run_sequential_commands(self, cmds, button=None, section=None, button_name=None):
        # Ensure cmds is a list
        if not isinstance(cmds, list):
            cmds = [cmds]
            
        # Have the button immediately lose focus to prevent persistent outline
        if button:
            button.clearFocus()
            
        # Store the command list and reset index
        self._seq_cmds = cmds
        self._seq_index = 0
        self._seq_button = button
        self._seq_section = section
        self._seq_button_name = button_name
        
        # Special case for reset button in any section (assuming it contains "Clear" or "Reset")
        if (section in self.section_buttons and button and 
            ("Clear" in button_name or "Reset" in button_name)):
            # Reset all section button colors before running the command
            self._reset_section_button_colors(section)
        
        self._run_next_seq_command()

    def _reset_section_button_colors(self, section):
        """Reset colors for all buttons in a section"""
        if section in self.section_buttons:
            for button_info in self.section_buttons[section]:
                button = button_info[0]
                button.setStyleSheet("")  # This will revert to the style from CollapsibleGroupBox

    def _run_next_seq_command(self):
        if self._seq_index >= len(self._seq_cmds):
            # No more commands; sequence complete
            # Update button color if this section should change colors and it's not a reset button
            if (self._seq_section in self.color_change_sections and self._seq_button and 
                not any(reset_term in self._seq_button_name for reset_term in ["Clear", "Reset"])):
                # Get the color for this section, or use default blue
                color = self.section_colors.get(self._seq_section, "rgba(0, 0, 255, 0.3)")
                
                # Set completed style while preserving all original behaviors
                self._seq_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color}; 
                        border: 1px solid lightgray;
                        border-radius: 6px;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(173, 216, 230, 0.5);
                        border: 1px solid #87CEEB;
                    }}
                    QPushButton:pressed {{
                        background-color: #d1e7ff;
                    }}
                    QPushButton:focus {{
                        border: 1px solid lightgray;
                        outline: none;
                    }}
                """)
            return

        # Rest of the method implementation unchanged
        cmd = self._seq_cmds[self._seq_index]
        self._seq_index += 1

        # Check if the command is asynchronous (has task_description)
        if hasattr(cmd, 'task_description'):
            interface = cmd.__self__
            # Define a temporary slot that waits for the command to finish
            def on_finished(exit_code, message):
                try:
                    interface.task_finished.disconnect(on_finished)
                except Exception:
                    pass
                # Launch next command after current one finishes
                self._run_next_seq_command()
            # Connect to the task_finished signal
            interface.task_finished.connect(on_finished)
            # Start the task and execute the command
            self.start_task(cmd.task_description, interface)
            if interface in self.interface_signals:
                command_signal, _ = self.interface_signals[interface]
                command_signal.emit(cmd, None)
            else:
                cmd(None)
        else:
            # Synchronous command: run it immediately
            cmd()
            self._run_next_seq_command()


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

    def addButtonList(self, box_name: str, layout: QtWidgets.QVBoxLayout, buttonNames: list[list[str]], 
                    cmds, sequential=False, change_color_on_complete=False, 
                    completion_color="rgba(0, 0, 255, 0.3)"):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(box_name)
        rows = QtWidgets.QVBoxLayout()
        
        # Initialize list to store buttons for this section
        section_buttons = []
        
        # Store color change preference and custom color for this section
        if change_color_on_complete:
            self.color_change_sections.append(box_name)
            self.section_colors[box_name] = completion_color
        
        for i, buttons_in_row in enumerate(buttonNames):
            new_row = QtWidgets.QHBoxLayout()
            new_row.setAlignment(Qt.AlignLeft)

            for j, button_name in enumerate(buttons_in_row):
                button = QtWidgets.QPushButton(button_name)
                button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                button.setMinimumWidth(50)
                button.setMinimumHeight(50)
                
                # Track this button for this section
                section_buttons.append((button, i, j, button_name))

                # Use a lambda function with default arguments to correctly capture the command
                if i < len(cmds) and j < len(cmds[i]):
                    button_cmd = cmds[i][j]
                    if sequential:
                        button.clicked.connect(lambda state, cmd=button_cmd, btn=button, section=box_name, 
                                            name=button_name: self.run_sequential_commands(cmd, btn, section, name))
                    else:
                        button.clicked.connect(lambda state, cmd=button_cmd: self.run_command(cmd))
                else:
                    button.clicked.connect(self.do_nothing)

                new_row.addWidget(button)
            rows.addLayout(new_row)
        
        # Store buttons for this section
        self.section_buttons[box_name] = section_buttons

        box.setContentLayout(rows)
        layout.addWidget(box)

class ClassicPatchButtons(ButtonTabWidget):
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

        self.stage_calibration = [self.pipette_interface.set_floor, self.pipette_interface.calibrate_stage, self.pipette_interface.move_microscope]
        self.pipette_calibration = [self.pipette_interface.calibrate_manipulator, self.patch_interface.store_calibration_positions, self.patch_interface.move_to_safe_space]
        self.pipette_cleaning_calibration = [self.patch_interface.store_cleaning_position,self.patch_interface.move_pipette_up,self.patch_interface.move_to_safe_space]


        # Add a box for calibration setup
        buttonList = [['Calibrate Stage','Calibrate Pipette'],['Store Cleaning Position'],['Clear Calibration']]
        cmds = [[self.stage_calibration, self.pipette_calibration],
                [self.pipette_cleaning_calibration],
                [self.patch_interface.clear_positions]
        ]
        self.addButtonList('calibration', layout, buttonList, cmds, sequential=True, 
                        change_color_on_complete=True, completion_color="rgba(173, 216, 230, 0.5)")

        # Add a box for movement commands - example with different color
        buttonList = [['move group down','move group up'],['Move to Safe Position','Move to Home Position'],['Move to cell plane'],['Center Pipette','Clean pipette','Focus Pipette']]
        cmds = [
            [self.patch_interface.move_group_down, self.patch_interface.move_group_up],
            [self.patch_interface.move_to_safe_space, self.patch_interface.move_to_home_space],
            [self.pipette_interface.go_to_floor],
            [self.pipette_interface.center_pipette,self.patch_interface.clean_pipette,self.pipette_interface.focus_pipette]
        ]
        self.addButtonList('movement', layout, buttonList, cmds, sequential=True)


        # Add a box for patching commands
        buttonList = [['Select Cell','Remove Last Cell','Center on Cell'],['Locate Cell','Hunt Cell','Gigaseal'],['Break-in','Run Protocols'],['Patch Cell','Escape Cell']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell, self.patch_interface.center_on_cell],
                [self.patch_interface.locate_cell,[self.start_recording,self.patch_interface.hunt_cell],self.patch_interface.gigaseal],
                [self.patch_interface.break_in,[self.stop_recording,self.recording_state_manager.increment_sample_number,self.patch_interface.run_protocols]],
                [[self.start_recording,self.patch_interface.patch,self.stop_recording],[self.stop_recording,self.patch_interface.escape_cell]]
]
        self.addButtonList('patching', layout, buttonList, cmds,sequential=True)

        # Add a box for Rig Recorder
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.record_button.setMinimumWidth(50)
        self.record_button.setMinimumHeight(50)
        layout.addWidget(self.record_button)

        self.setLayout(layout)

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
        super(ClassicPatchButtons, self).close()

    def closeEvent(self, event):
        self.save_stage_pos()
        self.recorder.close()
        super(ClassicPatchButtons, self).closeEvent(event)

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
            timestamp = datetime.now().timestamp()
            # logging.info(f"the current time is {timestamp}")
            self.recorder.write_movement_data_batch(
                timestamp,
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

