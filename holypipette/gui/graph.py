import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton,QSlider, QToolButton, QComboBox
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.colors import LinearSegmentedColormap, to_hex


from pyqtgraph import PlotWidget
from pyqtgraph.exporters import ImageExporter
import io
from PIL import Image

import threading

import numpy as np
from collections import deque
from holypipette.devices.amplifier import NiDAQ
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.pressurecontroller import PressureController
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.utils import FileLogger
from holypipette.utils import EPhysLogger
import time

from datetime import datetime

# new import

from holypipette.interface.graph import GraphInterface

__all__ = ["EPhysGraph", "CurrentProtocolGraph", "VoltageProtocolGraph", "HoldingProtocolGraph"]


class ProtocolGraph(QWidget):
    def __init__(self, daq: NiDAQ, recording_state_manager: RecordingStateManager,
                 window_title: str, y_label: str, y_unit: str,
                 x_label: str, x_unit: str, ephys_filename: str):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        self.daq = daq

        # Set window title and layout
        self.setWindowTitle(window_title)
        layout = QVBoxLayout()

        # Create and configure the PlotWidget
        self.plotWidget = PlotWidget()
        self.plotWidget.setBackground("w")
        self.plotWidget.getAxis("left").setPen("k")
        self.plotWidget.getAxis("bottom").setPen("k")
        self.plotWidget.setLabel("left", y_label, units=y_unit)
        self.plotWidget.setLabel("bottom", x_label, units=x_unit)
        layout.addWidget(self.plotWidget)

        self.latestDisplayedData = None

        self.setLayout(layout)
        self.raise_()
        self.show()

        # Hide window initially and remap close event to simply hide it
        self.setHidden(True)
        self.closeEvent = lambda event: self.setHidden(True)

        # Set up timer to call update_plot() every updateDt ms
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 10  # milliseconds
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)
        if not self.updateTimer.isActive():
            logging.info(f"{window_title} Timer not active")

        # Initialize the ephys logger with a protocol-specific filename
        self.ephys_logger = EPhysLogger(ephys_filename=ephys_filename,
                                        recording_state_manager=self.recording_state_manager)

    def update_plot(self):
        """This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement update_plot()")


class CurrentProtocolGraph(ProtocolGraph):
    def __init__(self, daq: NiDAQ, recording_state_manager: RecordingStateManager):
        super().__init__(daq, recording_state_manager,
                         window_title="Current Protocol",
                         y_label="Voltage", y_unit="V",
                         x_label="Time", x_unit="s",
                         ephys_filename="CurrentProtocol")

    def update_plot(self):
        # Check if new data exists and if it’s different from what was last displayed
        if self.daq.current_protocol_data is None or self.latestDisplayedData == self.daq.current_protocol_data:
            return

        index = self.recording_state_manager.sample_number

        # If the window was hidden, show it again
        if self.isHidden():
            self.setHidden(False)

        # Create a gradient color list based on the number of pulses
        color_range = self.daq.pulseRange
        logging.debug(f"color range: {color_range}")
        start_color = "#003153"  # Prussian Blue
        end_color = "#ffffff"    # White
        cmap = LinearSegmentedColormap.from_list("", [start_color, end_color])
        colors = [to_hex(cmap(float(i) / color_range)) for i in range(color_range)]
        pulses = self.daq.pulses

        self.plotWidget.clear()

        # Plot each pulse and log its data
        for i, graph in enumerate(self.daq.current_protocol_data):
            timeData = graph[0]
            respData = graph[1]
            readData = graph[2]
            self.plotWidget.plot(timeData, respData, pen=colors[i])
            logging.info("Writing current ephys data to file")
            pulse = str(pulses[i])
            marker = colors[i] + "_" + pulse
            self.ephys_logger.write_ephys_data(index, timeData, readData, respData, marker)
            if i == color_range - 1:
                logging.info("Saving current ephys plot")
                self.ephys_logger.save_ephys_plot(index, self.plotWidget)
                self.daq.current_protocol_data = None  # Reset after saving

        # Update latestDisplayedData (make a copy if data is still present)
        self.latestDisplayedData = (self.daq.current_protocol_data.copy() 
                                    if self.daq.current_protocol_data is not None else None)


class VoltageProtocolGraph(ProtocolGraph):
    def __init__(self, daq: NiDAQ, recording_state_manager: RecordingStateManager):
        super().__init__(daq, recording_state_manager,
                         window_title="Voltage Protocol (Membrane Test)",
                         y_label="PicoAmps", y_unit="A",
                         x_label="Time", x_unit="s",
                         ephys_filename="VoltageProtocol")

    def update_plot(self):
        # Compare arrays; if no new data or data is None, exit early.
        if (self.daq.voltage_protocol_data is None or 
            (self.latestDisplayedData is not None and 
             np.array_equal(np.array(self.latestDisplayedData), np.array(self.daq.voltage_protocol_data)))):
            return

        index = self.recording_state_manager.sample_number

        if self.isHidden():
            self.setHidden(False)

        self.plotWidget.clear()
        colors = ["k"]
        # Plot voltage protocol data
        self.plotWidget.plot(self.daq.voltage_protocol_data[0, :],
                             self.daq.voltage_protocol_data[1, :],
                             pen=colors[0])

        # Prepare data for logging
        timeData = self.daq.voltage_protocol_data[0, :]
        respData = self.daq.voltage_protocol_data[1, :]
        readData = self.daq.voltage_command_data[1, :]

        self.ephys_logger.write_ephys_data(index, timeData, readData, respData, colors[0])
        self.ephys_logger.save_ephys_plot(index, self.plotWidget)

        self.latestDisplayedData = self.daq.voltage_protocol_data.copy()
        self.daq.voltage_protocol_data = None  # Reset after plotting


class HoldingProtocolGraph(ProtocolGraph):
    def __init__(self, daq: NiDAQ, recording_state_manager: RecordingStateManager):
        super().__init__(daq, recording_state_manager,
                         window_title="Holding Protocol (E/I PSC Test)",
                         y_label="PicoAmps", y_unit="A",
                         x_label="Time", x_unit="s",
                         ephys_filename="HoldingProtocol")

    def update_plot(self):
        if self.daq.holding_protocol_data is None:
            return

        index = self.recording_state_manager.sample_number

        if self.isHidden():
            self.setHidden(False)

        self.plotWidget.clear()
        colors = ["k"]
        self.plotWidget.plot(self.daq.holding_protocol_data[0, :],
                             self.daq.holding_protocol_data[1, :],
                             pen=colors[0])
        self.ephys_logger.write_ephys_data(index,
                                           self.daq.holding_protocol_data[0, :],
                                           self.daq.holding_protocol_data[1, :],
                                           self.daq.holding_protocol_data[2, :],
                                           colors[0])
        self.ephys_logger.save_ephys_plot(index, self.plotWidget)

        self.latestDisplayedData = self.daq.holding_protocol_data.copy()
        self.daq.holding_protocol_data = None  # Reset after plotting

import logging
import time
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider,
    QPushButton, QToolButton, QComboBox
)
from PyQt5 import QtCore, QtGui
from pyqtgraph import PlotWidget

from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.utils import FileLogger
from holypipette.interface.graph import GraphInterface

class EPhysGraph(QWidget):
    pressureLowerBound = -450
    pressureUpperBound = 730

    def __init__(self, graph_interface: GraphInterface, recording_state_manager):
        """
        Initialize the electrophysiology GUI.
        :param graph_interface: An instance of GraphInterface that abstracts hardware operations.
        :param recording_state_manager: The recording state manager used for data logging.
        """
        super().__init__()
        self.setWindowTitle("Electrophysiology")
        self.recording_state_manager = recording_state_manager
        self.graph_interface = graph_interface  # ONLY import GraphInterface!

        # Initialize plots.
        self.cmdPlot = PlotWidget()
        self.respPlot = PlotWidget()
        self.pressurePlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        for plot in [self.cmdPlot, self.respPlot, self.pressurePlot, self.resistancePlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        self.cmdPlot.setLabel("left", "Command Voltage", units="V")
        self.cmdPlot.setLabel("bottom", "Time", units="s")
        self.respPlot.setLabel("left", "Current (resp)", units="A")
        self.respPlot.setLabel("bottom", "Time", units="s")
        self.pressurePlot.setLabel("left", "Pressure", units="mbar")
        self.pressurePlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", "Resistance", units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")

        # Build bottom control bar.
        self.bottomBar = QWidget()
        bottomBarLayout = QHBoxLayout()
        self.bottomBar.setLayout(bottomBarLayout)

        self.resistanceLabel = QLabel("Resistance:")
        bottomBarLayout.addWidget(self.resistanceLabel)

        # Toggle button for cell vs. bath mode.
        self.modelType = QPushButton("Bath Mode")
        self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
        bottomBarLayout.addWidget(self.modelType)

        self.accessResistanceLabel = QLabel("Access Resistance: N/A")
        bottomBarLayout.addWidget(self.accessResistanceLabel)
        self.membraneResistanceLabel = QLabel("Membrane Resistance: N/A")
        bottomBarLayout.addWidget(self.membraneResistanceLabel)
        self.membraneCapacitanceLabel = QLabel("Membrane Capacitance: N/A")
        bottomBarLayout.addWidget(self.membraneCapacitanceLabel)

        self.pressureLabel = QLabel("Pressure:")
        bottomBarLayout.addWidget(self.pressureLabel)

        # Pressure command text box.
        self.pressureCommandBox = QLineEdit()
        self.pressureCommandBox.setMaxLength(5)
        self.pressureCommandBox.setFixedWidth(100)
        initial_pressure = self.graph_interface.get_last_pressure() or 0
        self.pressureCommandBox.setPlaceholderText(f"{initial_pressure} mbar")
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(self.pressureLowerBound, self.pressureUpperBound))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)
        bottomBarLayout.addWidget(self.pressureCommandBox)

        # Pressure slider.
        self.pressureCommandSlider = QSlider(QtCore.Qt.Horizontal)
        self.pressureCommandSlider.setMinimum(self.pressureLowerBound)
        self.pressureCommandSlider.setMaximum(self.pressureUpperBound)
        self.pressureCommandSlider.setValue(initial_pressure)
        self.pressureCommandSlider.setTickInterval(100)
        self.pressureCommandSlider.setTickPosition(QSlider.TicksBelow)
        self.pressureCommandSlider.valueChanged.connect(self.updatePressureLabel)
        self.pressureCommandSlider.sliderReleased.connect(self.pressureCommandSliderChanged)
        bottomBarLayout.addWidget(self.pressureCommandSlider)

        # Up and down buttons for fine pressure control.
        self.upButton = QToolButton()
        self.upButton.setArrowType(QtCore.Qt.UpArrow)
        self.upButton.setFixedWidth(50)
        self.upButton.clicked.connect(self.incrementPressure)
        bottomBarLayout.addWidget(self.upButton)

        self.downButton = QToolButton()
        self.downButton.setArrowType(QtCore.Qt.DownArrow)
        self.downButton.setFixedWidth(50)
        self.downButton.clicked.connect(self.decrementPressure)
        bottomBarLayout.addWidget(self.downButton)

        # Atmospheric pressure toggle.
        self.atmosphericPressureButton = QPushButton("ATM Pressure OFF")
        bottomBarLayout.addWidget(self.atmosphericPressureButton)
        self.atmosphericPressureButton.clicked.connect(self.togglePressure)
        self.atmtoggle = True

        # Zap controls.
        self.zapLabel = QLabel("Zap Duration:")
        bottomBarLayout.addWidget(self.zapLabel)
        self.zapDurationDropdown = QComboBox()
        self.zapDurationDropdown.setFixedWidth(100)
        zap_options = ["25 µs", "50 µs", "100 µs", "200 µs", "500 µs", "1 ms", "10 ms", "20 ms", "50 ms"]
        for option in zap_options:
            self.zapDurationDropdown.addItem(option)
        self.zapDurationDropdown.currentIndexChanged.connect(self.handle_zap_duration_change)
        bottomBarLayout.addWidget(self.zapDurationDropdown)
        self.zapButton = QPushButton("Zap")
        self.zapButton.setFixedWidth(50)
        self.zapButton.clicked.connect(self.handle_zap_button_press)
        bottomBarLayout.addWidget(self.zapButton)

        bottomBarLayout.addStretch(1)
        self.bottomBar.setMaximumHeight(20)
        self.bottomBar.setMinimumHeight(20)

        # Compose main layout.
        mainLayout = QVBoxLayout()
        for plot in [self.cmdPlot, self.respPlot, self.pressurePlot, self.resistancePlot]:
            mainLayout.addWidget(plot)
        mainLayout.addWidget(self.bottomBar)
        self.setLayout(mainLayout)

        # Data containers.
        self.pressureData = deque(maxlen=100)
        self.resistanceDeque = deque(maxlen=100)
        self.lastReadData = None
        self.lastrespData = None

        # Recorder for saving data.
        self.recorder = FileLogger(
            recording_state_manager,
            folder_path="experiments/Data/rig_recorder_data/",
            recorder_filename="graph_recording"
        )

        # Connect the model type toggle. (Assumes GraphInterface now provides toggle_cell_mode() and get_cell_mode().)
        self.modelType.clicked.connect(self.toggleModelType)

        # QTimer for periodic GUI updates (approximately 20 hz).
        self.updateDt = 50 # ms
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.show()
        self.raise_()

    def update_plot(self):
        """
        Periodically update plots with the latest pressure and DAQ data retrieved via GraphInterface.
        """
        # --- Update Pressure Plot ---
        pressure = self.graph_interface.get_last_pressure()
        if pressure is not None:
            self.pressureData.append(pressure)
            pressureX = [i * self.updateDt / 1000 for i in range(len(self.pressureData))]
            self.pressurePlot.clear()
            self.pressurePlot.plot(pressureX, list(self.pressureData))
            self.pressureCommandBox.setPlaceholderText(f"{pressure} mbar")

        # --- Update DAQ Data (Command & Response) ---
        daq_data = self.graph_interface.get_last_data()
        if daq_data is not None:
            timeData = daq_data.get("timeData")
            respData = daq_data.get("respData")
            readData = daq_data.get("readData")
            totalResistance = daq_data.get("totalResistance")
            accessResistance = daq_data.get("accessResistance")
            membraneResistance = daq_data.get("membraneResistance")
            membraneCapacitance = daq_data.get("membraneCapacitance")

            if timeData is not None and readData is not None:
                self.cmdPlot.clear()
                self.cmdPlot.plot(timeData, readData, pen="k")
                self.lastReadData = readData

            if timeData is not None and respData is not None:
                self.respPlot.clear()
                self.respPlot.plot(timeData, respData, pen="k")
                self.lastrespData = respData

            if totalResistance is not None:
                self.resistanceDeque.append(totalResistance)
                self.resistanceLabel.setText(f"Resistance: {totalResistance:.2f} MΩ")
                x_vals = list(range(len(self.resistanceDeque)))
                self.resistancePlot.clear()
                self.resistancePlot.plot(x_vals, list(self.resistanceDeque), pen="k")
            if accessResistance is not None:
                self.accessResistanceLabel.setText(f"Access Resistance: {accessResistance:.2f} MΩ")
            if membraneResistance is not None:
                self.membraneResistanceLabel.setText(f"Membrane Resistance: {membraneResistance:.2f} MΩ")
            if membraneCapacitance is not None:
                self.membraneCapacitanceLabel.setText(f"Membrane Capacitance: {membraneCapacitance:.2f} pF")

            # --- Data Recording ---
            if self.recording_state_manager.is_recording_enabled():
                timestamp = datetime.now().timestamp()
                currentPressure = pressure if pressure is not None else 0
                try:
                    self.recorder.write_graph_data(
                        timestamp,
                        currentPressure,
                        totalResistance,
                        list(self.lastrespData) if self.lastrespData is not None else [],
                        list(self.lastReadData) if self.lastReadData is not None else []
                    )
                except Exception as e:
                    logging.error(f"Error writing graph data: {e}")

    def updatePressureLabel(self, value):
        self.pressureCommandBox.setPlaceholderText(f"Set to: {value} mbar")

    def pressureCommandSliderChanged(self):
        """
        On slider release, set the pressure via GraphInterface.
        """
        pressure = self.pressureCommandSlider.value()
        self.graph_interface.set_pressure(pressure)
        self.pressureCommandBox.setPlaceholderText(f"{pressure} mbar")

    def pressureCommandBoxReturnPressed(self):
        """
        When a pressure value is entered in the text box, update the setpoint.
        """
        try:
            text = self.pressureCommandBox.text()
            self.pressureCommandBox.clear()
            pressure = float(text)
            pressure = max(self.pressureLowerBound, min(self.pressureUpperBound, pressure))
            self.graph_interface.set_pressure(pressure)
            self.pressureCommandSlider.setValue(int(pressure))
            self.pressureCommandSlider.sliderReleased.emit()
        except ValueError:
            logging.warning("Invalid pressure input.")
        except Exception as e:
            logging.error(f"Error in pressureCommandBoxReturnPressed: {e}")

    def incrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value + 5
        if new_value <= self.pressureUpperBound:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()

    def decrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value - 5
        if new_value >= self.pressureLowerBound:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()

    def togglePressure(self):
        """
        Toggle atmospheric pressure mode using GraphInterface.
        """
        if self.atmtoggle:
            self.atmosphericPressureButton.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.atmosphericPressureButton.setText("ATM Pressure ON")
            self.graph_interface.set_ATM(True)
        else:
            self.atmosphericPressureButton.setStyleSheet("")
            self.atmosphericPressureButton.setText("ATM Pressure OFF")
            self.graph_interface.set_ATM(False)
        self.atmtoggle = not self.atmtoggle

    def toggleModelType(self):
        """
        Toggle between cell and bath modes using GraphInterface.
        This requires GraphInterface to provide toggle_cell_mode() and get_cell_mode() methods.
        """
        self.graph_interface.toggle_cell_mode()
        if self.graph_interface.get_cell_mode():
            self.modelType.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Cell Mode")
        else:
            self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Bath Mode")

    def handle_zap_button_press(self):
        """
        Provide visual feedback and execute the zap command via GraphInterface.
        """
        self.zapButton.setStyleSheet("background-color: yellow; color: black; border-radius: 5px; padding: 5px;")
        logging.info("Zapping...")
        self.graph_interface.zap()
        QtCore.QTimer.singleShot(250, self.reset_zap_button)

    def reset_zap_button(self):
        self.zapButton.setStyleSheet("")

    def handle_zap_duration_change(self, index):
        """
        Convert the selected zap duration to seconds and update via GraphInterface.
        """
        text = self.zapDurationDropdown.currentText()
        if "µs" in text or "us" in text:
            value = float(text.replace("µs", "").replace("us", "").strip())
            zap_duration = value * 1e-6
        elif "ms" in text:
            value = float(text.replace("ms", "").strip())
            zap_duration = value * 1e-3
        else:
            zap_duration = float(text)
        logging.info(f"Setting zap duration to {zap_duration} seconds")
        self.graph_interface.set_zap_duration(zap_duration)
