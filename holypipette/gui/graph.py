import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton, QDesktopWidget, QDesktopWidget, QSlider, QToolButton, QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.colors import LinearSegmentedColormap, to_hex


from pyqtgraph import PlotWidget




import threading

import numpy as np
from collections import deque
from holypipette.devices.amplifier import DAQ
from holypipette.devices.pressurecontroller import PressureController
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.utils import FileLogger
from holypipette.utils import EPhysLogger
import time

from datetime import datetime

__all__ = ["EPhysGraph", "CurrentProtocolGraph", "VoltageProtocolGraph", "HoldingProtocolGraph"]


class CurrentProtocolGraph(QWidget):
    def __init__(self, daq: DAQ, rescording_state_manager: RecordingStateManager):
        super().__init__()
        self.recording_state_manager = rescording_state_manager
        layout = QVBoxLayout()
        self.setWindowTitle("Current Protocol")
        logging.getLogger("matplotlib.font_manager").disabled = True
        self.daq = daq
        self.cprotocolPlot = PlotWidget()
        self.cprotocolPlot.setBackground("w")
        self.cprotocolPlot.getAxis("left").setPen("k")
        self.cprotocolPlot.getAxis("bottom").setPen("k")
        self.cprotocolPlot.setLabel("left", "Voltage", units = "V")
        self.cprotocolPlot.setLabel("bottom", "Time", units = "s")
        layout.addWidget(self.cprotocolPlot)

        self.latestDisplayedData = None

        self.setLayout(layout)
        self.raise_()
        self.show()

        #hide window
        self.setHidden(True)

        #remap close event to hide window
        self.closeEvent = lambda: self.setHidden(True)

        #start async daq data update
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 10 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)
        if  not self.updateTimer.isActive():
            logging.info("Cprot Timer not active")

        self.ephys_logger = EPhysLogger(ephys_filename = "CurrentProtocol", recording_state_manager = self.recording_state_manager)

    def update_plot(self):
        # is what we displayed the exact same?
        if self.latestDisplayedData == self.daq.current_protocol_data or self.daq.current_protocol_data is None:
            return
        
        index = self.recording_state_manager.sample_number
        # logging.info(f" current index: {index}")
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True
        # curr = self.daq.latest_protocol_data
        # logging.info('length of current protocol data: ' + str(len(curr[0])) + ' ' + str(len(curr[1])))
        # make a color gradient based on a list
        color_range = self.daq.pulseRange
        logging.debug(f"color range: {color_range}")
        #make a gradient of colors based off of color_range, a value that should describe the number of pulses (as letters)
        colors = [format((i / color_range), ".2f") for i in range(color_range)]
        start_color = "#003153" #Prussian Blue
        end_color =  "#ffffff" #White
        cmap = LinearSegmentedColormap.from_list("", [start_color, end_color])
        colors = [to_hex(cmap(float(i) / color_range)) for i in range(color_range)]
        pulses = self.daq.pulses
        # colors = ["k", 'r', 'g', 'b', 'y', 'm', 'c']
        self.cprotocolPlot.clear()

        # timestamp = datetime.now().timestamp()
        for i, graph in enumerate(self.daq.current_protocol_data):
            timeData = graph[0]
            respData = graph[1]
            readData = graph[2]
            self.cprotocolPlot.plot(timeData, respData, pen=colors[i])
            logging.info("writing current ephys data to file")
            pulse = str(pulses[i])
            marker = colors[i] + "_" + pulse
            self.ephys_logger.write_ephys_data(index, timeData, readData, respData, marker)
            
            if i == color_range-1:

                logging.info ("saving current ephys plot")
                # self.ephys_logger.write_ephys_data(timestamp, index, timeData, readData, respData, marker)
                self.ephys_logger.save_ephys_plot(index, self.cprotocolPlot)
                self.daq.current_protocol_data = None


        self.latestDisplayedData = self.daq.current_protocol_data.copy()
        
class VoltageProtocolGraph(QWidget):
    def __init__(self, daq: DAQ, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        layout = QVBoxLayout()
        self.setWindowTitle("Voltage Protocol (Membrane Test)")
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.daq = daq
        self.vprotocolPlot = PlotWidget()
        self.vprotocolPlot.setBackground('w')
        self.vprotocolPlot.getAxis('left').setPen("k")
        self.vprotocolPlot.getAxis('bottom').setPen("k")
        self.vprotocolPlot.setLabel('left', "PicoAmps", units='A')
        self.vprotocolPlot.setLabel('bottom', "time", units='s')
        layout.addWidget(self.vprotocolPlot)

        self.latestDisplayedData = None

        self.setLayout(layout)
        self.raise_()
        self.show()

        #hide window
        self.setHidden(True)

        #remap close event to hide window
        self.closeEvent = lambda: self.setHidden(True)

        self.updateTimer = QtCore.QTimer()
        self.updateDt = 10 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)
        if  not self.updateTimer.isActive():
            logging.info("Vprot Timer not active")

        self.ephys_logger = EPhysLogger(ephys_filename = "VoltageProtocol", recording_state_manager=self.recording_state_manager)


    def update_plot(self):
        #is what we displayed the exact same?

        # logging.warning("window should be shown")
        if np.array_equal(np.array(self.latestDisplayedData), np.array(self.daq.voltage_protocol_data)) or self.daq.voltage_protocol_data is None:
            return

        index = self.recording_state_manager.sample_number
        # logging.info(f"voltage index: {index}")
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        if self.daq.voltage_protocol_data is not None:
            self.vprotocolPlot.clear()
            # print(self.daq.voltage_protocol_data[0, :])
            # print(self.daq.voltage_protocol_data[1, :])
            
            colors = ["k"]
            self.vprotocolPlot.plot(self.daq.voltage_protocol_data[0, :], self.daq.voltage_protocol_data[1, :], pen=colors[0])


            # timestamp = datetime.now().timestamp()
            timeData = self.daq.voltage_protocol_data[0, :]
            respData = self.daq.voltage_protocol_data[1, :]
            readData = self.daq.voltage_command_data[1,:]
            
            # logging.info("writing Voltage ephys data to file")
            self.ephys_logger.write_ephys_data(index, timeData, readData, respData, colors[0])
            self.ephys_logger.save_ephys_plot( index, self.vprotocolPlot)
            self.latestDisplayedData = self.daq.voltage_protocol_data.copy()
            self.daq.voltage_protocol_data = None # This causes a crash

class HoldingProtocolGraph(QWidget):
    def __init__(self, daq : DAQ, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        layout = QVBoxLayout()
        self.setWindowTitle("Holding Protocol (E/I PSC Test")
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.daq = daq
        self.hprotocolPlot = PlotWidget()
        self.hprotocolPlot.setBackground('w')
        self.hprotocolPlot.getAxis('left').setPen("k")
        self.hprotocolPlot.getAxis('bottom').setPen("k")
        self.hprotocolPlot.setLabel('left', "PicoAmps", units='A')
        self.hprotocolPlot.setLabel("bottom", "Time", units="s")
        layout.addWidget(self.hprotocolPlot)

        self.latestDisplayedData = None

        self.setLayout(layout)
        self.raise_()
        self.show()

        #hide window
        self.setHidden(True)

        #remap close event to hide window
        self.closeEvent = lambda: self.setHidden(True)

        self.updateTimer = QtCore.QTimer()
        self.updateDt = 10 # ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)
        if  not self.updateTimer.isActive():
            logging.info("Hprot Timer not active")

        self.ephys_logger = EPhysLogger(ephys_filename = "HoldingProtocol", recording_state_manager = self.recording_state_manager)

    def update_plot(self):
        # logging.warning("window should be shown")
        # is what we displayed the exact same?
        if self.daq.holding_protocol_data is None:
            # logging.warning("no new data, skipping plot update")
            return
        
        index = self.recording_state_manager.sample_number
        # logging.info(f" holding index: {index}")
        # logging.warning("new data, updating plot")
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        self.hprotocolPlot.clear()

        colors = ["k"]
        self.hprotocolPlot.plot(self.daq.holding_protocol_data[0, :], self.daq.holding_protocol_data[1, :], pen=colors[0])
        # timestamp = datetime.now().timestamp()
        self.ephys_logger.write_ephys_data(index, self.daq.holding_protocol_data[0,:],self.daq.holding_protocol_data[1,:],self.daq.holding_protocol_data[2,:], colors[0])
        self.ephys_logger.save_ephys_plot( index, self.hprotocolPlot)
    
        self.latestDisplayedData = self.daq.holding_protocol_data.copy()
        self.daq.holding_protocol_data = None

        # self.latestDisplayedData = self.daq.holding_protocol_data.copy()



class EPhysGraph(QWidget):
    """
    A window that plots electrophysiology data from the DAQ
    """

    # Define a signal that can accept DAQ data
    data_updated = pyqtSignal(object, object, object, object, object, object)

    pressureLowerBound = -450
    pressureUpperBound = 730

    def __init__(self, daq: DAQ, pressureController: PressureController, recording_state_manager: RecordingStateManager):
        super().__init__()

        # Stop matplotlib font warnings
        logging.getLogger("matplotlib.font_manager").disabled = True
        self.atmtoggle = True
        self.daq = daq

        # Initialize mode
        self.cellMode = False  # Initially set to Bath Mode

        self.pressureController = pressureController
        self.recording_state_manager = recording_state_manager  # Include the state manager in the graph
        self.setpoint = 0

        # Constants for Multi Clamp
        self.externalCommandSensitivity = 20  # mV/V
        self.triggerLevel = 0.05  # V

        # Setup window
        self.setWindowTitle("Electrophysiology")

        # Initialize plots
        self.cmdPlot = PlotWidget()
        self.respPlot = PlotWidget()
        self.pressurePlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        # Set background color of plots
        self.cmdPlot.setBackground("w")
        self.respPlot.setBackground("w")
        self.pressurePlot.setBackground("w")
        self.resistancePlot.setBackground("w")

        # Set axis colors to black
        for plot in [self.cmdPlot, self.respPlot, self.resistancePlot, self.pressurePlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        # Set labels
        self.cmdPlot.setLabel("left", "Command Voltage", units="V")
        self.cmdPlot.setLabel("bottom", "Time", units="s")
        self.respPlot.setLabel("left", "Current (resp)", units="A")
        self.respPlot.setLabel("bottom", "Time", units="s")
        self.pressurePlot.setLabel("left", "Pressure", units="mbar")
        self.pressurePlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", "Resistance", units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")

        # Initialize data containers
        self.pressureData = deque(maxlen=100)
        self.resistanceDeque = deque(maxlen=100)

        # Thread-safe storage for the latest pressure value
        self.latest_pressure = 0
        self.pressure_lock = threading.Lock()

        # Create layout and add plots
        layout = QVBoxLayout()
        for plot in [self.cmdPlot, self.respPlot, self.pressurePlot, self.resistancePlot]:
            layout.addWidget(plot)

        # Create bottom bar for controls and labels
        self.bottomBar = QWidget()
        self.bottomBarLayout = QHBoxLayout()
        self.bottomBar.setLayout(self.bottomBarLayout)

        self.resistanceLabel = QLabel("Resistance:")
        self.bottomBarLayout.addWidget(self.resistanceLabel)

        self.modelType = QPushButton(f"{'Cell' if self.cellMode else 'Bath'} Mode")
        self.modelType.setStyleSheet(f"background-color: {'green' if self.cellMode else 'blue'}; color: white; border-radius: 5px; padding: 5px;")
        self.bottomBarLayout.addWidget(self.modelType)

        self.accessResistanceLabel = QLabel("Access Resistance: N/A")
        self.bottomBarLayout.addWidget(self.accessResistanceLabel)
        self.membraneResistanceLabel = QLabel("Membrane Resistance: N/A")
        self.bottomBarLayout.addWidget(self.membraneResistanceLabel)
        self.membraneCapacitanceLabel = QLabel("Membrane Capacitance: N/A")
        self.bottomBarLayout.addWidget(self.membraneCapacitanceLabel)

        # Make bottom bar height 20px
        self.bottomBar.setMaximumHeight(20)
        self.bottomBar.setMinimumHeight(20)
        self.bottomBarLayout.setContentsMargins(0, 0, 0, 0)

        # Add a pressure label
        self.pressureLabel = QLabel("Pressure:")
        self.bottomBarLayout.addWidget(self.pressureLabel)

        # Add pressure command box
        self.pressureCommandBox = QLineEdit()
        self.pressureCommandBox.setMaxLength(5)
        self.pressureCommandBox.setFixedWidth(100)
        self.pressureCommandBox.setPlaceholderText(f"{self.pressureController.measure()} mbar")
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(EPhysGraph.pressureLowerBound, EPhysGraph.pressureUpperBound))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)

        # Add pressure command slider
        self.pressureCommandSlider = QSlider(Qt.Horizontal)
        self.pressureCommandSlider.setValue(int(self.pressureController.measure()))
        self.pressureCommandSlider.setMinimum(EPhysGraph.pressureLowerBound)
        self.pressureCommandSlider.setMaximum(EPhysGraph.pressureUpperBound)
        self.pressureCommandSlider.setTickInterval(100)
        self.pressureCommandSlider.setTickPosition(QSlider.TicksBelow)
        self.pressureCommandSlider.valueChanged.connect(self.updatePressureLabel)
        self.pressureCommandSlider.sliderReleased.connect(self.pressureCommandSliderChanged)

        # Add up and down buttons for pressure adjustment
        self.upButton = QToolButton()
        self.upButton.setArrowType(Qt.UpArrow)
        self.downButton = QToolButton()
        self.downButton.setArrowType(Qt.DownArrow)
        self.upButton.setFixedWidth(50)
        self.downButton.setFixedWidth(50)
        self.upButton.clicked.connect(self.incrementPressure)
        self.downButton.clicked.connect(self.decrementPressure)

        self.bottomBarLayout.addWidget(self.pressureCommandSlider)
        self.bottomBarLayout.addWidget(self.pressureCommandBox)
        self.bottomBarLayout.addWidget(self.upButton)
        self.bottomBarLayout.addWidget(self.downButton)

        # Add an Atmospheric Pressure toggle button
        self.atmosphericPressureButton = QPushButton("ATM Pressure OFF")
        self.bottomBarLayout.addWidget(self.atmosphericPressureButton)

        # Add spacer to push everything to the left
        self.bottomBarLayout.addStretch(1)

        # Add bottom bar to the main layout
        layout.addWidget(self.bottomBar)

        # Set the main layout
        self.setLayout(layout)

        # Initialize timers
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 33  # ms (~30 Hz)
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        # Initialize a separate timer for pressure updates
        self.pressureUpdateTimer = QtCore.QTimer()
        self.pressureUpdateTimer.timeout.connect(self.update_pressure)
        self.pressureUpdateTimer.start(20)  # 50 Hz

        # Initialize data variables
        self.latestReadData = None
        self.latestrespData = None
        self.lastrespData = []
        self.lastReadData = []

        # Initialize recorder
        self.recorder = FileLogger(
            recording_state_manager,
            folder_path="experiments/Data/rig_recorder_data/",
            recorder_filename="graph_recording"
        )

        # Connect buttons
        self.atmosphericPressureButton.clicked.connect(self.togglePressure)
        self.modelType.clicked.connect(self.toggleModelType)

        # Connect the signal to the slot
        self.data_updated.connect(self.handle_data_update)

        # Start the background thread for DAQ data acquisition
        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
    
        self.daqUpdateThread.start()

        # Show window and bring to front
        self.raise_()
        self.show()



    def closeEvent(self, event):
        """
        Override the close event to properly close the recorder and hide the window.
        """
        self.recorder.close()
        logging.info("Closing EPhysGraph window")
        event.ignore()
        self.hide()

    def updateDAQDataAsync(self):
        """
        Background thread that continuously fetches data from the DAQ and emits signals for GUI updates.
        """
        while True:
            # sleep for 50 ms
            # time.sleep(0.010)
            if self.daq.isRunningProtocol:
                continue  # Don't run membrane test while running a current protocol
            try:
                # Fetch data from DAQ
                # start = time.perf_counter_ns()
                data = self.daq.getDataFromSquareWave(
                    wave_freq=40,
                    samplesPerSec=25000,
                    dutyCycle=0.5,
                    amplitude=0.5,
                    recordingTime=0.025
                )
                # end = time.perf_counter_ns()
                # logging.info(f"Time taken to get data from square wave: {(end-start)/1e6} ms")
            except Exception as e:
                logging.error(f"Error fetching data from DAQ: {e}")
                continue

            if data:
                latestrespData, latestReadData, totalResistance, MembraneResistance, AccessResistance, MembraneCapacitance = data

                # Emit the signal with the fetched DAQ data
                self.data_updated.emit(
                    totalResistance,
                    AccessResistance,
                    MembraneResistance,
                    MembraneCapacitance,
                    latestrespData,
                    latestReadData
                )

    @pyqtSlot(object, object, object, object, object, object)
    def handle_data_update(self, totalResistance, AccessResistance, MembraneResistance, MembraneCapacitance, latestrespData, latestReadData):
        """
        Slot to handle DAQ data updates emitted from the background thread.
        Updates GUI elements safely in the main thread.
        """
        try:
            # Update resistance labels
            if totalResistance is not None:
                self.resistanceDeque.append(totalResistance)
                self.resistanceLabel.setText(f"Total Resistance: {totalResistance:.2f} MΩ\t")
            else:
                self.resistanceLabel.setText("Total Resistance: N/A\t")

            if AccessResistance is not None:
                self.accessResistanceLabel.setText(f"Access Resistance: {AccessResistance:.2f} MΩ\t")
            else:
                self.accessResistanceLabel.setText("Access Resistance: N/A\t")

            if MembraneResistance is not None:
                self.membraneResistanceLabel.setText(f"Membrane Resistance: {MembraneResistance:.2f} MΩ\t")
            else:
                self.membraneResistanceLabel.setText("Membrane Resistance: N/A\t")

            # Update plotting data for cmdPlot and respPlot
            if latestReadData is not None:
                self.cmdPlot.clear()
                self.cmdPlot.plot(latestReadData[0, :], latestReadData[1, :])  # Removed pen color
                self.lastReadData = latestReadData

            if latestrespData is not None:
                self.respPlot.clear()
                self.respPlot.plot(latestrespData[0, :], latestrespData[1, :])  # Removed pen color
                self.lastrespData = latestrespData

            # Handle recording
            if self.recording_state_manager.is_recording_enabled():
                with self.pressure_lock:
                    currentPressureReading = self.latest_pressure
                timestamp = datetime.now().timestamp()
                try:
                    self.recorder.write_graph_data(
                        timestamp,
                        currentPressureReading,
                        totalResistance,
                        list(self.lastrespData[1, :]) if self.lastrespData is not None else [],
                        list(self.lastReadData[1, :]) if self.lastReadData is not None else []
                    )
                except Exception as e:
                    logging.error(f"Error in writing graph data to file: {e}, {self.lastrespData}")

        except Exception as e:
            logging.error(f"Error in handle_data_update: {e}", exc_info=True)

    def update_plot(self):
        """
        Periodically called by a QTimer to update the GUI plots.
        Updates pressure command box placeholder and resistance plot.
        """
        try:
            # Update pressure label based on slider value
            currentSliderValue = self.pressureCommandSlider.value()
            self.pressureCommandBox.setPlaceholderText(f"Set to: {currentSliderValue} mbar")

            # Update resistance plot
            self.resistancePlot.clear()
            displayDequeY = list(self.resistanceDeque)
            displayDequeX = list(range(len(displayDequeY)))
            self.resistancePlot.plot(displayDequeX, displayDequeY, pen="k")

        except Exception as e:
            logging.error(f"Error in update_plot: {e}", exc_info=True)

    def update_pressure(self):
        """
        Periodically called by a separate QTimer to update the pressure graph.
        """
        try:
            currentPressureReading = int(self.pressureController.measure())

            with self.pressure_lock:
                self.latest_pressure = currentPressureReading

            self.pressureData.append(currentPressureReading)

            self.pressurePlot.clear()
            pressureX = [i * self.updateDt / 1000 for i in range(len(self.pressureData))]
            self.pressurePlot.plot(pressureX, list(self.pressureData))

        except Exception as e:
            logging.error(f"Error in update_pressure: {e}", exc_info=True)

    def incrementPressure(self):
        """
        Increments the pressure setpoint by 5 mbar.
        """
        try:
            current_value = self.pressureCommandSlider.value()
            new_value = current_value + 5
            if new_value <= EPhysGraph.pressureUpperBound:
                self.pressureCommandSlider.setValue(new_value)
                self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release
        except Exception as e:
            logging.error(f"Error in incrementPressure: {e}", exc_info=True)

    def decrementPressure(self):
        """
        Decrements the pressure setpoint by 5 mbar.
        """
        try:
            current_value = self.pressureCommandSlider.value()
            new_value = current_value - 5
            if new_value >= EPhysGraph.pressureLowerBound:
                self.pressureCommandSlider.setValue(new_value)
                self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release
        except Exception as e:
            logging.error(f"Error in decrementPressure: {e}", exc_info=True)

    def togglePressure(self):
        """
        Toggles the atmospheric pressure setting.
        """
        try:
            if self.atmtoggle:
                self.atmosphericPressureButton.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
                self.atmosphericPressureButton.setText("ATM Pressure ON")
                self.pressureController.set_ATM(True)
            else:
                self.atmosphericPressureButton.setStyleSheet("")
                self.atmosphericPressureButton.setText("ATM Pressure OFF")
                self.pressureController.set_ATM(False)
            self.atmtoggle = not self.atmtoggle
        except Exception as e:
            logging.error(f"Error in togglePressure: {e}", exc_info=True)

    def toggleModelType(self):
        """
        Toggles between Cell Mode and Bath Mode.
        """
        try:
            if self.daq.cellMode:
                self.modelType.setStyleSheet("background-color: blue; color: white; border-radius: 5px; padding: 5px;")
                self.modelType.setText("Bath Mode")
                self.daq.setCellMode(False)
            else:
                self.modelType.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
                self.modelType.setText("Cell Mode")
                self.daq.setCellMode(True)

            logging.info(f"Cell Mode: {self.daq.cellMode}")
            self.cellMode = self.daq.cellMode
        except Exception as e:
            logging.error(f"Error in toggleModelType: {e}", exc_info=True)

    def updatePressureLabel(self, value):
        """
        Updates the pressure command box placeholder text based on the slider value.
        """
        try:
            self.pressureCommandBox.setPlaceholderText(f"Set to: {value} mbar")
        except Exception as e:
            logging.error(f"Error in updatePressureLabel: {e}", exc_info=True)

    def pressureCommandSliderChanged(self):
        """
        Manually change pressure setpoint based on slider position.
        """
        try:
            pressure = self.pressureCommandSlider.value()
            self.pressureController.set_pressure(pressure)
            self.pressureCommandBox.setPlaceholderText(f"{pressure} mbar")
        except Exception as e:
            logging.error(f"Error in pressureCommandSliderChanged: {e}", exc_info=True)

    def pressureCommandBoxReturnPressed(self):
        """
        Manually change pressure setpoint based on user input in the command box.
        """
        try:
            text = self.pressureCommandBox.text()
            self.pressureCommandBox.clear()

            pressure = float(text)
            # Clamp the pressure within bounds
            pressure = max(EPhysGraph.pressureLowerBound, min(EPhysGraph.pressureUpperBound, pressure))

            self.pressureController.set_pressure(pressure)
            self.setpoint = pressure
            self.pressureCommandSlider.setValue(int(pressure))
            self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release
        except ValueError:
            logging.warning("Invalid pressure input. Please enter a valid number.")
        except Exception as e:
            logging.error(f"Error in pressureCommandBoxReturnPressed: {e}", exc_info=True)
