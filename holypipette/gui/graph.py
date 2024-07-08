import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt


from pyqtgraph import PlotWidget

import threading

import numpy as np
from collections import deque
from holypipette.devices.amplifier import DAQ
from holypipette.devices.pressurecontroller import PressureController
from holypipette.utils.RecordingStateManager import RecordingStateManager
from holypipette.utils import FileLogger
from holypipette.utils import EPhysLogger

from datetime import datetime

__all__ = ["EPhysGraph", "CurrentProtocolGraph", "VoltageProtocolGraph", "HoldingProtocolGraph"]

global_index = 0

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
        self.cprotocolPlot.setLabel("bottom", "Samples", units = "")
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
        self.updateDt = 100 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.ephys_logger = EPhysLogger(ephys_filename = "CurrentProtocol", recording_state_manager = self.recording_state_manager)

    def update_plot(self):
        # is what we displayed the exact same?
        if self.latestDisplayedData == self.daq.current_protocol_data or self.daq.current_protocol_data is None:
            return
        
        index = self.recording_state_manager.sample_number
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True
        # curr = self.daq.latest_protocol_data
        # logging.info('length of current protocol data: ' + str(len(curr[0])) + ' ' + str(len(curr[1])))
        colors = ["k", 'r', 'g', 'b', 'y', 'm', 'c']
        self.cprotocolPlot.clear()

        save_data = None
        temp_data = deque(self.daq.current_protocol_data.copy())
        timestamp = datetime.now().timestamp()
        for i, graph in enumerate(self.daq.current_protocol_data):
            # logging.info(f"enumerating graph: {i}")
        #     #logging the data type of self.daq.latest_protocol_data
        #     # logging.info(f"data type: {type(self.daq.current_protocol_data)}")
        
        #     # print("Enumerating graph:, ", i, graph)
            save_data = temp_data.popleft()

            xData = graph[0]
            yData = graph[1]
            # print(len(xData), len(yData))
            self.cprotocolPlot.plot(xData, yData, pen=colors[i])
            save_data = np.array([xData, yData])
            # logging.info("writing current ephys data to file")
            self.ephys_logger.write_ephys_data(timestamp, index, save_data, colors[i])
            if i == 5:
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
        self.vprotocolPlot.setLabel('left', "PicoAmps", units='pA')
        self.vprotocolPlot.setLabel('bottom', "Samples", units='')
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
        self.updateDt = 100 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.ephys_logger = EPhysLogger(ephys_filename = "VoltageProtocol", recording_state_manager=self.recording_state_manager)

    def update_plot(self):
        #is what we displayed the exact same?

        # logging.warning("window should be shown")
        if np.array_equal(np.array(self.latestDisplayedData), np.array(self.daq.voltage_protocol_data)) or self.daq.voltage_protocol_data is None:
            return

        index = self.recording_state_manager.sample_number
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
            timestamp = datetime.now().timestamp()
            
            # logging.info("writing Voltage ephys data to file")
            self.ephys_logger.write_ephys_data(timestamp, index, self.daq.voltage_protocol_data, colors[0])
            self.latestDisplayedData = self.daq.voltage_protocol_data.copy()
            self.daq.voltage_protocol_data = None
        


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
        self.hprotocolPlot.setLabel('left', "PicoAmps", units='pA')
        self.hprotocolPlot.setLabel('bottom', "Samples", units='')
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
        self.updateDt = 10 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.ephys_logger = EPhysLogger(ephys_filename = "HoldingProtocol", recording_state_manager = self.recording_state_manager)

    def update_plot(self):
        # logging.warning("window should be shown")
        # is what we displayed the exact same?
        if np.array_equal(np.array(self.latestDisplayedData), np.array(self.daq.holding_protocol_data)) or self.daq.holding_protocol_data is None:
            # logging.warning("no new data, skipping plot update")
            return
        
        index = self.recording_state_manager.sample_number

        # logging.warning("new data, updating plot")
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        if self.daq.holding_protocol_data is not None:
            self.hprotocolPlot.clear()

            colors = ["k"]
            self.hprotocolPlot.plot(self.daq.holding_protocol_data[0, :], self.daq.holding_protocol_data[1, :], pen=colors[0])
            timestamp = datetime.now().timestamp()
            self.ephys_logger.write_ephys_data(timestamp, index, self.daq.holding_protocol_data, colors[0])
            self.latestDisplayedData = self.daq.holding_protocol_data.copy()
            self.daq.holding_protocol_data = None

        # self.latestDisplayedData = self.daq.holding_protocol_data.copy()


class EPhysGraph(QWidget):
    """A window that plots electrophysiology data from the DAQ
    """
    def __init__(self, daq : DAQ, pressureController : PressureController, recording_state_manager, parent = None):
        super().__init__()

        #stop matplotlib font warnings
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.atmtoggle = True
        # self.atmtogglecount = 0
        self.daq = daq
        self.pressureController = pressureController
        # self.recording_state_manager = recording_state_manager  # Include the state manager in the graph
        self.setpoint = 0

        #constants for Multi Clamp
        self.externalCommandSensitivity = 20 #mv/V
        self.triggerLevel = 0.05 #V

        #setup window
        self.setWindowTitle("Electrophysiology")

        self.squareWavePlot = PlotWidget()
        # * numerator below is ms!
        # self.squareWavePlot.setXRange(0, 10/1000, padding=0)
        self.pressurePlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        #set background color of plots
        self.squareWavePlot.setBackground("w")
        self.pressurePlot.setBackground("w")
        self.resistancePlot.setBackground("w")

        #set axis colors to black
        self.squareWavePlot.getAxis("left").setPen("k")
        self.squareWavePlot.getAxis("bottom").setPen("k")
        self.pressurePlot.getAxis("left").setPen("k")
        self.pressurePlot.getAxis("bottom").setPen("k")
        self.resistancePlot.getAxis("left").setPen("k")
        self.resistancePlot.getAxis("bottom").setPen("k")

        #set labels
        self.squareWavePlot.setLabel("left", "Current", units="A")
        self.squareWavePlot.setLabel("bottom", "Time", units="s")
        self.pressurePlot.setLabel("left", "Pressure", units="mbar")
        self.pressurePlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", "Resistance", units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")


        # self.pressureData = deque([0.0]*100, maxlen=100)
        self.pressureData = deque(maxlen=100)
        self.resistanceDeque = deque(maxlen=100)
        # print(self.pressureData)

        #create a quarter layout for 4 graphs
        layout = QVBoxLayout()
        layout.addWidget(self.squareWavePlot)
        layout.addWidget(self.pressurePlot)
        layout.addWidget(self.resistancePlot)

        #make resistance plot show current resistance in text
        self.bottomBar = QWidget()
        self.bottomBarLayout = QHBoxLayout()
        self.bottomBar.setLayout(self.bottomBarLayout)
        
        self.resistanceLabel = QLabel("Resistance:")
        self.bottomBarLayout.addWidget(self.resistanceLabel)

        self.capacitanceLabel = QLabel("Capacitance:")
        self.bottomBarLayout.addWidget(self.capacitanceLabel)

        #make bottom bar height 20px
        self.bottomBar.setMaximumHeight(20)
        self.bottomBar.setMinimumHeight(20)
        self.bottomBarLayout.setContentsMargins(0, 0, 0, 0)

        # add a pressure label
        self.pressureLabel = QLabel("Pressure:")
        self.bottomBarLayout.addWidget(self.pressureLabel)
        layout.addWidget(self.bottomBar)

        #add pressure command box
        self.pressureCommandBox = QLineEdit()
        self.pressureCommandBox.setMaxLength(5)
        self.pressureCommandBox.setFixedWidth(100)
        self.pressureCommandBox.setPlaceholderText(f"{self.pressureController.measure()} mbar")
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(-1000, 1000))

        self.pressureCommandSlider = QSlider(Qt.Horizontal)
        self.pressureCommandSlider.setValue(int(self.pressureController.measure()))
        self.pressureCommandSlider.setMinimum(-400)
        self.pressureCommandSlider.setMaximum(700)
        self.pressureCommandSlider.setTickInterval(100)
        self.pressureCommandSlider.setTickPosition(QSlider.TicksBelow)
        self.pressureCommandSlider.valueChanged.connect(self.updatePressureLabel)
        self.pressureCommandSlider.sliderReleased.connect(self.pressureCommandSliderChanged)

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
        # add an Atmospheric Pressure toggle button
        self.atmosphericPressureButton = QPushButton("ATM Pressure OFF")
        self.bottomBarLayout.addWidget(self.atmosphericPressureButton)

        #add spacer to push everything to the left
        self.bottomBarLayout.addStretch(1)

        self.setLayout(layout)
        
        self.updateTimer = QtCore.QTimer()
        # this has to match the arduino sensor delay
        self.updateDt = 20 # ms roughly 33 Hz with lag added
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        # start async daq data update
        self.lastestDaqData = None
        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
        self.daqUpdateThread.start()
    

        self.recorder = FileLogger(recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", recorder_filename="graph_recording")
        self.lastDaqData = []

        self.atmosphericPressureButton.clicked.connect(self.togglePressure)
        #show window and bring to front
        self.raise_()
        self.show()


    def close(self):
        self.recorder.close()
        logging.info("closing graph")
        super(EPhysGraph, self).hide()

    def updateDAQDataAsync(self):
        while True:
            # This slows down the rate your graph gets updated (bc the graph only gets updated if there is NEW data)
            # time.sleep(0.1)

            if self.daq.isRunningProtocol:
                continue # don't run membrane test while running a current protocol

            # * setting frequency to 100Hz fixed the resistance chart on bath mode but isn't needed on cell mode (it can be 10Hz??)
            # self.lastestDaqData, resistance = self.daq.getDataFromSquareWave(100, 50000, 0.5, 0.5, 0.1)
            # self.lastestDaqData, resistance = self.daq.getDataFromSquareWave(100, 50000, 0.5, 0.5, 0.03)
            # * best option so far is below, should we make it more flexible? --> sometimes the min appears before the max, messing up the gradient calculation and
            # * subsequent shiftin in daq.getDataFromSquareWave
            self.lastestDaqData, resistance = self.daq.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
            if resistance is not None:
                self.resistanceDeque.append(resistance)
                self.resistanceLabel.setText("Resistance: {:.2f} MOhms\t".format(resistance / 1e6))
                self.capacitanceLabel.setText("Capacitance: {:.2f} pF\t".format(resistance / 1e-12))

    def update_plot(self):
        # update current graph

        if self.lastestDaqData is not None:
            self.squareWavePlot.clear()
            self.squareWavePlot.plot(self.lastestDaqData[0, :], self.lastestDaqData[1, :])
            self.lastDaqData = self.lastestDaqData
            # self.lastestDaqData = None
        
        #update pressure graph
        currentPressureReading = int(self.pressureController.measure())
        self.pressureData.append(currentPressureReading)

        # print(pressureX)
        # print(len(pressureX))
        self.pressurePlot.clear()
        pressureX = [i * self.updateDt / 1000 for i in range(len(self.pressureData))]
        self.pressurePlot.plot(pressureX, self.pressureData, pen="k")

        # update resistance graph
        self.resistancePlot.clear()
        displayDequeY = self.resistanceDeque.copy()
        displayDequeX = [i for i in range(len(displayDequeY))]
        # resistanceDeque = [i for i in range(len(self.resistanceDeque))]
        self.resistancePlot.plot(displayDequeX, displayDequeY, pen="k")

        # self.pressureCommandBox.setPlaceholderText("{:.2f} (mbar)".format(currentPressureReading))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)
        # self.atmosphericPressureButton.clicked.connect(self.togglePressure)

        # logging.debug("graph updated") # uncomment for debugging in log.csv file

        try:
            self.recorder.write_graph_data(datetime.now().timestamp(), currentPressureReading, displayDequeY[-1], list(self.lastDaqData[1, :]))
        except Exception as e:
            logging.error(f"Error in writing graph data to file: {e}, {self.lastDaqData}")

        self.lastestDaqData = None

    def incrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value + 5
        if new_value <= 500:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release

    def decrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value - 5
        if new_value >= -500:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release


    def togglePressure(self):
        if self.atmtoggle:
            self.atmosphericPressureButton.setStyleSheet("background-color: green; color: white;border-radius: 5px; padding: 5px;")
            self.atmosphericPressureButton.setText("ATM Pressure ON")
            self.pressureController.set_ATM(True)
        else:
            self.atmosphericPressureButton.setStyleSheet("")
            self.atmosphericPressureButton.setText("ATM Pressure OFF")
            self.pressureController.set_ATM(False)
        self.atmtoggle = not self.atmtoggle
        # self.atmtogglecount += 1
        # self.pressureCommandBox.setPlaceholderText(f"{pressure} mbar")

        # logging.info(f"toggle pressure called: {self.atmtogglecount} times")

    def updatePressureLabel(self, value):
        self.pressureCommandBox.setPlaceholderText(f"Set to: {value} mbar")

    def pressureCommandSliderChanged(self):
        '''
        Manually change pressure setpoint
        '''
        # get text from box
        text = self.pressureCommandSlider.value()

        try:
            pressure = float(text)
        except ValueError:
            return

        #set pressure
        self.pressureController.set_pressure(pressure)
        self.pressureCommandBox.setPlaceholderText(f"{pressure} mbar")

    def pressureCommandBoxReturnPressed(self):
        '''
        Manually change pressure setpoint
        '''
        # get text from box
        text = self.pressureCommandBox.text()
        self.pressureCommandBox.clear()

        try:
            pressure = float(text)
            # set pressure
            self.pressureController.set_pressure(pressure)
            self.setpoint = pressure
            self.pressureCommandSlider.setValue(int(pressure))
            self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release
        except ValueError:
            return