import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton, QDesktopWidget
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from matplotlib.colors import LinearSegmentedColormap, to_hex


from pyqtgraph import PlotWidget
from pyqtgraph.exporters import ImageExporter
import io
from PIL import Image

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
        # make a color gradient based on a list
        color_range = self.daq.pulseRange
        #make a gradient of colors based off of color_range, a value that should describe the number of pulses (as letters)
        colors = [format((i / color_range), ".2f") for i in range(color_range)]
        start_color = "#003153" #Prussian Blue
        end_color =  "#ffffff" #White
        cmap = LinearSegmentedColormap.from_list("", [start_color, end_color])
        colors = [to_hex(cmap(float(i) / color_range)) for i in range(color_range)]
        pulses = self.daq.pulses
        # colors = ["k", 'r', 'g', 'b', 'y', 'm', 'c']
        self.cprotocolPlot.clear()

        save_data = None
        temp_data = deque(self.daq.current_protocol_data.copy())
        # identify the shape of temp_data
    
        
        print("temp_data: ", temp_data)
        timestamp = datetime.now().timestamp()
        for i, graph in enumerate(self.daq.current_protocol_data):
            # logging.info(f"enumerating graph: {i}")
        #     #logging the data type of self.daq.latest_protocol_data
        #     # logging.info(f"data type: {type(self.daq.current_protocol_data)}")
        
        #     # print("Enumerating graph:, ", i, graph)
            save_data = temp_data.popleft()

            # print("Graph: ", graph)
            timeData = graph[0]
            respData = graph[1]
            readData = graph[2]
            # print("timeData: ", len(timeData))
            # print("respData: ", len(respData))
            # print("readData: ", len(readData))
            self.cprotocolPlot.plot(timeData, respData, pen=colors[i])
            # self.cprotocolPlot.plot(timeData, respData, pen="b")
            logging.info("writing current ephys data to file")
            # convert pulses to string with _ before it
            
            pulse = str(pulses[i])
            marker = colors[i] + "_" + pulse
            self.ephys_logger.write_ephys_data(timestamp, index, timeData, readData, respData, marker)
            self.ephys_logger.save_ephys_plot(timestamp, index, self.cprotocolPlot)
            if i == colors[-1]:
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
            timeData = self.daq.voltage_protocol_data[0, :]
            respData = self.daq.voltage_protocol_data[1, :]
            readData = self.daq.voltage_command_data[1,:]
            
            # logging.info("writing Voltage ephys data to file")
            self.ephys_logger.write_ephys_data(timestamp, index, timeData, readData, respData, colors[0])
            self.ephys_logger.save_ephys_plot(timestamp, index, self.vprotocolPlot)
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

        self.ephys_logger = EPhysLogger(ephys_filename = "HoldingProtocol", recording_state_manager = self.recording_state_manager)

    def update_plot(self):
        # logging.warning("window should be shown")
        # is what we displayed the exact same?
        if self.daq.holding_protocol_data is None:
            # logging.warning("no new data, skipping plot update")
            return
        
        index = self.recording_state_manager.sample_number
        # logging.warning("new data, updating plot")
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        self.hprotocolPlot.clear()

        colors = ["k"]
        self.hprotocolPlot.plot(self.daq.holding_protocol_data[0, :], self.daq.holding_protocol_data[1, :], pen=colors[0])
        timestamp = datetime.now().timestamp()
        self.ephys_logger.write_ephys_data(timestamp, index, self.daq.holding_protocol_data[0,:],self.daq.holding_protocol_data[1,:],self.daq.holding_protocol_data[2,:], colors[0])
        self.ephys_logger.save_ephys_plot(timestamp, index, self.hprotocolPlot)
    
        self.latestDisplayedData = self.daq.holding_protocol_data.copy()
        self.daq.holding_protocol_data = None

        # self.latestDisplayedData = self.daq.holding_protocol_data.copy()
class EPhysGraph(QWidget):
    """
    A window that plots electrophysiology data from the DAQ
    """

    pressureLowerBound = -450
    pressureUpperBound = 730

    def __init__(self, daq : DAQ, pressureController : PressureController, recording_state_manager: RecordingStateManager):
        super().__init__()

        #stop matplotlib font warnings
        logging.getLogger("matplotlib.font_manager").disabled = True
        self.atmtoggle = True
        self.daq = daq

        # I'm scared of pointers and .copy() doesn't work on bools :/
        self.cellMode = False
        # self.cellMode = self.daq.cellMode

        self.pressureController = pressureController
        # self.recording_state_manager = recording_state_manager  # Include the state manager in the graph
        self.setpoint = 0

        #constants for Multi Clamp
        self.externalCommandSensitivity = 20 # mv/V
        self.triggerLevel = 0.05 # V

        #setup window
        self.setWindowTitle("Electrophysiology")

        self.cmdPlot = PlotWidget()
        self.respPlot = PlotWidget()
        # * numerator below is ms!
        # self.squareWavePlot.setXRange(0, 10/1000, padding=0)
        self.pressurePlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        #set background color of plots
        self.cmdPlot.setBackground("w")
        self.respPlot.setBackground("w")
        self.pressurePlot.setBackground("w")
        self.resistancePlot.setBackground("w")

        #set axis colors to black
        for plot in [self.cmdPlot, self.respPlot, self.resistancePlot, self.pressurePlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        # set labels
        # * This "A" makes things annoying for the recorder and Daq, but its serves as a base unit. If set to pA, then if the data is 3 orders of magnitude higher,
        # * it will show the label as kpA instead of nA.
        # * This means we have to keep doing some "conversions" to make the graph happy
        self.cmdPlot.setLabel("left", "Command Voltage", units="V")
        self.cmdPlot.setLabel("bottom", "Time", units="s")
        self.respPlot.setLabel("left", "Current (resp)", units="A")
        self.respPlot.setLabel("bottom", "Time", units="s")
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
        for plot in [self.cmdPlot, self.respPlot, self.pressurePlot, self.resistancePlot]:
            layout.addWidget(plot)


        #make resistance plot show current resistance in text
        self.bottomBar = QWidget()
        self.bottomBarLayout = QHBoxLayout()
        self.bottomBar.setLayout(self.bottomBarLayout)
        
        self.resistanceLabel = QLabel("Resistance:")
        self.bottomBarLayout.addWidget(self.resistanceLabel)

        self.modelType = QPushButton(f"{'Cell' if self.cellMode else 'Bath'} Mode")
        self.modelType.setStyleSheet(f"background-color: {'green' if self.cellMode else 'blue'}; color: white; border-radius: 5px; padding: 5px;")

        self.bottomBarLayout.addWidget(self.modelType)
        self.accessResistanceLabel = QLabel("Access Resistance: NA")
        self.bottomBarLayout.addWidget(self.accessResistanceLabel)
        self.membraneResistanceLabel = QLabel("Membrane Resistance: NA")
        self.bottomBarLayout.addWidget(self.membraneResistanceLabel)
        self.membraneCapacitanceLabel = QLabel("Membrane Capacitance: NA")
        self.bottomBarLayout.addWidget(self.membraneCapacitanceLabel)

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
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(EPhysGraph.pressureLowerBound, EPhysGraph.pressureUpperBound))

        self.pressureCommandSlider = QSlider(Qt.Horizontal)
        self.pressureCommandSlider.setValue(int(self.pressureController.measure()))
        self.pressureCommandSlider.setMinimum(EPhysGraph.pressureLowerBound)
        self.pressureCommandSlider.setMaximum(EPhysGraph.pressureUpperBound)
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
        self.latestReadData = None
        self.latestrespData = None
        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
        self.daqUpdateThread.start()
    

        self.recorder = FileLogger(recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", recorder_filename="graph_recording")
        self.lastrespData = []
        self.lastReadData = []

        self.atmosphericPressureButton.clicked.connect(self.togglePressure)
        self.modelType.clicked.connect(self.toggleModelType)
        #show window and bring to front
        self.raise_()
        self.show()

    def location_on_the_screen(self):
        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()
        # print(f"Available geometry: {ag.width()} x {ag.height()}")
        # print(ag.width(), ag.height())
        # print(f"Screen geometry: {sg.width()} x {sg.height()}")
        # print(sg.width(), sg.height())
    
        x = ag.width() // 2    # Adjusted calculation for x coordinate
        y = 30  # Adjusted calculation for y coordinate
        width = ag.width() // 2
        height = ag.height() - 30    # Subtract 50 from the total height
        # print(f"x: {x}, y: {y}")
        # print(x, y)
        self.setGeometry(x, y, width, height)

    def close(self):
        self.recorder.close()
        logging.info("closing graph")
        super(EPhysGraph, self).hide()

    def updateDAQDataAsync(self):
        while True:
            # This slows down the rate your graph gets updated (bc the graph only gets updated if there is NEW data)
            time.sleep(0.01)

            if self.daq.isRunningProtocol:
                continue # don't run membrane test while running a current protocol

            # * setting frequency to 100Hz fixed the resistance chart on bath mode but isn't needed on cell mode (it can be 10Hz??)
            # * best option so far is below, should we make it more flexible? --> sometimes the min appears before the max, messing up the gradient calculation and
            # * subsequent shift in in daq.getDataFromSquareWave
            # 1 = 20mV  -> 5V on the oscilloscope
            # 0.5 = 10mV -> 2.5V on the oscilloscope
            # wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime
            self.latestrespData, self.latestReadData, totalResistance, MembraneResistance, AccessResistance, MembraneCapacitance = self.daq.getDataFromSquareWave(wave_freq = 20, samplesPerSec = 50000, dutyCycle = 0.5, amplitude = 0.5, recordingTime= 0.05)


            # logging.debug("DAQ data updated")
            # print("Total Resistance: ", totalResistance)
            # print("Access Resistance: ", AccessResistance)
            # print("Membrane Resistance: ", MembraneResistance)
            # print("Membrane Capacitance: ", MembraneCapacitance)


            if totalResistance is not None:
                self.resistanceDeque.append(totalResistance)
                self.resistanceLabel.setText("Total Resistance: {:.2f} M Ohms\t".format(totalResistance))
                # print("Total Resistance: ", totalResistance)
                # print("Access Resistance: ", AccessResistance)
                # print("Membrane Resistance: ", MembraneResistance)
                # print("Membrane Capacitance: ", MembraneCapacitance)
            if AccessResistance is not None:
                self.accessResistanceLabel.setText("Access Resistance: {:.2f} MΩ\t".format(AccessResistance))
            if MembraneResistance is not None:
                self.membraneResistanceLabel.setText("Membrane Resistance: {:.2f} MΩ\t".format(MembraneResistance))
            if MembraneCapacitance is not None:
                self.membraneCapacitanceLabel.setText("Membrane Capacitance: {:.2f} pF\t".format(MembraneCapacitance))

    def update_plot(self):
        # update current graph

        if self.latestReadData is not None:
            # print("Latest Read Data: ", self.latestReadData[1, :])
            self.cmdPlot.clear()
            self.cmdPlot.plot(self.latestReadData[0, :], self.latestReadData[1, :])
            self.lastReadData = self.latestReadData
            self.latestReadData = None

        if self.latestrespData is not None:
            self.respPlot.clear()
            self.respPlot.plot(self.latestrespData[0, :], self.latestrespData[1, :])
            self.lastrespData = self.latestrespData
            self.latestrespData = None
     
        
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
            self.recorder.write_graph_data(datetime.now().timestamp(), currentPressureReading, displayDequeY[-1], list(self.lastrespData[1, :]), list(self.lastReadData[1, :]))
        except Exception as e:
            logging.error(f"Error in writing graph data to file: {e}, {self.lastrespData}")

    def incrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value + 5
        if new_value <= EPhysGraph.pressureUpperBound:
            self.pressureCommandSlider.setValue(new_value)
            self.pressureCommandSlider.sliderReleased.emit()  # Simulate slider release

    def decrementPressure(self):
        current_value = self.pressureCommandSlider.value()
        new_value = current_value - 5
        if new_value >= EPhysGraph.pressureLowerBound:
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
    
    def toggleModelType(self):
        if self.daq.cellMode:
            self.modelType.setStyleSheet("background-color: blue; color: white;border-radius: 5px; padding: 5px;")
            self.modelType.setText("Bath Mode")
            self.daq.setCellMode(False)
        else:
            self.modelType.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 5px;")
            self.modelType.setText("Cell Mode")
            self.daq.setCellMode(True)

        logging.info("Cell Mode: ", self.daq.cellMode)
        self.cellMode = self.daq.cellMode
        print("Graph Cell Mode: ", self.cellMode)

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