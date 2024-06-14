import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PyQt5 import QtCore, QtGui


from pyqtgraph import PlotWidget

import threading

import numpy as np
from collections import deque
from holypipette.devices.amplifier import DAQ
from holypipette.devices.pressurecontroller import PressureController

from holypipette.utils import FileLogger

from datetime import datetime

__all__ = ["EPhysGraph", "CurrentProtocolGraph", "VoltageProtocolGraph"]

class CurrentProtocolGraph(QWidget):
    def __init__(self, daq : DAQ):
        super().__init__()

        layout = QVBoxLayout()
        self.setWindowTitle("Current Protocol")
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.daq = daq
        self.cprotocolPlot = PlotWidget()
        self.cprotocolPlot.setBackground('w')
        self.cprotocolPlot.getAxis('left').setPen('k')
        self.cprotocolPlot.getAxis('bottom').setPen('k')
        self.cprotocolPlot.setLabel('left', "Voltage", units='V')
        self.cprotocolPlot.setLabel('bottom', "Samples", units='')
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


    def update_plot(self):
        #is what we displayed the exact same?
        if self.latestDisplayedData == self.daq.latest_protocol_data or self.daq.latest_protocol_data is None:
            return
        
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        colors = ['k', 'r', 'g', 'b', 'y', 'm', 'c']
        self.cprotocolPlot.clear()
        for i, graph in enumerate(self.daq.latest_protocol_data):
            xData = graph[0]
            yData = graph[1]
            self.cprotocolPlot.plot(xData, yData, pen=colors[i])

        self.latestDisplayedData = self.daq.latest_protocol_data.copy()


class VoltageProtocolGraph(QWidget):
    def __init__(self, daq : DAQ):
        super().__init__()

        layout = QVBoxLayout()
        self.setWindowTitle("Voltage Protocol (Membrane Test)")
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.daq = daq
        self.vprotocolPlot = PlotWidget()
        self.vprotocolPlot.setBackground('w')
        self.vprotocolPlot.getAxis('left').setPen('k')
        self.vprotocolPlot.getAxis('bottom').setPen('k')
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

    def update_plot(self):
        #is what we displayed the exact same?

        # logging.warning("window should be shown")
        if np.array_equal(np.array(self.latestDisplayedData), np.array(self.daq.voltage_protocol_data)) or self.daq.voltage_protocol_data is None:
            return
        
        #if the window was closed or hidden, relaunch it
        if self.isHidden():
            self.setHidden(False)
            self.isShown = True

        if self.daq.voltage_protocol_data is not None:
            self.vprotocolPlot.clear()
            print(self.daq.voltage_protocol_data[0, :])
            print(self.daq.voltage_protocol_data[1, :])
            self.vprotocolPlot.plot(self.daq.voltage_protocol_data[0, :], self.daq.voltage_protocol_data[1, :])
            self.daq.latest_protocol_data = None

        self.latestDisplayedData = self.daq.voltage_protocol_data.copy()


class EPhysGraph(QWidget):
    """A window that plots electrophysiology data from the DAQ
    """
    
    def __init__(self, daq : DAQ, pressureController : PressureController, parent=None):
        super().__init__()

        #stop matplotlib font warnings
        logging.getLogger('matplotlib.font_manager').disabled = True

        self.daq = daq
        self.pressureController = pressureController

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
        self.squareWavePlot.setBackground('w')
        self.pressurePlot.setBackground('w')
        self.resistancePlot.setBackground('w')

        #set axis colors to black
        self.squareWavePlot.getAxis('left').setPen('k')
        self.squareWavePlot.getAxis('bottom').setPen('k')
        self.pressurePlot.getAxis('left').setPen('k')
        self.pressurePlot.getAxis('bottom').setPen('k')
        self.resistancePlot.getAxis('left').setPen('k')
        self.resistancePlot.getAxis('bottom').setPen('k')

        #set labels
        self.squareWavePlot.setLabel('left', "Current", units='A')
        self.squareWavePlot.setLabel('bottom', "Time", units='s')
        self.pressurePlot.setLabel('left', "Pressure", units='mbar')
        self.pressurePlot.setLabel('bottom', "Time", units='s')
        self.resistancePlot.setLabel('left', "Resistance", units='Ohms')
        self.resistancePlot.setLabel('bottom', "Samples", units='')


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
        
        self.resistanceLabel = QLabel()
        self.resistanceLabel.setText("Resistance: ")
        self.bottomBarLayout.addWidget(self.resistanceLabel)

        self.capacitanceLabel = QLabel()
        self.capacitanceLabel.setText("Capacitance: ")
        self.bottomBarLayout.addWidget(self.capacitanceLabel)

        #make bottom bar height 20px
        self.bottomBar.setMaximumHeight(20)
        self.bottomBar.setMinimumHeight(20)
        self.bottomBarLayout.setContentsMargins(0, 0, 0, 0)

        #add a pressure label
        self.pressureLabel = QLabel()
        self.pressureLabel.setText("Pressure: ")
        self.bottomBarLayout.addWidget(self.pressureLabel)
        layout.addWidget(self.bottomBar)

        #add pressure command box
        self.pressureCommandBox = QLineEdit()
        self.pressureCommandBox.setMaxLength(5)
        self.pressureCommandBox.setFixedWidth(100)
        self.pressureCommandBox.setValidator(QtGui.QIntValidator(-1000, 1000))


        # self.pressureCommandSlider = QSlider(Qt.Horizontal)
        # self.pressureCommandSlider.setMinimum(-500)
        # self.pressureCommandSlider.setMaximum(500)
        # self.pressureCommandSlider.setValue(20)
        # self.pressureCommandSlider.setTickInterval(100)
        # self.pressureCommandSlider.setTickPosition(QSlider.TicksBelow)

        # self.bottomBarLayout.addWidget(self.pressureCommandSlider)
        self.bottomBarLayout.addWidget(self.pressureCommandBox)

        #add spacer to push everything to the left
        self.bottomBarLayout.addStretch(1)

        self.setLayout(layout)
        
        self.updateTimer = QtCore.QTimer()
        # this has to match the arduino sensor delay
        self.updateDt = 33 # ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        # start async daq data update
        self.lastestDaqData = None
        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
        self.daqUpdateThread.start()
    
        # self.pressureUpdateThread = threading.Thread(target=self.updatePressureAsync, daemon=True)
        # self.pressureUpdateThread.start()

        self.recorder = FileLogger(folder_path="experiments/Data/rig_recorder_data/", recorder_filename = "graph_recording.csv")
        self.lastDaqData = []


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
        self.pressurePlot.plot(pressureX, self.pressureData, pen='k')

        # update resistance graph
        self.resistancePlot.clear()
        resistanceDeque = [i for i in range(len(self.resistanceDeque))]
        self.resistancePlot.plot(resistanceDeque, self.resistanceDeque, pen='k')

        # self.pressureCommandBox.setPlaceholderText("{:.2f} (mbar)".format(currentPressureReading))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)
        # self.pressureCommandSlider.sliderReleased.connect(self.pressureCommandSliderChanged)

        self.recorder.write_graph_data(datetime.now().timestamp(), currentPressureReading, list(self.resistanceDeque), list(self.lastDaqData[1, :]))
        self.lastestDaqData = None


    def pressureCommandSliderChanged(self):
        '''
        Manually change pressure setpoint
        '''

        # get text from box
        text = self.pressureCommandSlider.value()

        #try to convert to float
        try:
            pressure = float(text)
        except ValueError:
            return

        #set pressure
        self.pressureController.set_pressure(pressure)

    def pressureCommandBoxReturnPressed(self):
        '''
        Manually change pressure setpoint
        '''

        # get text from box
        text = self.pressureCommandBox.text()
        self.pressureCommandBox.clear()

        #try to convert to float
        try:
            pressure = float(text)
            # set pressure
            self.pressureController.set_pressure(pressure)
        except ValueError:
            return
