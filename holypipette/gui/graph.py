import logging

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PyQt5 import QtCore, QtGui

from pyqtgraph import PlotWidget, plot

import threading
import time

import numpy as np
from collections import deque
from holypipette.devices.amplifier import DAQ
from holypipette.devices.pressurecontroller import PressureController

__all__ = ["EPhysGraph", "CurrentProtocolGraph"]

class CurrentProtocolGraph(QWidget):
    def __init__(self, daq : DAQ):
        super().__init__()

        layout = QVBoxLayout()
        self.setWindowTitle("Current Protocol")
        logging.getLogger('matplotlib.font_manager').disabled = True
        self.daq = daq
        self.protocolPlot = PlotWidget()
        self.protocolPlot.setBackground('w')
        self.protocolPlot.getAxis('left').setPen('k')
        self.protocolPlot.getAxis('bottom').setPen('k')
        self.protocolPlot.setLabel('left', "Voltage", units='V')
        self.protocolPlot.setLabel('bottom', "Samples", units='')
        layout.addWidget(self.protocolPlot)

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
        self.protocolPlot.clear()
        for i, graph in enumerate(self.daq.latest_protocol_data):
            xData = graph[0]
            yData = graph[1]
            self.protocolPlot.plot(xData, yData, pen=colors[i])

        self.latestDisplayedData = self.daq.latest_protocol_data.copy()

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


        self.pressureData = deque(maxlen=100)
        self.resistanceDeque = deque(maxlen=100)

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
        self.bottomBarLayout.addWidget(self.pressureCommandBox)

        #add spacer to push everything to the left
        self.bottomBarLayout.addStretch(1)

        self.setLayout(layout)
        
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 100 #ms
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        # start async daq data update
        self.lastestDaqData = None
        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
        self.daqUpdateThread.start()

        #show window and bring to front
        self.raise_()
        self.show()

    def updateDAQDataAsync(self):
        while True:
            # This slows down the rate your graph gets updated (bc the graph only gets updated if there is NEW data)
            # time.sleep(0.1)

            if self.daq.isRunningCurrentProtocol:
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

    def update_plot(self):
        # update current graph

        if self.lastestDaqData is not None:
            self.squareWavePlot.clear()
            self.squareWavePlot.plot(self.lastestDaqData[0, :], self.lastestDaqData[1, :])
            self.lastestDaqData = None
        
        #update pressure graph
        self.pressureData.append(self.pressureController.measure())
        pressureX = [i * self.updateDt / 1000 for i in range(len(self.pressureData))]
        self.pressurePlot.clear()
        self.pressurePlot.plot(pressureX, self.pressureData, pen='k')

        #update resistance graph
        self.resistancePlot.clear()
        resistanceDeque = [i for i in range(len(self.resistanceDeque))]
        self.resistancePlot.plot(resistanceDeque, self.resistanceDeque, pen='k')

        self.pressureCommandBox.setPlaceholderText("{:.2f} (mbar)".format(self.pressureController.measure()))
        self.pressureCommandBox.returnPressed.connect(self.pressureCommandBoxReturnPressed)

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
        except ValueError:
            return

        #set pressure
        self.pressureController.set_pressure(pressure)