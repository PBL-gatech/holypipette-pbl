import nidaqmx
import nidaqmx.system
import nidaqmx.constants

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal as signal
import time
import threading
import logging
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QToolButton, QDesktopWidget, QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget

from collections import deque
from datetime import datetime

__all__ = ["DAQtest", "EPhysGraph"]

class DAQtest:
    def __init__(self, readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel):
        self.readDev = readDev
        self.cmdDev = cmdDev
        self.respDev = respDev
        self.readChannel = readChannel
        self.cmdChannel = cmdChannel
        self.respChannel = respChannel
        self.latestAccessResistance = None
        self.totalResistance = None
        self.latestMembraneResistance = None
        self.latestMembraneCapacitance = None

        self.isRunningProtocol = False
        self.isRunningVoltageProtocol = False
        self._deviceLock = threading.Lock()

        self.latest_protocol_data = None
        self.current_protocol_data = None
        self.voltage_protocol_data = None
        self.holding_protocol_data = None

        self.cellMode = False

        logging.info(f'Using {self.readDev}/{self.readChannel} for reading the output of {self.cmdDev}/{self.cmdChannel} and {self.respDev}/{self.respChannel} for response.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        numSamples = int(samplesPerSec * recordingTime)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{self.readDev}/{self.readChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.ai_channels.add_ai_voltage_chan(f'{self.respDev}/{self.respChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numSamples)
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            task.stop()

        if data is None or np.where(data == None)[0].size > 0:
            data = np.zeros((2, numSamples))
            
        return data
    
    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(f'{self.cmdDev}/{self.cmdChannel}')
        task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        
        numSamples = int(samplesPerSec * recordingTime)
        data = np.zeros(numSamples)
        
        period = int(samplesPerSec / wave_freq)
        onTime = int(period * dutyCycle)

        for i in range(0, numSamples, period):
            data[i:i+onTime] = amplitude
            data[i+onTime:i+period] = 0

        task.write(data)
        
        return task, data

    def setCellMode(self, mode: bool) -> None:
        self.cellMode = mode

    def getDataFromSquareWave(self, wave_freq, samplesPerSec: int, dutyCycle, amplitude, recordingTime) -> tuple:
        self._deviceLock.acquire()
        sendTask, cmdData = self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        sendTask.start()
        data = self._readAnalogInput(samplesPerSec, recordingTime)
        sendTask.stop()
        sendTask.close()
        self._deviceLock.release()
        
        triggeredSamples = data.shape[1]
        xdata = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype = float)

        # Gradient
        gradientData = np.gradient(data[0], xdata)
        max_index = np.argmax(gradientData)
        # Find the index of the minimum value after the maximum value
        min_index = np.argmin(gradientData[max_index:]) + max_index


        respData = data[0]
        readData = data[1]
        
        # #Truncate the array
        # # left_bound = 10
        # # right_bound = 150
        # # * bound is arbitrary, just to make it look good on the graph
        # # respData = data[0][max_index - left_bound:min_index + right_bound]
        # # readData = data[1][max_index - left_bound:min_index + right_bound]
        # xdata = xdata[max_index - left_bound:min_index + right_bound]

        respData *= 1e-9
        readData *= 1e-9
        amplitude *= 1e-2

        self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = self._getParamsfromCurrent(respData, xdata, amplitude)
        self.totalResistance = self._getResistancefromCurrent(respData, amplitude)

        return np.array([xdata, respData]),np.array([xdata, readData]), self.totalResistance, self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance
    
    def _getParamsfromCurrent(self, data, xdata, cmdVoltage) -> tuple:
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None

        try:
            df = pd.DataFrame({'X': xdata, 'Y': data})
            df['X_ms'] = df['X'] * 1000
            df['Y_pA'] = df['Y']
            filtered_data, pre_filtered_data, post_filtered_data, plot_params, I_prev_pA, I_post_pA = self.filter_data(df)
            peak_time, peak_index, min_time, min_index = plot_params
            m, t, b = self.optimizer(filtered_data)
            if m is not None and t is not None and b is not None:
                tau = 1 / t
                I_peak_pA = df.loc[peak_index, 'Y_pA']
                R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, cmdVoltage, I_peak_pA, I_prev_pA, I_post_pA)
        except Exception as e:
            return None, None, None

        return R_a_MOhms, R_m_MOhms, C_m_pF
    
    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])

            resistance = cmdVoltage / (highAvg - lowAvg)

            return resistance
            
        except Exception as e:
            return None


class EPhysGraph(QWidget):
    def __init__(self, daq):
        super().__init__()

        self.daq = daq
        self.cellMode = self.daq.cellMode

        self.setWindowTitle("Electrophysiology")

    
        self.cmdPlot = PlotWidget()
        self.respPlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        for plot in [self.cmdPlot, self.respPlot, self.resistancePlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        self.cmdPlot.setLabel("left", "Command Voltage", units="V")
        self.cmdPlot.setLabel("bottom", "Time", units="s")
        self.respPlot.setLabel("left", "Current (resp)", units="A")
        self.respPlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", "Resistance", units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")


        self.latestReadData = None
        self.latestRespData = None
        self.resistanceDeque = deque(maxlen=100)  # Initialize the resistance deque

        layout = QVBoxLayout()
        layout.addWidget(self.cmdPlot)
        layout.addWidget(self.respPlot)
        layout.addWidget(self.resistancePlot)
        self.setLayout(layout)
        
        self.updateTimer = QtCore.QTimer()
        self.updateDt = 20
        self.updateTimer.timeout.connect(self.update_plot)
        self.updateTimer.start(self.updateDt)

        self.daqUpdateThread = threading.Thread(target=self.updateDAQDataAsync, daemon=True)
        self.daqUpdateThread.start()
    
    def close(self):
        logging.info("closing graph")
        super(EPhysGraph, self).hide()

    def updateDAQDataAsync(self):
        while True:
            if self.daq.isRunningProtocol:
                continue
            self.latestReadData, self.latestRespData, totalResistance, accessResistance, membraneResistance, membraneCapacitance = self.daq.getDataFromSquareWave(wave_freq = 50, samplesPerSec = 50000, dutyCycle = 0.5, amplitude = 0.5, recordingTime= 0.05)
            if totalResistance is not None:
                self.resistanceDeque.append(totalResistance)

    def update_plot(self):
        if  self.latestReadData is not None and self.latestRespData is not None:
            self.cmdPlot.clear()
            self.cmdPlot.plot(self.latestReadData[0, :], self.latestReadData[1, :])
            self.respPlot.clear()
            self.respPlot.plot(self.latestRespData[0, :], self.latestRespData[1, :])
            self.latestReadData = None
            self.latestRespData = None

        if len(self.resistanceDeque) > 0:
            self.resistancePlot.clear()
            displayDequeY = list(self.resistanceDeque)
            displayDequeX = list(range(len(displayDequeY)))
            self.resistancePlot.plot(displayDequeX, displayDequeY, pen="k")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    readDev = "cDAQ1Mod1"
    readChannel = "ai0"
    cmdDev = "cDAQ1Mod4"
    cmdChannel = "ao0"
    respDev = "cDAQ1Mod1"
    respChannel = "ai3"

    daq = DAQtest(readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel)
    graph = EPhysGraph(daq)
    graph.show()

    sys.exit(app.exec_())

