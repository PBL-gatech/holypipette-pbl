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
    def __init__(self, readDev, readChannel, cmdDev, cmdChannel):
        self.readDev = readDev
        self.cmdDev = cmdDev
        self.readChannel = readChannel
        self.cmdChannel = cmdChannel
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

        logging.info(f'Using {self.readDev}/{self.readChannel} for reading and {self.cmdDev}/{self.cmdChannel} for writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        numSamples = int(samplesPerSec * recordingTime)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{self.readDev}/{self.readChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numSamples)
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            task.stop()

        if data is None or np.where(data == None)[0].size > 0:
            data = np.zeros(numSamples)
            
        return data
    
    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(f'{self.cmdDev}/{self.cmdChannel}')
        task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        
        numSamples = int(samplesPerSec * recordingTime)
        data = np.zeros(numSamples)
        # print the size of hte array
        # print(f"Size of data: {data.size}")
        
        period = int(samplesPerSec / wave_freq)
        # print the lenght of the period
        # print(f"Period: {period}")
        onTime = int(period * dutyCycle)
        # print the length of the onTime
        # print(f"On Time: {onTime}")

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
        
        triggeredSamples = data.shape[0]
        xdata = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype = float)

        # gradientData = np.gradient(data, xdata)
        # max_index = np.argmax(gradientData)
        # min_index = np.argmin(gradientData[max_index:]) + max_index
        
        left_bound = 100
        right_bound = 150
        # data = data[max_index - left_bound:min_index + right_bound]
        # xdata = xdata[max_index - left_bound:min_index + right_bound]
        # cmdData = cmdData[max_index - left_bound:min_index + right_bound]

        data *= 1e-9
        amplitude *= 1e-2

        self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = self._getParamsfromCurrent(data, xdata, amplitude)
        self.totalResistance = self._getResistancefromCurrent(data, amplitude)

        return np.array([xdata, cmdData]), np.array([xdata, data]), self.totalResistance, self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance
    
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

        self.commandPlot = PlotWidget()
        self.responsePlot = PlotWidget()
        self.resistancePlot = PlotWidget()

        for plot in [self.commandPlot, self.responsePlot, self.resistancePlot]:
            plot.setBackground("w")
            plot.getAxis("left").setPen("k")
            plot.getAxis("bottom").setPen("k")

        self.commandPlot.setLabel("left", "Voltage", units="V")
        self.commandPlot.setLabel("bottom", "Time", units="s")
        self.responsePlot.setLabel("left", "Current", units="A")
        self.responsePlot.setLabel("bottom", "Time", units="s")
        self.resistancePlot.setLabel("left", "Resistance", units="Ohms")
        self.resistancePlot.setLabel("bottom", "Samples", units="")

        self.latestCmdData = None
        self.latestRspData = None
        self.resistanceDeque = deque(maxlen=100)  # Initialize the resistance deque

        layout = QVBoxLayout()
        layout.addWidget(self.commandPlot)
        layout.addWidget(self.responsePlot)
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
            self.latestCmdData, self.latestRspData, totalResistance, accessResistance, membraneResistance, membraneCapacitance = self.daq.getDataFromSquareWave(50, 50000, 0.5, 5, 0.06)
            if totalResistance is not None:
                self.resistanceDeque.append(totalResistance)

    def update_plot(self):
        if self.latestCmdData is not None and self.latestRspData is not None:
            self.commandPlot.clear()
            self.commandPlot.plot(self.latestCmdData[0, :], self.latestCmdData[1, :])
            self.responsePlot.clear()
            self.responsePlot.plot(self.latestRspData[0, :], self.latestRspData[1, :])
            self.latestCmdData = None
            self.latestRspData = None

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

    daq = DAQtest(readDev, readChannel, cmdDev, cmdChannel)
    graph = EPhysGraph(daq)
    graph.show()

    sys.exit(app.exec_())
