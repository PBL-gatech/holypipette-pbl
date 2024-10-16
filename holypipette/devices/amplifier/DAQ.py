import nidaqmx
import nidaqmx.system
import nidaqmx.constants

import numpy as np
import scipy.signal as signal
import math
# import sys, os
import time
import threading
import logging

__all__ = ['DAQ', 'FakeDAQ']

class DAQ:
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12 #400 pA (supplied cell) / V (DAQ out)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3) #10 mV (DAQ input) / V (cell out)

    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3) #20 mV (supplied cell) / V (DAQ out)
    V_CLAMP_VOLT_PER_AMP = 0.5 / 1e-9 #0.5V DAQ out (DAQ input) / nA (cell out)

    def __init__(self, readDev, readChannel, cmdDev, cmdChannel):
        self.readDev = readDev
        self.cmdDev = cmdDev

        self.readChannel = readChannel
        self.cmdChannel = cmdChannel
        self.latestResistance = None
        self.isRunningProtocol = False
        self.isRunningVoltageProtocol = False
        self._deviceLock = threading.Lock()

        self.latest_protocol_data = None
        self.current_protocol_data = None
        self.voltage_protocol_data = None
        self.holding_protocol_data = None

        #read constants

        logging.info(f'Using {self.readDev}/{self.readChannel} for reading and {self.cmdDev}/{self.cmdChannel} for writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        numSamples = int(samplesPerSec * recordingTime)
        # print("Num Samples", numSamples)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{self.readDev}/{self.readChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numSamples)
            # task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(f'{self.readDev}/{self.readChannel}', pretrigger_samples = 10, trigger_slope=nidaqmx.constants.Slope.RISING, trigger_level=0.2)
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            # print("Data len", data.shape)
            # print("Data", data)
            task.stop()

        #check for None values
        if data is None or np.where(data == None)[0].size > 0:
            data = np.zeros(self.numSamples)
            
        return data
        
    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(f'{self.cmdDev}/{self.cmdChannel}')
        task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        
        # create a wave_freq Hz square wave
        data = np.zeros(int(samplesPerSec * recordingTime))
        # print("data len", data.shape)
        
        period = int(1 / wave_freq * samplesPerSec)
        onTime = int(period * dutyCycle)

        wavesPerSec = samplesPerSec // period
        # print("wavesPerSec", wavesPerSec)


        # for i in range(wavesPerSec):
        #     data[i * period_old : i * period_old + onTime_old] = 0
        #     data[i * period_old + onTime_old : (i+1) * period_old] = amplitude

        # * This should achieve the same result as the above for loop
        data[:wavesPerSec*period:onTime] = 0
        data[onTime:wavesPerSec*period] = amplitude

        task.write(data)
        
        return task

    
    def getDataFromCurrentProtocol(self, startCurrentPicoAmp=-200, endCurrentPicoAmp=300, stepCurrentPicoAmp=100, highTimeMs=400):
        '''Sends a series of square waves from startCurrentPicoAmp to endCurrentPicoAmp (inclusive) with stepCurrentPicoAmp pA increments.
           Square wave period is 2 * highTimeMs ms. Returns a 2d array of data with each row being a square wave.
        '''

        self.isRunningProtocol = True
        self.latest_protocol_data = None # clear data
        num_waves = int((endCurrentPicoAmp - startCurrentPicoAmp) / stepCurrentPicoAmp) + 1

        # convert to amps
        startCurrent = startCurrentPicoAmp * 1e-12

        # get wave frequency Hz
        wave_freq = 1 / (2 * highTimeMs * 1e-3)

        #general constants for square waves
        samplesPerSec = 50000
        recordingTime = 3 * highTimeMs * 1e-3

        for i in range(num_waves):
            currentAmps = startCurrent + i * stepCurrentPicoAmp * 1e-12
            logging.info(f'Sending {currentAmps * 1e12} pA square wave.')

            #convert to DAQ output
            amplitude = currentAmps / self.C_CLAMP_AMP_PER_VOLT
            
            #send square wave to DAQ
            self._deviceLock.acquire()
            sendTask = self._sendSquareWave(wave_freq, samplesPerSec, 0.5, amplitude, recordingTime)
            sendTask.start()
            data = self._readAnalogInput(samplesPerSec, recordingTime)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()

            #convert to V (cell out)
            data = data / self.C_CLAMP_VOLT_PER_VOLT
            
            lowZero = currentAmps > 0
            data = self._shiftWaveToZero(data, lowZero)
            triggeredSamples = data.shape[0]
            xdata = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
            time.sleep(0.5)

            if self.current_protocol_data is None:
                self.current_protocol_data = [[xdata, data]]
            else:
                self.current_protocol_data.append([xdata, data])
        
        self.isRunningProtocol = False

        return self.current_protocol_data
    
    def getDataFromHoldingProtocol(self):
        ''' measures data from Post synaptic currents to determine spontaneous activity from other connected neurons'''
        self.isRunningProtocol = True
        self.holding_protocol_data = None # clear data
        self._deviceLock.acquire()
        data  = self._readAnalogInput(50000, 1)
        self._deviceLock.release()
        triggeredSamples = data.shape[0]
        xdata = np.linspace(0, triggeredSamples / 50000, triggeredSamples, dtype=float)
        self.isRunningProtocol = False

        self.holding_protocol_data = np.array([xdata, data])
        # print("Holding Protocol Data", self.holding_protocol_data)

        return self.holding_protocol_data



    def getDataFromVoltageProtocol(self):
        '''Sends a square wave to determine membrane properties, returns time constant, resistance, and capacitance.'''
        self.isRunningProtocol = True
        # self.latest_protocol_data = None # clear data
        self.voltage_protocol_data = None # clear data

        self.voltage_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
        # self.lastest_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)

        self.isRunningProtocol = False

        # return self.latest_protocol_data
        return self.voltage_protocol_data
    
    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        # measure the time it took to acquire the data
        # start0 = time.time()
        self._deviceLock.acquire()
        sendTask = self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
        sendTask.start()
        # start2 = time.time()
        data = self._readAnalogInput(samplesPerSec, recordingTime)
        # print("Time to read", time.time() - start2)
        sendTask.stop()
        sendTask.close()
        self._deviceLock.release()
        
        # print("Time to unlock", time.time() - start0)

        # print("Before Shape", data.shape)
        # ? Why 0.02?
        data, self.latestResistance = self._getResistancefromCurrent(data, amplitude * 0.02, samplesPerSec)
        triggeredSamples = data.shape[0]
        xdata = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
        # print(xdata)


        # Gradient
        gradientData = np.gradient(data, xdata)        
        max_index = np.argmax(gradientData)
        # Find the index of the minimum value after the maximum value
        min_index = np.argmin(gradientData[max_index:]) + max_index
        
        # Truncate the array
        left_bound = 10
        right_bound = 100
        # * bound is arbitrary, just to make it look good on the graph
        data = data[max_index-left_bound:min_index + right_bound]
        xdata = xdata[max_index-left_bound:min_index + right_bound]
        # gradientData = gradientData[max_index-left_bound:min_index + right_bound]

        # convert from V to pA
        data *= 2000
        # convert from pA to Amps
        data *= 1e-12

        # logging.info(f"Time to acquire & transform data: {time.time() - start0}")
        return np.array([xdata, data]), self.latestResistance
    
    def resistance(self):
        # logging.warn("latestResistance", self.latestResistance)
        return self.latestResistance

    def _filter60Hz(self, data):
        samplesPerSec = 50000
        #60 hz filter
        b, a = signal.iirnotch(48.828125, Q=30, fs=samplesPerSec)
        data = signal.filtfilt(b, a, data, irlen=1000)
        return data

    def _shiftWaveToZero(self, data, lowZero=True):
        median = np.median(data)

        if lowZero:
            zeroAvg = np.mean(data[data < median])
        else:
            zeroAvg = np.mean(data[data > median])

        #set data mean to 0
        shiftedData = data - zeroAvg

        return shiftedData

    def _getResistancefromCurrent(self, data, cmdVoltage, samplesPerSec, calcResistance=True):
        # print("Data", data)
        try:
            # shiftedData = self._shiftWaveToZero(data)
            shiftedData = data
            # print("Shifted Data 1", shiftedData.shape)
            mean = np.mean(shiftedData)
            lowAvg = np.mean(shiftedData[shiftedData < mean])
            highAvg = np.mean(shiftedData[shiftedData > mean])

            # #split data into high and low wave
            # triggerVal = np.mean(shiftedData)

            # #is trigger value ever reached?
            # # print("Trigger Value", triggerVal)
            # if np.where(shiftedData < triggerVal)[0].size == 0 or np.where(shiftedData > triggerVal)[0].size == 0:
            #     logging.warn("Trigger is reached!")
            #     if calcResistance:
            #         return shiftedData, None
            #     return shiftedData

            # # find the first index where the data is less than triggerVal
            # fallingEdge = np.where(shiftedData < triggerVal)[0][0]
            # shiftedData = shiftedData[fallingEdge:]
            # # print("Shifted Data 2", shiftedData.shape)

            # #find the first index where the data is greater than triggerVal
            # risingEdge = np.where(shiftedData > triggerVal)[0][0]
            # shiftedData = shiftedData[risingEdge:]
            # # print("Shifted Data 3", shiftedData.shape)

            # # find second rising edge location
            # secondFallingEdge = np.where(shiftedData < triggerVal)[0][0]
            
            # # find peak to peak spread on high side
            # # highSide = shiftedData[5:secondFallingEdge-5:]
            # # if highSide.size > 0:
            # #     highPeak = np.max(highSide)
            # #     lowPeak = np.min(highSide)
            # #     peakToPeak = highPeak - lowPeak
            # #     logging.info(f'Peak to peak: {peakToPeak * 1e12} ({highPeak * 1e12} - {lowPeak * 1e12})')

            # #find second rising edge after falling edge
        
            # ! resistance plot error comes from here
            # secondRisingEdge = np.where(shiftedData[secondFallingEdge:] > triggerVal)[0][0] + secondFallingEdge
            # print(secondFallingEdge, secondRisingEdge)
            # shiftedData = shiftedData[:secondRisingEdge]
            # print("Shifted Data 4", shiftedData.shape)
            
            # timestamps = np.linspace(0, data.shape[0] / samplesPerSec, data.shape[0], dtype=float)
            # data = np.gradient(data, timestamps)        


            if calcResistance:
                #convert high and low averages to pA
                highAvgPA = highAvg * 2000 * 1e-12
                lowAvgPA = lowAvg * 2000 * 1e-12
                #calculate resistance
                resistance = cmdVoltage / (highAvgPA - lowAvgPA)

                return data, resistance
                # return shiftedData, resistance
            return data
            # return shiftedData
            
        except Exception as e:
            # * we got an invalid square wave
            logging.error(f"Error in getResistancefromCurrent: {e}")
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # logging.error(f"{exc_type}, {fname}, {exc_tb.tb_lineno}")
            if calcResistance:
                return data, None
            return data
    
class FakeDAQ:
    def __init__(self):
        self.latestResistance = 6 * 10 ** 6
        self.latest_protocol_data = None
        self.isRunningProtocol = False
        self.current_protocol_data = None
        self.voltage_protocol_data = None

    def resistance(self):
        return self.latestResistance + np.random.normal(0, 0.1 * 10 ** 6)
    
    def getDataFromVoltageProtocol(self):
        '''Sends a square wave to determine membrane properties, returns time constant, resistance, and capacitance.'''
        self.isRunningProtocol = True
        # self.latest_protocol_data = None # clear data
        self.voltage_protocol_data = None # clear data

        self.voltage_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
        # self.lastest_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)

        self.isRunningProtocol = False

        # return self.latest_protocol_data
        return self.voltage_protocol_data

    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        #create a wave_freq Hz square wave
        data = np.zeros(int(samplesPerSec / recordingTime))
        onTime = 1 / wave_freq * dutyCycle * samplesPerSec
        offTime = 1 / wave_freq * (1-dutyCycle) * samplesPerSec

        #calc period
        period = onTime + offTime

        #convert to int
        onTime = int(onTime)
        offTime = int(offTime)
        period = int(period)

        wavesPerSec = samplesPerSec // period

        for i in range(wavesPerSec):
            data[i * period : i * period + onTime] = amplitude


        xdata = np.linspace(0, recordingTime, len(data), dtype=float)

        data = np.array([xdata, data]), self.resistance()
        return data