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

        # ! False is for BATH mode, True is for CELL mode
        self.cellMode = False 

        #read constants

        logging.info(f'Using {self.readDev}/{self.readChannel} for reading and {self.cmdDev}/{self.cmdChannel} for writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        # ?? HOW DO WE KNOW THE UNITS?
        numSamples = int(samplesPerSec * recordingTime)
        # print("Num Samples", numSamples)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{self.readDev}/{self.readChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numSamples)
            # task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(f'{self.readDev}/{self.readChannel}', pretrigger_samples = 10, trigger_slope=nidaqmx.constants.Slope.RISING, trigger_level=0.2)
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            # print("Data len", data.shape)
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
        
        period = int(1 / wave_freq * samplesPerSec)
        onTime = int(period * dutyCycle)

        wavesPerSec = samplesPerSec // period

        data[:wavesPerSec*period:onTime] = 0
        data[onTime:wavesPerSec*period] = amplitude

        task.write(data)
        
        return task

    def setCellMode(self, mode: bool) -> None:
        self.cellMode = mode

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
        '''measures data from Post synaptic currents to determine spontaneous activity from other connected neurons'''
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

        self.voltage_protocol_data, _, _, _, _ = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
        # self.lastest_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)

        self.isRunningProtocol = False

        # return self.latest_protocol_data
        return self.voltage_protocol_data
    
    def getDataFromSquareWave(self, wave_freq, samplesPerSec: int, dutyCycle, amplitude, recordingTime) -> tuple:
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
        
        triggeredSamples = data.shape[0]
        xdata = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype = float)
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
        data = data[max_index - left_bound:min_index + right_bound]
        xdata = xdata[max_index - left_bound:min_index + right_bound]

        # print("Data[0] ", data[0])
        # print("Max data", np.max(data))
        # print("Min data", np.min(data))
        
        if self.cellMode:
            print("Cell Mode")
            # self.totalResistance = None
            # ? Why 0.02?
        else:
            print("Bath Mode")
            # self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = None, None, None
            # ? Why 0.02?

        self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = self._getParamsfromCurrent(data, xdata, amplitude)
        self.totalResistance = self._getResistancefromCurrent(data, amplitude)

        # print("Difference: ", self.totalResistance - (self.latestAccessResistance + self.latestMembraneResistance))
        # logging.info(f"Time to acquire & transform data: {time.time() - start0}")
        
        # convert from pA to Amps (for the graph to be fooled, we believe the data is already in pA)
        return np.array([xdata, data * 1e-12]), self.totalResistance, self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance
    
    def resistance(self):
        # logging.warn("totalResistance", self.totalResistance)
        return self.totalResistance

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

    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])  

            print("Mean", mean)
            print("Low Avg", lowAvg)
            print("High Avg", highAvg) 

            print("cmdVoltage: ", cmdVoltage)
            # calculate resistance
            resistance = cmdVoltage / (highAvg - lowAvg)

            return resistance
            
        except Exception as e:
            # * we got an invalid square wave, or division by zero
            logging.error(f"Error in getResistancefromCurrent: {e}")
            return None
        
    def _getParamsfromCurrent(self, data, xdata, cmdVoltage) -> tuple:
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None

        try:
            # calculate the capicitance and resistance of the membrane and the access resistance
            print("Calculating parameters")
            # convert into pandas data frame
            df = pd.DataFrame({'X': xdata, 'Y': data})
            df['X_ms'] = df['X'] * 1000  # converting seconds to milliseconds
            df['Y_pA'] = df['Y']
            # Decay filter part
            filtered_data, pre_filtered_data, post_filtered_data, plot_params, I_prev, I_post = self.filter_data(df)
            peak_time, peak_index, min_time, min_index = plot_params
            m, t, b = self.optimizer(filtered_data)
            print("Optimized")
            print(m, t, b)
            if m is not None and t is not None and b is not None:
                tau = 1 / t
                    # Get peak current using peak_index
                I_peak = df.loc[peak_index, 'Y_pA']
                # Calculate parameters
                R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, cmdVoltage, I_peak, I_prev, I_post)
            print("Parameters calculated")

        except Exception as e:
            logging.error(f"Error in getParamsfromCurrent: {e}")

        print(R_a_MOhms, R_m_MOhms, C_m_pF)
        return R_a_MOhms, R_m_MOhms, C_m_pF

    def filter_data(self, data):
        # Decay filter part
        peak_index = data['Y_pA'].idxmax()
        peak_time = data.loc[peak_index, 'X_ms']

        min_index = data['Y_pA'].idxmin()
        min_time = data.loc[min_index, 'X_ms']

        # Extract the data between peaks
        sub_data = data[(data['X_ms'] >= peak_time) & (data['X_ms'] <= min_time)].copy()
        
        # Calculate the first numerical derivative
        sub_data['Y_derivative'] = sub_data['Y_pA'].diff() / sub_data['X_ms'].diff()
        
        # Remove the section with sudden changes
        change_threshold = sub_data['Y_derivative'].quantile(0.01)
        drop_indices = sub_data[sub_data['Y_derivative'] < change_threshold].index
        if not drop_indices.empty:
            drop_index = drop_indices[0]
            filtered_sub_data = sub_data.loc[:drop_index - 1]
        else:
            filtered_sub_data = sub_data

        # Pre-peak filter part
        peak_value = data.loc[peak_index, 'Y_pA']
        pre_peak_data = data[data['X_ms'] < peak_time].copy()
        std_pre_peak = pre_peak_data['Y_pA'].std()
        pre_peak_data = pre_peak_data[pre_peak_data['Y_pA'] < peak_value - 3 * std_pre_peak]
        mean_filtered_pre_peak = pre_peak_data['Y_pA'].mean()

        # Post-peak filter part
        post_peak_data = filtered_sub_data.copy()
        std_post_peak = post_peak_data['Y_pA'].std()
        post_peak_data = post_peak_data[post_peak_data['Y_pA'] < peak_value - 3 * std_post_peak]
        mean_filtered_post_peak = post_peak_data['Y_pA'].mean()

        return filtered_sub_data, pre_peak_data, post_peak_data, [peak_time, peak_index, min_time, min_index], mean_filtered_pre_peak, mean_filtered_post_peak

    
    def monoExp(self,x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self,filtered_data):
        start = filtered_data['X_ms'].iloc[0]
        # Shift the data to start at 0
        filtered_data['X_ms'] = filtered_data['X_ms'] - start
        p0 = (664, 0.24, 15)
        try:
            params, _ = scipy.optimize.curve_fit(self.monoExp, filtered_data['X_ms'], filtered_data['Y_pA'], maxfev=10000, p0=p0)
            m, t, b = params
            print("Params", m, t, b)
            return m, t, b
        except Exception as e:
            print("Error:", e)
            return None, None, None

        
    def calc_param(self, tau, dV, I_peak, I_prev, I_ss, epsilon=1e-12):
        tau_s = tau / 1000  # Convert ms to seconds
        dV_V = dV * 1e-3  # Convert mV to V
        I_d = I_peak - I_prev  # in pA
        I_dss = I_ss - I_prev  # in pA
        I_d_A = I_d * 1e-12
        I_dss_A = I_dss * 1e-12

        # Check for potential division by zero or very small values
        if abs(I_d_A) < epsilon or abs(I_dss_A) < epsilon:
            print("Warning: Division by zero or near zero encountered in R_a or R_m calculation")
            return float('nan'), float('nan'), float('nan')

        # Calculate Access Resistance (R_a) in ohms
        R_a_Ohms = dV_V / I_d_A  # Ohms
        R_a_MOhms = R_a_Ohms * 1e-6  # Convert to MOhms

        # Calculate Membrane Resistance (R_m) in ohms
        R_m_Ohms = (dV_V - (R_a_Ohms * I_dss_A)) / I_dss_A  # Ohms
        R_m_MOhms = R_m_Ohms * 1e-6  # Convert to MOhms

        # Check for potential invalid operations
        if abs(R_a_Ohms) < epsilon or abs(R_m_Ohms) < epsilon:
            print("Warning: Invalid operation encountered in C_m calculation")
            return float('nan'), float('nan'), float('nan')

        # Calculate Membrane Capacitance (C_m) in farads
        C_m_F = tau_s / (1/(1 / R_a_Ohms) + (1 / R_m_Ohms))  # Farads
        C_m_pF = C_m_F * 1e12  # Convert to pF

        print("R_a_MOhms, R_m_MOhms, C_m_pF", R_a_MOhms, R_m_MOhms, C_m_pF)

        return R_a_MOhms, R_m_MOhms, C_m_pF


class FakeDAQ:
    def __init__(self):
        self.totalResistance = 6 * 10 ** 6
        self.latest_protocol_data = None
        self.isRunningProtocol = False
        self.current_protocol_data = None
        self.voltage_protocol_data = None

    def resistance(self):
        return self.totalResistance + np.random.normal(0, 0.1 * 10 ** 6)
    
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