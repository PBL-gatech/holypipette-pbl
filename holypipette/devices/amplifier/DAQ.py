import nidaqmx
import nidaqmx.system
import nidaqmx.constants

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal as signal
import serial
import serial.tools.list_ports
import time
import threading
import logging



__all__ = ['DAQ','ArduinoDAQ','FakeDAQ']

class DAQ:
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12 #400 pA (supplied cell) / V (DAQ out)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3) #10 mV (DAQ input) / V (cell out)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3) #20 mV (supplied cell) / V (DAQ out)
    V_CLAMP_VOLT_PER_AMP = (2*1e-9) #0.5V DAQ out (DAQ input) / pA (cell out)

    def __init__(self, readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel):

        self.pulses = None
        self.pulseRange = None
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
        self.voltage_command_data = None
        self.voltageTotalResistance  =None
        self.voltageMembraneResistance = None
        self.voltageAccessResistance = None
        self.voltageMembraneCapacitance = None
        self.equalizer = False

        self.holding_current = None
        self.holding_voltage = None

        # ! False is for BATH mode, True is for CELL mode
        self.cellMode = False

        logging.info(f'Using {self.readDev}/{self.readChannel} for reading the output of {self.cmdDev}/{self.cmdChannel} and {self.respDev}/{self.respChannel} for response.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        # ?? HOW DO WE KNOW THE UNITS?
        numSamples = int(samplesPerSec * recordingTime)
        # print("Num Samples", numSamples)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{self.readDev}/{self.readChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.ai_channels.add_ai_voltage_chan(f'{self.respDev}/{self.respChannel}', max_val=10, min_val=0, terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=numSamples)
            # task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(f'{self.readDev}/{self.readChannel}', pretrigger_samples = 10, trigger_slope=nidaqmx.constants.Slope.RISING, trigger_level=0.2)
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            # print("Data len", data.shape)
            task.stop()

        #check for None values
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
        
        return task
    
    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
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

    def getDataFromCurrentProtocol(self, custom = False,factor = None,startCurrentPicoAmp=None, endCurrentPicoAmp=None, stepCurrentPicoAmp=10, highTimeMs=400):
        '''Sends a series of square waves from startCurrentPicoAmp to endCurrentPicoAmp (inclusive) with stepCurrentPicoAmp pA increments.
           Square wave period is 2 * highTimeMs ms. Returns a 2d array of data with each row being a square wave.

        '''
        # print(f"volt membrane capacitance: {self.voltageMembraneCapacitance}")
        if self.voltageMembraneCapacitance is None or self.voltageMembraneCapacitance is 0:
            self.voltageMembraneCapacitance = 0 
            # logging.warn("Is system set to cell mode?")
            # logging.error("Voltage membrane capacitance is not set. Please run voltage protocol first.")
            # logging.error("Returning None,Current clamp protocol cannot be run.")
            return None, None, None
        
        if not custom:
            factor = 2
            startCurrentPicoAmp = round(-self.voltageMembraneCapacitance * factor, -1)
            endCurrentPicoAmp = round(self.voltageMembraneCapacitance * factor, -1)
        else:
            if startCurrentPicoAmp is None or endCurrentPicoAmp is None:
                raise ValueError("startCurrentPicoAmp and endCurrentPicoAmp must be provided when custom is True.")
        # create a spaced list and count number of pulses from startCurrentPicoAmp to endCurrentPicoAmp based off of stepCurrentPicoAmp
        logging.info(f'Starting Current Protocol with start: {startCurrentPicoAmp}, end: {endCurrentPicoAmp}, step: {stepCurrentPicoAmp}.')
        self.pulses = np.arange(startCurrentPicoAmp, endCurrentPicoAmp + stepCurrentPicoAmp, stepCurrentPicoAmp)
        if 0 not in self.pulses:
            self.pulses = np.insert(self.pulses, len(self.pulses) // 2, 0)
        logging.info(f'Pulses: {self.pulses}')
        self.pulseRange = len(self.pulses)
        self.isRunningProtocol = True
        self.latest_protocol_data = None # clear data
        num_waves = int((endCurrentPicoAmp - startCurrentPicoAmp) / stepCurrentPicoAmp) + 2

        # convert to amps
        startCurrent = startCurrentPicoAmp * 1e-12
        print("Start Current", startCurrent)

        # get wave frequency Hz
        wave_freq = 1 / (2 * highTimeMs * 1e-3)

        #general constants for square waves
        samplesPerSec = 20000
        recordingTime = 4 * highTimeMs * 1e-3

        for i in range(num_waves-1):

            # param pulse at -20 pA
            amp_pulse = (-20*1e-12)/self.C_CLAMP_AMP_PER_VOLT
            wave_pulse = 1 / (2 * highTimeMs/2 * 1e-3)
            recording_pulse = 3 * highTimeMs/2 * 1e-3
            logging.info(f'Sending param pulse at -20 pA square wave.')
            self._deviceLock.acquire()
            sendTask = self._sendSquareWaveCurrent(wave_pulse, samplesPerSec, 0.5, amp_pulse, recording_pulse)
            sendTask.start()
            #read analog input
            data = self._readAnalogInput(samplesPerSec, recording_pulse)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()
            respData0 = data[1] # Current response from the cell
            readData0 = data[0] # voltage sent to the cell
            respData0 = respData0 / self.C_CLAMP_VOLT_PER_VOLT
            time.sleep(0.5)
            # logging.info(f'obtain parameters from  cell with -20 pA square wave.')
            
            currentAmps = self.pulses[i]*1e-12
            logging.info(f'Sending {currentAmps * 1e12} pA square wave.')
            #convert to DAQ output
            amplitude = currentAmps / self.C_CLAMP_AMP_PER_VOLT
            # print("Amplitude", amplitude)

            #send square wave to DAQ
            self._deviceLock.acquire()
            sendTask = self._sendSquareWaveCurrent(wave_freq, samplesPerSec, 0.5, amplitude, recordingTime)
            sendTask.start()
            #read analog input
            data = self._readAnalogInput(samplesPerSec, recordingTime)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()
            respData1 = data[1] # Current response from the cell
            readData1 = data[0] # voltage sent to the cell

            #convert to V (cell out)
            respData1 = respData1 / self.C_CLAMP_VOLT_PER_VOLT
            respData = np.concatenate((respData0,respData1))
            readData = np.concatenate((readData0, readData1))
            triggeredSamples = respData.shape[0]
            timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
            time.sleep(0.5)
            #combine data

        
            if self.current_protocol_data is None:
                self.current_protocol_data = [[timeData, respData, readData]]
            else:
                self.current_protocol_data.append([timeData, respData, readData])

        self.isRunningProtocol = False

        # print("Current Protocol Data", self.current_protocol_data)

        return self.current_protocol_data, self.pulses, self.pulseRange
    
    def getDataFromHoldingProtocol(self):
        '''measures data from Post synaptic currents to determine spontaneous activity from other connected neurons'''
        self.isRunningProtocol = True
        self.holding_protocol_data = None # clear data
        self._deviceLock.acquire()
        data  = self._readAnalogInput(50000, 1)
        respData = data[1]
        readData = data[0]
        self._deviceLock.release()
        triggeredSamples = respData.shape[0]
        timeData = np.linspace(0, triggeredSamples / 50000, triggeredSamples, dtype=float)
        self.isRunningProtocol = False
        #show shapes of data
        # assign data to holding_protocol_data
        self.holding_protocol_data = np.array([timeData, respData, readData])
        # print("Holding Protocol Data", self.holding_protocol_data)

        return self.holding_protocol_data

    def getDataFromVoltageProtocol(self):
        '''Sends a square wave to determine membrane properties, returns time constant, resistance, and capacitance.'''
        self.voltage_protocol_data, self.voltage_command_data = None, None  # clear data
        max_attempts = 5  # maximum number of attempts to avoid infinite loop
        attempts = 0
        try:
            self.isRunningProtocol = True
            while attempts < max_attempts:
                attempts += 1
                self.voltage_protocol_data, self.voltage_command_data, self.voltageTotalResistance, self.voltageMembraneResistance, self.voltageAccessResistance, self.voltageMembraneCapacitance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.05)
        
                if self.voltageMembraneCapacitance != 0:
                    break  # exit loop if capacitance is non-zero
                else:
                    logging.warning(f"Attempt {attempts}: Capacitance is zero, retrying...")
            # print(f"Latest membrane capacitance: {self.voltageMembraneCapacitance}")
        except Exception as e:
            self.voltage_protocol_data, self.voltage_command_data = None, None
            # logging.error(f"Error in getDataFromVoltageProtocol: {e}")
        finally:
            self.isRunningProtocol = False
        return self.voltageMembraneCapacitance

    def getDataFromSquareWave(self, wave_freq, samplesPerSec: int, dutyCycle, amplitude, recordingTime) -> tuple:
        self.equalizer = False
        
        while not self.equalizer:
            self._deviceLock.acquire()
            sendTask = self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
            sendTask.start()
            data = self._readAnalogInput(samplesPerSec, recordingTime)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()
            numSamples = int(samplesPerSec * recordingTime)

            if data is not None and data.shape[1] == numSamples:
                # logging.info(f"Data shape: {data.shape[1]}")
                try:
                    triggeredSamples = data.shape[1]
                    timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)

                    # Gradient
                    gradientData = np.gradient(data[0], timeData)
                    max_index = np.argmax(gradientData)
                    min_index = np.argmin(gradientData[max_index:])
                    max_grad = np.max(gradientData)
                    min_grad = np.min(gradientData[max_index:])
                    equalizer = abs(max_grad) - abs(min_grad)
                    
                    if abs(max_grad) - abs(min_grad) < 1000:
                        self.equalizer = True
                        # logging.error(f"gradients equal: {equalizer}")
                    else:
                        self.equalizer = False
                        # logging.warning(f"gradients not equal: {equalizer}")
                        # logging.info(f"Max grad value: {gradientData[max_index]}, Max value: {data[0][max_index]}, Max index: {max_index}")
                        # logging.info(f"Min grad value: {gradientData[min_index]}, Max value: {data[0][min_index]}, Min index: {min_index}")

                except Exception as e:
                    # logging.error(f"Error in getDataFromSquareWave: {e}")
                    # logging.info(f"Data shape: {data.shape[1]}")
                    return None, None, None, None, None, None

        if self.equalizer:
            try:
                self.equalizer = False
                min_index = np.argmin(gradientData[max_index:]) + max_index

                left_bound = 100
                right_bound = 300
                respData = data[1]
                readData = data[0]

                respData = data[1][max_index - left_bound:min_index + right_bound]
                readData = data[0][max_index - left_bound:min_index + right_bound]
                timeData = timeData[max_index - left_bound:min_index + right_bound]
                timeData = timeData - timeData[0]

                readData = readData * self.V_CLAMP_VOLT_PER_VOLT
                respData = respData * self.V_CLAMP_VOLT_PER_AMP

                self.latestAccessResistance = 0
                self.latestMembraneResistance = 0
                self.latestMembraneCapacitance = 0
                if self.cellMode:
                    self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = self._getParamsfromCurrent(readData, respData, timeData, amplitude * self.V_CLAMP_VOLT_PER_VOLT)

                self.totalResistance = 0
                if self.cellMode and self.latestAccessResistance is not None and self.latestMembraneResistance is not None:
                    self.totalResistance = self.latestAccessResistance + self.latestMembraneResistance
                else:
                    self.totalResistance = self._getResistancefromCurrent(respData, amplitude * self.V_CLAMP_VOLT_PER_VOLT)
                    self.totalResistance *= 1e-6  # to have it in MOhms

                return np.array([timeData, respData]), np.array([timeData, readData]), self.totalResistance, self.latestMembraneResistance, self.latestAccessResistance, self.latestMembraneCapacitance
            except Exception as e:
                # logging.error(f"Error in getDataFromSquareWave: {e}")
                # logging.info(f"resistance: {self.totalResistance}, membrane resistance: {self.latestMembraneResistance}, membrane capacitance: {self.latestMembraneCapacitance}")
                
                return None, None, None, None, None, None

    def resistance(self):
        # logging.warn("totalResistance", self.totalResistance)
        return self.totalResistance

    def _filter60Hz(self, data):
        samplesPerSec = 50000
        #60 hz filter
        b, a = signal.iirnotch(48.828125, Q=30, fs=samplesPerSec)
        data = signal.filtfilt(b, a, data, irlen=1000)
        return data

    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])
            resistance = cmdVoltage / (highAvg - lowAvg)

            # print("Resistance in func", resistance)
            return resistance
            
        except Exception as e:
            # * we got an invalid square wave, or division by zero
            # logging.error(f"Error in getResistancefromCurrent: {e}")
            return None
        
    def _getParamsfromCurrent(self, readData, respData, timeData, amplitude) -> tuple:
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None
        #check if data is not empty
        if len(readData) != 0 and len(respData) != 0 and len(timeData) != 0:
            try:
                # Create a data frame to obtain the peak and minimum values
                df = pd.DataFrame({'T': timeData, 'X': readData, 'Y': respData})
                #print first line
                # logging.info(f"First line: {df.iloc[0]}")
                #shift time axis to zero
                start = df['T'].iloc[0]
                df['T'] = df['T'] - start
                # dimensionality conversion
                df['T_ms'] = df['T'] * 1000  # converting seconds to milliseconds
                df['X_mV'] = df['X'] * 1000  # converting volts to millivolts
                df['Y_pA'] = df['Y'] * 1e12  # converting amps to picoamps
            
                # Decay filter part
                filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = self.filter_data(df)
            
                self.holding_current = I_prev_pA

                # logging.info("Data filtered")
                peak_time, peak_index, min_time, min_index = plot_params
                #get peak and min values
                I_peak_pA = df.loc[peak_index + 1, 'Y_pA']
                I_peak_time = df.loc[peak_index + 1, 'T_ms']
                # logging.info(f"Peak Current (pA): {I_peak_pA} at index: {peak_index}")
                # logging.info(f"steady state current (pA): {I_post_pA} at index: {min_index-10}")
                # logging.info(f"Peak time: {peak_time}, Peak index: {peak_index}, Min time: {min_time}, Min index: {min_index}")
                # logging.info(f"I_prev_pA: {I_prev_pA}, I_post_pA: {I_post_pA}, I_peak_pA: {I_peak_pA}")
                #group filtered_data, filtered_time, and filtered_command into a data frame
                fit_data = pd.DataFrame({'T_ms': filtered_time, 'X_mV': filtered_command, 'Y_pA': filtered_data})
                #print first line of fit_data
                # logging.info(f"First line of fit_data: {fit_data.iloc[0]}")
                # calculate mean voltage from filtered_command
                mean_voltage = filtered_command.mean()
                start = fit_data['T_ms'].iloc[0]
                fit_data['T_ms'] = fit_data['T_ms'] - start
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA)
                # print(f"m: {m}, t: {t}, b: {b}")
                if m is not None and t is not None and b is not None:
                    tau = 1 / t
                    # Calculate parameters
                    R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
                    # logging.info(f"R_a: {R_a_MOhms}, R_m: {R_m_MOhms}, C_m: {C_m_pF}")
                    
            
            except Exception as e:
                # logging.error(f"Error in getParamsfromCurrent: {e}")
                return None, None, None
                # return 0, 0, 0
        else:
            # logging.error("One or more of the data arrays is empty")
            return 0,0,0
        # logging.info("Returning parameters")
        return R_a_MOhms, R_m_MOhms, C_m_pF

    def filter_data(self, data):
        #calculate derivative of X_mV and add it to the data frame
        # convert to np array
        X_mV = data['X_mV'].to_numpy()
        T_ms = data['T_ms'].to_numpy()

        Y_pA = data['Y_pA'].to_numpy()

        X_dT = np.gradient(X_mV, T_ms)
        data["X_dT"] = X_dT
        # Find the index of the maximum value
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        # Find the index of the minimum value after the maximum value
        peak_current_index = np.argmax(Y_pA)
        peak_time = data.loc[peak_current_index, 'T_ms']
        # get peak and min times
        # positive_peak_time = data.loc[positive_peak_index, 'T_ms'].copy()
        negative_peak_time = data.loc[negative_peak_index, 'T_ms']
        # Extract the data between peaks

        pre_peak_current = data.loc[:positive_peak_index, "Y_pA"]
        sub_data = data.loc[peak_current_index:negative_peak_index, "Y_pA"]
        sub_time = data.loc[peak_current_index:negative_peak_index, "T_ms"]
        sub_command = data.loc[peak_current_index:negative_peak_index, "X_mV"]
        # calculate the mean current prior to voltage pulse (I_prev)
        mean_pre_peak = pre_peak_current.mean()
        # print("Mean pre peak", mean_pre_peak)
        # calculate the mean current post voltage pulse (I_ss)
        gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time.iloc[zero_gradient_index]
        # print(f"Zero gradient time: {zero_gradient_time}")
        # Calculate the mean of the data between the zero gradient time and the min time
        if zero_gradient_time:
            post_peak_current_data = data[(data['T_ms'] >= zero_gradient_time) & (data['T_ms'] <= data.loc[negative_peak_index, 'T_ms'])]
            mean_post_peak = post_peak_current_data['Y_pA'].mean()
        else:
            mean_post_peak = None
        # print("Mean post peak", mean_post_peak)
        return sub_data, sub_time, sub_command, [peak_time, peak_current_index, negative_peak_time, negative_peak_index], mean_pre_peak, mean_post_peak
    
    def monoExp(self,x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, fit_data, I_peak_pA, I_peak_time, I_ss):
        start = fit_data['T_ms'].iloc[0]
        # logging.info(f"Unshifted Start time: {start}")
        # Shift the data to start at 0
        fit_data['T_ms'] = fit_data['T_ms'] - start
        #print all of the fit data  to copy to a txt
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # print("T_ms: ", fit_data['T_ms'])
        # print("Y_pA: ", fit_data['Y_pA']) 

        # Save T_ms and Y_pA to a single CSV file and overwrite it each time
        # fit_data[['T_ms', 'Y_pA']].to_csv(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_patch_clamp_data\test_data.csv", index=False)
        # logging.info("Data saved to CSV")
        # print("I_SS: ", I_ss)
        p0 = (I_peak_pA, I_peak_time, I_ss)
        # print("P0", p0)
        # print if there is a nan value in fit_data
        try:
            # print("NAN VALUES: ", fit_data.isnull().values.any())
            params, _ = scipy.optimize.curve_fit(self.monoExp, fit_data['T_ms'], fit_data['Y_pA'], maxfev=1000000, p0=p0)
            m, t, b = params
            # print("Params: ", params)
            # print("Params", m, t, b)
            return m, t, b
        except Exception as e:
            # logging.error(f"Error in the optimizer: {e}")
            return None, None, None

    def calc_param(self, tau, mean_voltage, I_peak, I_prev, I_ss):
        I_d = I_peak - I_prev  # in pA
        I_dss = I_ss - I_prev  # in pA
        # logging.info(f"tau: {tau}, dmV: {mean_voltage}, I_d: {I_d}, I_dss: {I_dss}")
        # calculate access resistance:
        R_a_Mohms = ((mean_voltage*1e-3) / (I_d*1e-12))*1e-6 # 10 mV / 800 pA = 12.5 MOhms --> supposed to be 10 MOhms
        # print("R_a_MOhms in calc", R_a_Mohms)
        
        # calculate membrane resistance:
        R_m_Mohms = (((mean_voltage*1e-3) - R_a_Mohms*1e6*I_dss*1e-12)/(I_dss*1e-12))*1e-6 #530 Mohms --> supposed to be 500 MOhms
        C_m_pF = (tau*1e-3) / (1/(1/(R_a_Mohms*1e6) + 1/(R_m_Mohms*1e6))) * 1e12 # supposed to be 33 pF
        # print("C_m_pF in calc", C_m_pF)
        return R_a_Mohms, R_m_Mohms, C_m_pF



class ArduinoDAQ:
    # Calibration Constants
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA / V (DAQ out)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV / V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV / V (DAQ out)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)  # 2 V / A (DAQ input)

    def __init__(self, DAQSerial=None):
        """
        Initialize the ArduinoDAQ class with an already opened serial.Serial instance.

        Args:
            DAQSerial (serial.Serial): An already opened serial.Serial instance for Arduino communication.
        """
        if DAQSerial is not None and isinstance(DAQSerial, serial.Serial):
            self.DAQSerial = DAQSerial
            logging.info(f"ArduinoDAQ initialized with serial port: {self.DAQSerial.port} at {self.DAQSerial.baudrate} baud.")
        else:
            self.DAQSerial = None
            logging.error("DAQSerial must be an instance of serial.Serial and already opened.")
            raise ValueError("DAQSerial must be an instance of serial.Serial and already opened.")

        # Initialize DAQ variables
        self.pulses = None
        self.pulseRange = None
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
        self.voltage_command_data = None
        self.voltageTotalResistance = None
        self.voltageMembraneResistance = None
        self.voltageAccessResistance = None
        self.voltageMembraneCapacitance = None
        self.equalizer = False

        self.holding_current = None
        self.holding_voltage = None

        # False is for BATH mode, True is for CELL mode
        self.cellMode = False

        logging.info(f'Using {self.DAQSerial.port} for reading and writing.')
    def getDataFromCurrentProtocol(self, custom = False,factor = None,startCurrentPicoAmp=None, endCurrentPicoAmp=None, stepCurrentPicoAmp=10, highTimeMs=400):
        '''Sends a series of square waves from startCurrentPicoAmp to endCurrentPicoAmp (inclusive) with stepCurrentPicoAmp pA increments.
           Square wave period is 2 * highTimeMs ms. Returns a 2d array of data with each row being a square wave.

        '''
        # print(f"volt membrane capacitance: {self.voltageMembraneCapacitance}")
        if self.voltageMembraneCapacitance is None or self.voltageMembraneCapacitance is 0:
            self.voltageMembraneCapacitance = 0 
            # logging.warn("Is system set to cell mode?")
            # logging.error("Voltage membrane capacitance is not set. Please run voltage protocol first.")
            # logging.error("Returning None,Current clamp protocol cannot be run.")
            return None, None, None
        
        if not custom:
            factor = 2
            startCurrentPicoAmp = round(-self.voltageMembraneCapacitance * factor, -1)
            endCurrentPicoAmp = round(self.voltageMembraneCapacitance * factor, -1)
        else:
            if startCurrentPicoAmp is None or endCurrentPicoAmp is None:
                raise ValueError("startCurrentPicoAmp and endCurrentPicoAmp must be provided when custom is True.")
        # create a spaced list and count number of pulses from startCurrentPicoAmp to endCurrentPicoAmp based off of stepCurrentPicoAmp
        logging.info(f'Starting Current Protocol with start: {startCurrentPicoAmp}, end: {endCurrentPicoAmp}, step: {stepCurrentPicoAmp}.')
        self.pulses = np.arange(startCurrentPicoAmp, endCurrentPicoAmp + stepCurrentPicoAmp, stepCurrentPicoAmp)
        if 0 not in self.pulses:
            self.pulses = np.insert(self.pulses, len(self.pulses) // 2, 0)
        logging.info(f'Pulses: {self.pulses}')
        self.pulseRange = len(self.pulses)
        self.isRunningProtocol = True
        self.latest_protocol_data = None # clear data
        num_waves = int((endCurrentPicoAmp - startCurrentPicoAmp) / stepCurrentPicoAmp) + 2

        # convert to amps
        startCurrent = startCurrentPicoAmp * 1e-12
        print("Start Current", startCurrent)

        # get wave frequency Hz
        wave_freq = 1 / (2 * highTimeMs * 1e-3)

        #general constants for square waves
        samplesPerSec = 20000
        recordingTime = 4 * highTimeMs * 1e-3

        for i in range(num_waves-1):

            # param pulse at -20 pA
            amp_pulse = (-20*1e-12)/self.C_CLAMP_AMP_PER_VOLT
            wave_pulse = 1 / (2 * highTimeMs/2 * 1e-3)
            recording_pulse = 3 * highTimeMs/2 * 1e-3
            logging.info(f'Sending param pulse at -20 pA square wave.')
            self._deviceLock.acquire()
            sendTask = self._sendSquareWaveCurrent(wave_pulse, samplesPerSec, 0.5, amp_pulse, recording_pulse)
            sendTask.start()
            #read analog input
            data = self._readAnalogInput(samplesPerSec, recording_pulse)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()
            respData0 = data[1] # Current response from the cell
            readData0 = data[0] # voltage sent to the cell
            respData0 = respData0 / self.C_CLAMP_VOLT_PER_VOLT
            time.sleep(0.5)
            # logging.info(f'obtain parameters from  cell with -20 pA square wave.')
            
            currentAmps = self.pulses[i]*1e-12
            logging.info(f'Sending {currentAmps * 1e12} pA square wave.')
            #convert to DAQ output
            amplitude = currentAmps / self.C_CLAMP_AMP_PER_VOLT
            # print("Amplitude", amplitude)

            #send square wave to DAQ
            self._deviceLock.acquire()
            sendTask = self._sendSquareWaveCurrent(wave_freq, samplesPerSec, 0.5, amplitude, recordingTime)
            sendTask.start()
            #read analog input
            data = self._readAnalogInput(samplesPerSec, recordingTime)
            sendTask.stop()
            sendTask.close()
            self._deviceLock.release()
            respData1 = data[1] # Current response from the cell
            readData1 = data[0] # voltage sent to the cell

            #convert to V (cell out)
            respData1 = respData1 / self.C_CLAMP_VOLT_PER_VOLT
            respData = np.concatenate((respData0,respData1))
            readData = np.concatenate((readData0, readData1))
            triggeredSamples = respData.shape[0]
            timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
            time.sleep(0.5)
            #combine data

        
            if self.current_protocol_data is None:
                self.current_protocol_data = [[timeData, respData, readData]]
            else:
                self.current_protocol_data.append([timeData, respData, readData])

        self.isRunningProtocol = False

        # print("Current Protocol Data", self.current_protocol_data)

        return self.current_protocol_data, self.pulses, self.pulseRange
    
    def getDataFromHoldingProtocol(self):
        '''measures data from Post synaptic currents to determine spontaneous activity from other connected neurons'''
        self.isRunningProtocol = True
        self.holding_protocol_data = None # clear data
        self._deviceLock.acquire()
        data  = self._readAnalogInput(20000, 1)
        respData = data[1]
        readData = data[0]
        self._deviceLock.release()
        triggeredSamples = respData.shape[0]
        timeData = np.linspace(0, triggeredSamples / 20000, triggeredSamples, dtype=float)
        self.isRunningProtocol = False
        #show shapes of data
        # assign data to holding_protocol_data
        self.holding_protocol_data = np.array([timeData, respData, readData])
        # print("Holding Protocol Data", self.holding_protocol_data)

        return self.holding_protocol_data

    def getDataFromVoltageProtocol(self):
        '''Sends a square wave to determine membrane properties, returns time constant, resistance, and capacitance.'''
        self.voltage_protocol_data, self.voltage_command_data = None, None  # clear data
        max_attempts = 5  # maximum number of attempts to avoid infinite loop
        attempts = 0
        try:
            self.isRunningProtocol = True
            while attempts < max_attempts:
                attempts += 1
                self.voltage_protocol_data, self.voltage_command_data, self.voltageTotalResistance, self.voltageMembraneResistance, self.voltageAccessResistance, self.voltageMembraneCapacitance = self.getDataFromSquareWave(20, 20000, 0.5, 0.5, 0.05)
        
                if self.voltageMembraneCapacitance != 0:
                    break  # exit loop if capacitance is non-zero
                else:
                    logging.warning(f"Attempt {attempts}: Capacitance is zero, retrying...")
            # print(f"Latest membrane capacitance: {self.voltageMembraneCapacitance}")
        except Exception as e:
            self.voltage_protocol_data, self.voltage_command_data = None, None
            # logging.error(f"Error in getDataFromVoltageProtocol: {e}")
        finally:
            self.isRunningProtocol = False
        return self.voltageMembraneCapacitance

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Read analog input from the Arduino via serial in binary format.

        Args:
            samplesPerSec (int): Sampling frequency in Hz.
            recordingTime (float): Recording duration in seconds.

        Returns:
            np.array: Array with shape (2, numSamples) where
                      data[0] = readData (Analog In 1),
                      data[1] = respData (Analog In 2)

        Raises:
            RuntimeError: If the number of samples received does not match the expected count.
        """
        if self.DAQSerial is None:
            logging.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")

        numSamplesExpected = int(recordingTime * samplesPerSec) 
        collecting_data = False
        command = []
        response = []
        ADC_scale =  float(3.3/4096)  # ADC scaling factor
        # Continuously read data from the Arduino
        while True:
            if self.DAQSerial.in_waiting > 0:  # Check if data is available
                line = self.DAQSerial.readline().decode('utf-8').strip()
                if line == "start":
                    collecting_data = True
                    # logging.debug("Data collection started")
                elif line == "end":
                    # logging.debug("Data collection ended")
                    break
                elif collecting_data:
                    # Split the line by comma to get command and response values
                    values = line.split(',')
                    if len(values) == 2:
                        command_value, response_value = (values)
                        command_value = float(command_value)*ADC_scale
                        response_value  = float(response_value)*ADC_scale
                        command.append( (command_value))
                        response.append((response_value))
                        # logging.debug(f"Received: Command={(command_value)}, Response={(response_value)}")
                    else:
                        logging.error(f"Unexpected data format: {line}")


        readData = np.array(command, dtype=float)
        respData = np.array(response, dtype=float)
        
        # logging.debug(f"Read data shape: {readData.shape}, Response data shape: {respData.shape}")
        return np.array([readData, respData])

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Send a square wave command to the Arduino.

        Args:
            wave_freq (float): Frequency of the square wave in Hz.
            samplesPerSec (int): Sampling frequency in Hz.
            dutyCycle (float): Duty cycle as a fraction (0 to 1).
            amplitude (int): Amplitude scaled to 10-bit DAC (0-1023).
            recordingTime (float): Duration in seconds.
        """
        scaling = 1024  # 10-bit DAC scaling
        if self.DAQSerial is None:
            logging.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")

        signalDurationMicros = int(recordingTime * 1e6)  # Duration in microseconds
        waveFrequencyMicros = int(1e6 / wave_freq)  # Wave period in microseconds
        waveAmplitude = int(amplitude*scaling)  # Amplitude scaled to 10-bit DAC
        sampleIntervalMicros = int(1e6 / samplesPerSec)  # Sampling interval in microseconds
        dutyCyclePercent = int(dutyCycle * 100)  # Duty cycle in percentage

        command = f"a {signalDurationMicros} {waveFrequencyMicros} {waveAmplitude} {sampleIntervalMicros} {dutyCyclePercent}\n"
        # command = f"a 40000 50000 511 100 50\n"
 

        self.DAQSerial.reset_input_buffer()
        self.DAQSerial.reset_output_buffer()
        self.DAQSerial.write(command.encode())
        # logging.debug(f"Sent command: {command.strip()}")

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Send a current square wave command to the Arduino.

        Args:
            wave_freq (float): Frequency of the square wave in Hz.
            samplesPerSec (int): Sampling frequency in Hz.
            dutyCycle (float): Duty cycle as a fraction (0 to 1).
            amplitude (int): Amplitude scaled to 10-bit DAC (0-1023).
            recordingTime (float): Duration in seconds.
        """
        self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)

    def getDataFromSquareWave(self, wave_freq, samplesPerSec: int, dutyCycle, amplitude, recordingTime) -> tuple:
        self.equalizer = False
        
        while not self.equalizer:

            # logging.info(f"Sending square wave with frequency: {wave_freq} Hz, amplitude: {amplitude}, recording time: {recordingTime} s.")
            self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
            # logging.info("Reading analog input...")
            data = self._readAnalogInput(samplesPerSec, recordingTime)

            numSamples = int(samplesPerSec * recordingTime)


            if data is not None and data.shape[1] == numSamples:
                # logging.info(f"Data shape: {data.shape[1]}")
                # logging.info(f"Data: {data}")
                try:
                    triggeredSamples = data.shape[1]
                    timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)

                    # Gradient
                    gradientData = np.gradient(data[0], timeData)
                    max_index = np.argmax(gradientData)
                    min_index = np.argmin(gradientData[max_index:])
                    max_grad = np.max(gradientData)
                    min_grad = np.min(gradientData[max_index:])
                    equalizer = abs(max_grad) - abs(min_grad)
                    
                    if abs(max_grad) - abs(min_grad) < 1000:
                        self.equalizer = True
                        # logging.error(f"gradients equal: {equalizer}")
                    else:
                        self.equalizer = False
                        # logging.warning(f"gradients not equal: {equalizer}")
                        # logging.info(f"Max grad value: {gradientData[max_index]}, Max value: {data[0][max_index]}, Max index: {max_index}")
                        # logging.info(f"Min grad value: {gradientData[min_index]}, Max value: {data[0][min_index]}, Min index: {min_index}")

                except Exception as e:
                    # logging.error(f"Error in getDataFromSquareWave: {e}")
                    # logging.info(f"Data shape: {data.shape[1]}")
                    return None, None, None, None, None, None

        if self.equalizer:
            try:
                self.equalizer = False
                min_index = np.argmin(gradientData[max_index:]) + max_index

                left_bound = 100
                right_bound = 300
                respData = data[1]
                readData = data[0]

                respData = data[1][max_index - left_bound:min_index + right_bound]
                readData = data[0][max_index - left_bound:min_index + right_bound]
                timeData = timeData[max_index - left_bound:min_index + right_bound]
                timeData = timeData - timeData[0]

                readData = readData * self.V_CLAMP_VOLT_PER_VOLT
                respData = respData * self.V_CLAMP_VOLT_PER_AMP

                self.latestAccessResistance = 0
                self.latestMembraneResistance = 0
                self.latestMembraneCapacitance = 0
                if self.cellMode:
                    self.latestAccessResistance, self.latestMembraneResistance, self.latestMembraneCapacitance = self._getParamsfromCurrent(readData, respData, timeData, amplitude * self.V_CLAMP_VOLT_PER_VOLT)

                self.totalResistance = 0
                if self.cellMode and self.latestAccessResistance is not None and self.latestMembraneResistance is not None:
                    self.totalResistance = self.latestAccessResistance + self.latestMembraneResistance
                else:
                    self.totalResistance = self._getResistancefromCurrent(respData, amplitude * self.V_CLAMP_VOLT_PER_VOLT)
                    self.totalResistance *= 1e-6  # to have it in MOhms

                return np.array([timeData, respData]), np.array([timeData, readData]), self.totalResistance, self.latestMembraneResistance, self.latestAccessResistance, self.latestMembraneCapacitance
            except Exception as e:
                # logging.error(f"Error in getDataFromSquareWave: {e}")
                # logging.info(f"resistance: {self.totalResistance}, membrane resistance: {self.latestMembraneResistance}, membrane capacitance: {self.latestMembraneCapacitance}")
                
                return None, None, None, None, None, None

    def setCellMode(self, mode: bool) -> None:
        self.cellMode = mode

    def resistance(self):
        # logging.warn("totalResistance", self.totalResistance)
        return self.totalResistance

    def _filter60Hz(self, data):
        samplesPerSec = 20000
        #60 hz filter
        b, a = signal.iirnotch(48.828125, Q=30, fs=samplesPerSec)
        data = signal.filtfilt(b, a, data, irlen=1000)
        return data

    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])
            resistance = cmdVoltage / (highAvg - lowAvg)

            # print("Resistance in func", resistance)
            return resistance
            
        except Exception as e:
            # * we got an invalid square wave, or division by zero
            # logging.error(f"Error in getResistancefromCurrent: {e}")
            return None
        
    def _getParamsfromCurrent(self, readData, respData, timeData, amplitude) -> tuple:
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None
        #check if data is not empty
        if len(readData) != 0 and len(respData) != 0 and len(timeData) != 0:
            try:
                # Create a data frame to obtain the peak and minimum values
                df = pd.DataFrame({'T': timeData, 'X': readData, 'Y': respData})
                #print first line
                # logging.info(f"First line: {df.iloc[0]}")
                #shift time axis to zero
                start = df['T'].iloc[0]
                df['T'] = df['T'] - start
                # dimensionality conversion
                df['T_ms'] = df['T'] * 1000  # converting seconds to milliseconds
                df['X_mV'] = df['X'] * 1000  # converting volts to millivolts
                df['Y_pA'] = df['Y'] * 1e12  # converting amps to picoamps
            
                # Decay filter part
                filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = self.filter_data(df)
            
                self.holding_current = I_prev_pA

                # logging.info("Data filtered")
                peak_time, peak_index, min_time, min_index = plot_params
                #get peak and min values
                I_peak_pA = df.loc[peak_index + 1, 'Y_pA']
                I_peak_time = df.loc[peak_index + 1, 'T_ms']
                # logging.info(f"Peak Current (pA): {I_peak_pA} at index: {peak_index}")
                # logging.info(f"steady state current (pA): {I_post_pA} at index: {min_index-10}")
                # logging.info(f"Peak time: {peak_time}, Peak index: {peak_index}, Min time: {min_time}, Min index: {min_index}")
                # logging.info(f"I_prev_pA: {I_prev_pA}, I_post_pA: {I_post_pA}, I_peak_pA: {I_peak_pA}")
                #group filtered_data, filtered_time, and filtered_command into a data frame
                fit_data = pd.DataFrame({'T_ms': filtered_time, 'X_mV': filtered_command, 'Y_pA': filtered_data})
                #print first line of fit_data
                # logging.info(f"First line of fit_data: {fit_data.iloc[0]}")
                # calculate mean voltage from filtered_command
                mean_voltage = filtered_command.mean()
                start = fit_data['T_ms'].iloc[0]
                fit_data['T_ms'] = fit_data['T_ms'] - start
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA)
                # print(f"m: {m}, t: {t}, b: {b}")
                if m is not None and t is not None and b is not None:
                    tau = 1 / t
                    # Calculate parameters
                    R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
                    # logging.info(f"R_a: {R_a_MOhms}, R_m: {R_m_MOhms}, C_m: {C_m_pF}")
                    
            
            except Exception as e:
                # logging.error(f"Error in getParamsfromCurrent: {e}")
                return None, None, None
                # return 0, 0, 0
        else:
            # logging.error("One or more of the data arrays is empty")
            return 0,0,0
        # logging.info("Returning parameters")
        return R_a_MOhms, R_m_MOhms, C_m_pF

    def filter_data(self, data):
        #calculate derivative of X_mV and add it to the data frame
        # convert to np array
        X_mV = data['X_mV'].to_numpy()
        T_ms = data['T_ms'].to_numpy()

        Y_pA = data['Y_pA'].to_numpy()

        X_dT = np.gradient(X_mV, T_ms)
        data["X_dT"] = X_dT
        # Find the index of the maximum value
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        # Find the index of the minimum value after the maximum value
        peak_current_index = np.argmax(Y_pA)
        peak_time = data.loc[peak_current_index, 'T_ms']
        # get peak and min times
        # positive_peak_time = data.loc[positive_peak_index, 'T_ms'].copy()
        negative_peak_time = data.loc[negative_peak_index, 'T_ms']
        # Extract the data between peaks

        pre_peak_current = data.loc[:positive_peak_index, "Y_pA"]
        sub_data = data.loc[peak_current_index:negative_peak_index, "Y_pA"]
        sub_time = data.loc[peak_current_index:negative_peak_index, "T_ms"]
        sub_command = data.loc[peak_current_index:negative_peak_index, "X_mV"]
        # calculate the mean current prior to voltage pulse (I_prev)
        mean_pre_peak = pre_peak_current.mean()
        # print("Mean pre peak", mean_pre_peak)
        # calculate the mean current post voltage pulse (I_ss)
        gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time.iloc[zero_gradient_index]
        # print(f"Zero gradient time: {zero_gradient_time}")
        # Calculate the mean of the data between the zero gradient time and the min time
        if zero_gradient_time:
            post_peak_current_data = data[(data['T_ms'] >= zero_gradient_time) & (data['T_ms'] <= data.loc[negative_peak_index, 'T_ms'])]
            mean_post_peak = post_peak_current_data['Y_pA'].mean()
        else:
            mean_post_peak = None
        # print("Mean post peak", mean_post_peak)
        return sub_data, sub_time, sub_command, [peak_time, peak_current_index, negative_peak_time, negative_peak_index], mean_pre_peak, mean_post_peak
    
    def monoExp(self,x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, fit_data, I_peak_pA, I_peak_time, I_ss):
        start = fit_data['T_ms'].iloc[0]
        # logging.info(f"Unshifted Start time: {start}")
        # Shift the data to start at 0
        fit_data['T_ms'] = fit_data['T_ms'] - start
        #print all of the fit data  to copy to a txt
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # print("T_ms: ", fit_data['T_ms'])
        # print("Y_pA: ", fit_data['Y_pA']) 

        # Save T_ms and Y_pA to a single CSV file and overwrite it each time
        # fit_data[['T_ms', 'Y_pA']].to_csv(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_patch_clamp_data\test_data.csv", index=False)
        # logging.info("Data saved to CSV")
        # print("I_SS: ", I_ss)
        p0 = (I_peak_pA, I_peak_time, I_ss)
        # print("P0", p0)
        # print if there is a nan value in fit_data
        try:
            # print("NAN VALUES: ", fit_data.isnull().values.any())
            params, _ = scipy.optimize.curve_fit(self.monoExp, fit_data['T_ms'], fit_data['Y_pA'], maxfev=1000000, p0=p0)
            m, t, b = params
            # print("Params: ", params)
            # print("Params", m, t, b)
            return m, t, b
        except Exception as e:
            # logging.error(f"Error in the optimizer: {e}")
            return None, None, None

    def calc_param(self, tau, mean_voltage, I_peak, I_prev, I_ss):
        I_d = I_peak - I_prev  # in pA
        I_dss = I_ss - I_prev  # in pA
        # logging.info(f"tau: {tau}, dmV: {mean_voltage}, I_d: {I_d}, I_dss: {I_dss}")
        # calculate access resistance:
        R_a_Mohms = ((mean_voltage*1e-3) / (I_d*1e-12))*1e-6 # 10 mV / 800 pA = 12.5 MOhms --> supposed to be 10 MOhms
        # print("R_a_MOhms in calc", R_a_Mohms)
        
        # calculate membrane resistance:
        R_m_Mohms = (((mean_voltage*1e-3) - R_a_Mohms*1e6*I_dss*1e-12)/(I_dss*1e-12))*1e-6 #530 Mohms --> supposed to be 500 MOhms
        C_m_pF = (tau*1e-3) / (1/(1/(R_a_Mohms*1e6) + 1/(R_m_Mohms*1e6))) * 1e12 # supposed to be 33 pF
        # print("C_m_pF in calc", C_m_pF)
        return R_a_Mohms, R_m_Mohms, C_m_pF
   

class FakeDAQ:
    def __init__(self):
        self.totalResistance = 6 * 10 ** 6
        self.latest_protocol_data = None
        self.isRunningProtocol = False
        self.current_protocol_data = None
        self.voltage_protocol_data = None
        self.holding_protocol_data = None
        self._deviceLock = threading.Lock()

    def resistance(self):
        return self.totalResistance + np.random.normal(0, 0.1 * 10 ** 6)
    
    def getDataFromVoltageProtocol(self):
        '''Sends a square wave to determine membrane properties, returns time constant, resistance, and capacitance.'''
        self.isRunningProtocol = True
        # self.latest_protocol_data = None # clear data
        self.voltage_protocol_data = None # clear data

        self.voltage_protocol_data, fake_response, fake_total, fake_access,fake_membrane_r,fake_capacitance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
        # self.lastest_protocol_data, resistance = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)

        self.isRunningProtocol = False

        # return self.latest_protocol_data
        return self.voltage_protocol_data

    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
            # Lock the creation and modification of shared data
            with self._deviceLock:
                # Create square wave data array with locked resource access
                data = np.zeros(int(samplesPerSec * recordingTime))
                onTime = int(1 / wave_freq * dutyCycle * samplesPerSec)
                offTime = int(1 / wave_freq * (1 - dutyCycle) * samplesPerSec)
                period = onTime + offTime
                wavesPerSec = samplesPerSec // period

                # Populate the square wave data with amplitude values for 'on' times
                for i in range(wavesPerSec):
                    data[i * period: i * period + onTime] = amplitude

            # Perform calculations outside the lock to avoid GUI conflicts or memory issues
            timeData = np.linspace(0, recordingTime, len(data), dtype=float)
            data = np.array([timeData, data])

            # Set up fake resistance and capacitance values
            fake_resistance = 20
            fake_capacitance = 20
            total_resistance = fake_resistance * 2

            # Return the assembled data
            return data, data, fake_resistance, total_resistance, fake_resistance, fake_capacitance