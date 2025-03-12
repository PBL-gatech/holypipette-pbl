import threading
import time
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal as signal

# Import your hardware libraries as needed
import nidaqmx
import nidaqmx.constants
import serial

from holypipette.controller.base import TaskController

__all__ = ['NiDAQ', 'ArduinoDAQ', 'FakeDAQ']

class DAQ(TaskController):
    """
    Base DAQ class with common methods for patch-clamp protocols.
    Subclasses must override the _readAnalogInput() and _sendSquareWave()
    methods to handle device-specific operations.
    """

    # These constants can be defined at the class level and/or overwritten by subclasses.
    C_CLAMP_AMP_PER_VOLT = None
    C_CLAMP_VOLT_PER_VOLT = None
    V_CLAMP_VOLT_PER_VOLT = None
    V_CLAMP_VOLT_PER_AMP = None

    def __init__(self):
        super().__init__()
        self.pulses = None
        self.pulseRange = None
        self.latest_protocol_data = None
        self.current_protocol_data = None
        self.voltage_protocol_data = None
        self.holding_protocol_data = None
        self._deviceLock = threading.Lock()
        self.isRunningProtocol = False
        self.isRunningVoltageProtocol = False
        self.totalResistance = None
        self.latestAccessResistance = None
        self.latestMembraneResistance = None
        self.latestMembraneCapacitance = None
        self.equalizer = False
        self.holding_current = None
        self.holding_voltage = None
        # False is for BATH mode, True for CELL mode
        self.cellMode = False

    def setCellMode(self, mode: bool) -> None:
        self.cellMode = mode

    # --------------------------
    # ABSTRACT (DEVICE-SPECIFIC)
    # --------------------------
    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Read analog input data from the device.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _readAnalogInput()")

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Send a square wave command to the device.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement _sendSquareWave()")

    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Default implementation for sending a current square wave.
        Subclasses may override if needed.
        """
        return self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)

    # --------------------------
    # COMMON PROTOCOL METHODS
    # --------------------------
    def getDataFromSquareWave(self, wave_freq, samplesPerSec: int, dutyCycle, amplitude, recordingTime) -> tuple:
        """
        Send a square wave (voltage protocol) and acquire the response.
        Returns a tuple containing:
          - [timeData, respData]
          - [timeData, readData]
          - totalResistance, membraneResistance, accessResistance, membraneCapacitance
        """
        self.equalizer = False
        while not self.equalizer:
            with self._deviceLock:
                sendTask = self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
                # Some device tasks (e.g. NI) may return an object with start/stop/close methods.
                if hasattr(sendTask, 'start'):
                    sendTask.start()
                data = self._readAnalogInput(samplesPerSec, recordingTime)
                if hasattr(sendTask, 'stop'):
                    sendTask.stop()
                if hasattr(sendTask, 'close'):
                    sendTask.close()
            numSamples = int(samplesPerSec * recordingTime)
            if data is not None and data.shape[1] == numSamples:
                try:
                    triggeredSamples = data.shape[1]
                    timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
                    gradientData = np.gradient(data[0], timeData)
                    max_index = np.argmax(gradientData)
                    # Look for the minimum after the maximum index
                    min_index = np.argmin(gradientData[max_index:]) + max_index
                    max_grad = np.max(gradientData)
                    min_grad = np.min(gradientData[max_index:])
                    if abs(max_grad) - abs(min_grad) < 1000:
                        self.equalizer = True
                    else:
                        self.equalizer = False
                except Exception as e:
                    return None, None, None, None, None, None

        if self.equalizer:
            try:
                # Reset the equalizer flag for the next use
                self.equalizer = False
                left_bound = 100
                right_bound = 300
                respData = data[1][max_index - left_bound: min_index + right_bound]
                readData = data[0][max_index - left_bound: min_index + right_bound]
                timeData = timeData[max_index - left_bound: min_index + right_bound]
                # Rezero time axis
                timeData = timeData - timeData[0]
                # Convert the data from DAQ units to cell units
                readData = readData * self.V_CLAMP_VOLT_PER_VOLT
                respData = respData * self.V_CLAMP_VOLT_PER_AMP

                # Reset parameters; then calculate cell parameters if in CELL mode.
                self.latestAccessResistance = 0
                self.latestMembraneResistance = 0
                self.latestMembraneCapacitance = 0
                if self.cellMode:
                    (self.latestAccessResistance, self.latestMembraneResistance,
                     self.latestMembraneCapacitance) = self._getParamsfromCurrent(
                        readData, respData, timeData, amplitude * self.V_CLAMP_VOLT_PER_VOLT
                    )

                self.totalResistance = 0
                if self.cellMode and self.latestAccessResistance is not None and self.latestMembraneResistance is not None:
                    self.totalResistance = self.latestAccessResistance + self.latestMembraneResistance
                else:
                    self.totalResistance = self._getResistancefromCurrent(
                        respData, amplitude * self.V_CLAMP_VOLT_PER_VOLT
                    )
                    self.totalResistance *= 1e-6  # convert to MOhms

                return (np.array([timeData, respData]),
                        np.array([timeData, readData]),
                        self.totalResistance,
                        self.latestMembraneResistance,
                        self.latestAccessResistance,
                        self.latestMembraneCapacitance)
            except Exception as e:
                return None, None, None, None, None, None

    def getDataFromCurrentProtocol(self, custom=False, factor=None,
                                   startCurrentPicoAmp=None, endCurrentPicoAmp=None,
                                   stepCurrentPicoAmp=10, highTimeMs=400):
        """
        Send a series of current square waves to measure cell current responses.
        Returns the acquired data, the pulse amplitudes, and the pulse range.
        """
        if self.voltageMembraneCapacitance is None or self.voltageMembraneCapacitance == 0:
            self.voltageMembraneCapacitance = 0
            return None, None, None

        if not custom:
            factor = 2
            startCurrentPicoAmp = round(-self.voltageMembraneCapacitance * factor, -1)
            endCurrentPicoAmp = round(self.voltageMembraneCapacitance * factor, -1)
        else:
            if startCurrentPicoAmp is None or endCurrentPicoAmp is None:
                raise ValueError("startCurrentPicoAmp and endCurrentPicoAmp must be provided when custom is True.")

        self.info(f'Starting Current Protocol with start: {startCurrentPicoAmp}, '
                  f'end: {endCurrentPicoAmp}, step: {stepCurrentPicoAmp}.')
        self.pulses = np.arange(startCurrentPicoAmp, endCurrentPicoAmp + stepCurrentPicoAmp, stepCurrentPicoAmp)
        if 0 not in self.pulses:
            self.pulses = np.insert(self.pulses, len(self.pulses) // 2, 0)
        self.info(f'Pulses: {self.pulses}')
        self.pulseRange = len(self.pulses)
        self.isRunningProtocol = True
        self.latest_protocol_data = None  # clear previous data

        num_waves = int((endCurrentPicoAmp - startCurrentPicoAmp) / stepCurrentPicoAmp) + 2
        # Convert to amps
        startCurrent = startCurrentPicoAmp * 1e-12
        self.info(f"Start Current: {startCurrent}")
        # Determine the square wave frequency (Hz)
        wave_freq = 1 / (2 * highTimeMs * 1e-3)
        samplesPerSec = 20000
        recTime = 4 * highTimeMs * 1e-3

        for i in range(num_waves - 1):
            # First, send a parameter pulse at -20 pA
            amp_pulse = (-20 * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
            wave_pulse = 1 / (2 * (highTimeMs / 2) * 1e-3)
            recording_pulse = 3 * highTimeMs / 2 * 1e-3
            self.info('Sending param pulse at -20 pA square wave.')
            with self._deviceLock:
                sendTask = self._sendSquareWaveCurrent(wave_pulse, samplesPerSec, 0.5, amp_pulse, recording_pulse)
                if hasattr(sendTask, 'start'):
                    sendTask.start()
                data = self._readAnalogInput(samplesPerSec, recording_pulse)
                if hasattr(sendTask, 'stop'):
                    sendTask.stop()
                if hasattr(sendTask, 'close'):
                    sendTask.close()
            respData0 = data[1]
            readData0 = data[0]
            respData0 = respData0 / self.C_CLAMP_VOLT_PER_VOLT
            self.sleep(0.5)

            currentAmps = self.pulses[i] * 1e-12
            self.info(f'Sending {currentAmps * 1e12} pA square wave.')
            amplitude = currentAmps / self.C_CLAMP_AMP_PER_VOLT

            with self._deviceLock:
                sendTask = self._sendSquareWaveCurrent(wave_freq, samplesPerSec, 0.5, amplitude, recTime)
                if hasattr(sendTask, 'start'):
                    sendTask.start()
                data = self._readAnalogInput(samplesPerSec, recTime)
                if hasattr(sendTask, 'stop'):
                    sendTask.stop()
                if hasattr(sendTask, 'close'):
                    sendTask.close()
            respData1 = data[1]
            readData1 = data[0]
            respData1 = respData1 / self.C_CLAMP_VOLT_PER_VOLT
            # Combine the two phases of data
            respData = np.concatenate((respData0, respData1))
            readData = np.concatenate((readData0, readData1))
            triggeredSamples = respData.shape[0]
            timeData = np.linspace(0, triggeredSamples / samplesPerSec, triggeredSamples, dtype=float)
            self.sleep(0.5)
            if self.current_protocol_data is None:
                self.current_protocol_data = [[timeData, respData, readData]]
            else:
                self.current_protocol_data.append([timeData, respData, readData])
        self.isRunningProtocol = False
        return self.current_protocol_data, self.pulses, self.pulseRange

    def getDataFromHoldingProtocol(self):
        """
        Measures holding (baseline) currents to determine spontaneous activity.
        """
        self.isRunningProtocol = True
        self.holding_protocol_data = None
        with self._deviceLock:
            data = self._readAnalogInput(50000, 1)
        respData = data[1]
        readData = data[0]
        triggeredSamples = respData.shape[0]
        timeData = np.linspace(0, triggeredSamples / 50000, triggeredSamples, dtype=float)
        self.isRunningProtocol = False
        self.holding_protocol_data = np.array([timeData, respData, readData])
        return self.holding_protocol_data

    def getDataFromVoltageProtocol(self):
        """
        Sends a square wave to determine membrane properties (e.g. time constant, resistance, capacitance).
        Retries up to 5 times if an invalid (zero) capacitance is obtained.
        """
        self.voltage_protocol_data, self.voltage_command_data = None, None  # clear data
        max_attempts = 5
        attempts = 0
        try:
            self.isRunningProtocol = True
            while attempts < max_attempts:
                attempts += 1
                result = self.getDataFromSquareWave(20, 20000, 0.5, 0.5, 0.05)
                if result is None:
                    continue
                (self.voltage_protocol_data, self.voltage_command_data,
                 self.voltageTotalResistance, self.voltageMembraneResistance,
                 self.voltageAccessResistance, self.voltageMembraneCapacitance) = result
                if self.voltageMembraneCapacitance != 0:
                    break
                else:
                    self.warning(f"Attempt {attempts}: Capacitance is zero, retrying...")
        except Exception as e:
            self.voltage_protocol_data, self.voltage_command_data = None, None
            self.error(f"Error in getDataFromVoltageProtocol: {e}")
        finally:
            self.isRunningProtocol = False
        return self.voltageMembraneCapacitance

    # --------------------------
    # COMMON CALCULATION METHODS
    # --------------------------
    def _getResistancefromCurrent(self, data, cmdVoltage) -> float | None:
        try:
            mean = np.mean(data)
            lowAvg = np.mean(data[data < mean])
            highAvg = np.mean(data[data > mean])
            resistance = cmdVoltage / (highAvg - lowAvg)
            return resistance
        except Exception as e:
            return None

    def _getParamsfromCurrent(self, readData, respData, timeData, amplitude) -> tuple:
        """
        Calculate access resistance, membrane resistance, and membrane capacitance
        from the response of a voltage protocol.
        """
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None
        if len(readData) != 0 and len(respData) != 0 and len(timeData) != 0:
            try:
                df = pd.DataFrame({'T': timeData, 'X': readData, 'Y': respData})
                # Shift time axis to start at zero
                start = df['T'].iloc[0]
                df['T'] = df['T'] - start
                df['T_ms'] = df['T'] * 1000  # seconds -> milliseconds
                df['X_mV'] = df['X'] * 1000   # volts -> millivolts
                df['Y_pA'] = df['Y'] * 1e12   # amps -> picoamps

                filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = self.filter_data(df)
                self.holding_current = I_prev_pA
                peak_time, peak_index, min_time, min_index = plot_params
                I_peak_pA = df.loc[peak_index + 1, 'Y_pA']
                I_peak_time = df.loc[peak_index + 1, 'T_ms']
                fit_data = pd.DataFrame({'T_ms': filtered_time, 'X_mV': filtered_command, 'Y_pA': filtered_data})
                mean_voltage = filtered_command.mean()
                start = fit_data['T_ms'].iloc[0]
                fit_data['T_ms'] = fit_data['T_ms'] - start
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA)
                if m is not None and t is not None and b is not None:
                    tau = 1 / t
                    R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
            except Exception as e:
                return None, None, None
        else:
            return 0, 0, 0
        return R_a_MOhms, R_m_MOhms, C_m_pF

    def filter_data(self, data):
        """
        Process the acquired data to extract the relevant segment for curve-fitting.
        """
        X_mV = data['X_mV'].to_numpy()
        T_ms = data['T_ms'].to_numpy()
        Y_pA = data['Y_pA'].to_numpy()
        X_dT = np.gradient(X_mV, T_ms)
        data["X_dT"] = X_dT
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        peak_current_index = np.argmax(data['Y_pA'])
        peak_time = data.loc[peak_current_index, 'T_ms']
        negative_peak_time = data.loc[negative_peak_index, 'T_ms']
        pre_peak_current = data.loc[:positive_peak_index, "Y_pA"]
        sub_data = data.loc[peak_current_index:negative_peak_index, "Y_pA"]
        sub_time = data.loc[peak_current_index:negative_peak_index, "T_ms"]
        sub_command = data.loc[peak_current_index:negative_peak_index, "X_mV"]
        mean_pre_peak = pre_peak_current.mean()
        gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time.iloc[zero_gradient_index]
        if zero_gradient_time:
            post_peak_current_data = data[(data['T_ms'] >= zero_gradient_time) &
                                          (data['T_ms'] <= data.loc[negative_peak_index, 'T_ms'])]
            mean_post_peak = post_peak_current_data['Y_pA'].mean()
        else:
            mean_post_peak = None
        return sub_data, sub_time, sub_command, [peak_time, peak_current_index, negative_peak_time, negative_peak_index], mean_pre_peak, mean_post_peak

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, fit_data, I_peak_pA, I_peak_time, I_ss):
        """
        Use curve fitting to extract the decay time constant.
        """
        start = fit_data['T_ms'].iloc[0]
        fit_data['T_ms'] = fit_data['T_ms'] - start
        p0 = (I_peak_pA, I_peak_time, I_ss)
        try:
            params, _ = scipy.optimize.curve_fit(self.monoExp, fit_data['T_ms'], fit_data['Y_pA'],
                                                   maxfev=1000000, p0=p0)
            m, t, b = params
            return m, t, b
        except Exception as e:
            return None, None, None

    def calc_param(self, tau, mean_voltage, I_peak, I_prev, I_ss):
        """
        Calculate access resistance (R_a), membrane resistance (R_m) and membrane capacitance (C_m)
        from the measured currents.
        """
        I_d = I_peak - I_prev   # in pA
        I_dss = I_ss - I_prev   # in pA
        R_a_MOhms = ((mean_voltage * 1e-3) / (I_d * 1e-12)) * 1e-6
        R_m_MOhms = (((mean_voltage * 1e-3) - R_a_MOhms * 1e6 * I_dss * 1e-12) / (I_dss * 1e-12)) * 1e-6
        C_m_pF = (tau * 1e-3) / (1 / (1 / (R_a_MOhms * 1e6) + 1 / (R_m_MOhms * 1e6))) * 1e12
        return R_a_MOhms, R_m_MOhms, C_m_pF

# ========================================================
#  NI-DAQ Subclass
# ========================================================
class NiDAQ(DAQ):
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 mV per pA (DAQ input)

    def __init__(self, readDev, readChannel, cmdDev, cmdChannel, respDev, respChannel):
        super().__init__()
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
        self.cellMode = False
        self.info(f'Using {self.readDev}/{self.readChannel} for reading; '
                  f'{self.cmdDev}/{self.cmdChannel} for command; '
                  f'and {self.respDev}/{self.respChannel} for response.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        numSamples = int(samplesPerSec * recordingTime)
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                f'{self.readDev}/{self.readChannel}',
                max_val=10, min_val=0,
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF
            )
            task.ai_channels.add_ai_voltage_chan(
                f'{self.respDev}/{self.respChannel}',
                max_val=10, min_val=0,
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFF
            )
            task.timing.cfg_samp_clk_timing(
                samplesPerSec,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=numSamples
            )
            data = task.read(number_of_samples_per_channel=numSamples, timeout=10)
            data = np.array(data, dtype=float)
            task.stop()
        if data is None or np.where(data == None)[0].size > 0:
            data = np.zeros((2, numSamples))
        return data

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(f'{self.cmdDev}/{self.cmdChannel}')
        task.timing.cfg_samp_clk_timing(
            samplesPerSec,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )
        numSamples = int(samplesPerSec * recordingTime)
        data = np.zeros(numSamples)
        period = int(samplesPerSec / wave_freq)
        onTime = int(period * dutyCycle)
        for i in range(0, numSamples, period):
            data[i:i + onTime] = amplitude
            data[i + onTime:i + period] = 0
        task.write(data)
        return task
# ========================================================
#  ArduinoDAQ Subclass
# ========================================================
class ArduinoDAQ(DAQ):
    C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
    C_CLAMP_VOLT_PER_VOLT = (10 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
    V_CLAMP_VOLT_PER_VOLT = (20 * 1e-3)  # 20 mV per V (DAQ output)
    V_CLAMP_VOLT_PER_AMP = (2 * 1e-9)   # 2 V per A (DAQ input)

    def __init__(self, DAQSerial=None):
        if DAQSerial is not None and isinstance(DAQSerial, serial.Serial):
            self.DAQSerial = DAQSerial
            self.info(f"ArduinoDAQ initialized with serial port: {self.DAQSerial.port} at {self.DAQSerial.baudrate} baud.")
        else:
            self.DAQSerial = None
            self.error("DAQSerial must be an instance of serial.Serial and already opened.")
            raise ValueError("DAQSerial must be an instance of serial.Serial and already opened.")
        super().__init__()
        self.info(f'Using {self.DAQSerial.port} for reading and writing.')

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        if self.DAQSerial is None:
            self.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")
        numSamplesExpected = int(recordingTime * samplesPerSec)
        collecting_data = False
        command_vals = []
        response_vals = []
        ADC_scale = 3.3 / 4096  # ADC scaling factor

        while True:
            if self.DAQSerial.in_waiting > 0:
                line = self.DAQSerial.readline().decode('utf-8').strip()
                if line == "start":
                    collecting_data = True
                elif line == "end":
                    break
                elif collecting_data:
                    values = line.split(',')
                    if len(values) == 2:
                        cmd_val, resp_val = values
                        command_vals.append(float(cmd_val) * ADC_scale)
                        response_vals.append(float(resp_val) * ADC_scale)
                    else:
                        self.error(f"Unexpected data format: {line}")

        readData = np.array(command_vals, dtype=float)
        respData = np.array(response_vals, dtype=float)
        return np.array([readData, respData])

    def _sendSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        scaling = 1024  # 10-bit DAC scaling
        if self.DAQSerial is None:
            self.error("Serial port not initialized.")
            raise RuntimeError("Serial port not initialized.")

        signalDurationMicros = int(recordingTime * 1e6)  # Duration in microseconds
        waveFrequencyMicros = int(1e6 / wave_freq)  # Period in microseconds
        waveAmplitude = int(amplitude * scaling)  # Scale amplitude
        sampleIntervalMicros = int(1e6 / samplesPerSec)  # Sampling interval
        dutyCyclePercent = int(dutyCycle * 100)
        command_str = f"a {signalDurationMicros} {waveFrequencyMicros} {waveAmplitude} {sampleIntervalMicros} {dutyCyclePercent}\n"
        self.DAQSerial.reset_input_buffer()
        self.DAQSerial.reset_output_buffer()
        self.DAQSerial.write(command_str.encode())
        # For Arduino, we do not return a task object; simply return None.
        return None

# ========================================================
#  FakeDAQ Subclass 
# ========================================================
class FakeDAQ(DAQ):
    def __init__(self):
        super().__init__()
        self.totalResistance = 6 * 10 ** 6  # Set a baseline fake resistance

    def resistance(self):
        return self.totalResistance + np.random.normal(0, 0.1 * 10 ** 6)

    def getDataFromVoltageProtocol(self):
        """
        For the fake DAQ, simply return a fake square wave response.
        """
        self.isRunningProtocol = True
        self.voltage_protocol_data = None
        # Use a fake square wave generator
        result = self.getDataFromSquareWave(20, 50000, 0.5, 0.5, 0.03)
        self.isRunningProtocol = False
        return self.voltage_protocol_data

    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        with self._deviceLock:
            numSamples = int(samplesPerSec * recordingTime)
            data = np.zeros(numSamples)
            onTime = int((1 / wave_freq) * dutyCycle * samplesPerSec)
            period = int(1 / wave_freq * samplesPerSec)
            waves = samplesPerSec // period
            for i in range(waves):
                start = i * period
                data[start:start + onTime] = amplitude
        timeData = np.linspace(0, recordingTime, numSamples, dtype=float)
        # Fake both channels as the same data for simplicity.
        data_arr = np.array([timeData, data])
        fake_resistance = 20
        fake_capacitance = 20
        total_resistance = fake_resistance * 2
        return data_arr, data_arr, fake_resistance, total_resistance, fake_resistance, fake_capacitance
