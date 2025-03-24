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

import threading
import numpy as np
import scipy.optimize
import collections
import logging


class DAQAcquisitionThread(threading.Thread):
    """
    A thread that continuously sends a square wave through the DAQ,
    acquires the response, calculates resistance and capacitance,
    and stores only the most recent measurement in a queue (maxlen=1).

    Parameters:
      daq: An instance of a DAQ subclass (e.g. NiDAQ, ArduinoDAQ, or FakeDAQ)
      wave_freq: Frequency of the square wave (Hz)
      samplesPerSec: Sampling rate in samples per second
      dutyCycle: Duty cycle for the square wave (between 0 and 1)
      amplitude: Amplitude of the square wave (in DAQ output units)
      recordingTime: Duration of each acquisition (seconds)
      callback: Optional function to call with the acquired data.
                The callback receives:
                  totalResistance, accessResistance, membraneResistance,
                  membraneCapacitance, respData, readData.
      interval: Time (seconds) to wait between acquisitions (default is recordingTime).
    """
    def __init__(self, daq, wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                 amplitude=0.5, recordingTime=0.025, callback=None, interval=None):
        super().__init__(daemon=True)
        self.daq = daq
        self.wave_freq = wave_freq
        self.samplesPerSec = samplesPerSec
        self.dutyCycle = dutyCycle
        self.amplitude = amplitude
        self.recordingTime = recordingTime
        self.callback = callback
        self.interval = interval if interval is not None else recordingTime
        self.running = True
        self._last_data_queue = collections.deque(maxlen=1)

    def run(self):
        while self.running:
            if not self.daq.isRunningProtocol:
                try:
                    data = self.daq.getDataFromSquareWave(
                        self.wave_freq,
                        self.samplesPerSec,
                        self.dutyCycle,
                        self.amplitude,
                        self.recordingTime
                    )
                    if data:
                        (timeData, respData), (timeData, readData), totalResistance, \
                            membraneResistance, accessResistance, membraneCapacitance = data

                        measurement = {
                            "timeData": timeData,
                            "respData": respData,
                            "readData": readData,
                            "totalResistance": totalResistance,
                            "membraneResistance": membraneResistance,
                            "accessResistance": accessResistance,
                            "membraneCapacitance": membraneCapacitance
                        }
                        
                        self._last_data_queue.append(measurement)
                        
                        if self.callback is not None:
                            self.callback(
                                totalResistance,
                                accessResistance,
                                membraneResistance,
                                membraneCapacitance,
                                respData,
                                readData
                            )
                except Exception as e:
                    pass
                    # logging.warning(f"Error in DAQAcquisitionThread: {e}")
                time.sleep(self.interval)
            else:
                continue

    def get_last_data(self):
        """Return the most recent measurement or None if not available."""
        return self._last_data_queue[-1] if self._last_data_queue else None

    def stop(self):
        """Stop the acquisition thread."""
        self.running = False
class DAQ(TaskController):
    """
    Base DAQ class with common methods for patch-clamp protocols.
    Subclasses must override the _readAnalogInput() and _sendSquareWave()
    methods to handle device-specific operations.
    """
    # Class-level constants (may be overwritten by subclasses)
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
        self.info(f"Setting cell mode to {mode}")

    def getCellMode(self) -> bool:
        return self.cellMode
    # --------------------------
    # Acquisition Methods
    # --------------------------
    def start_acquisition(self, wave_freq=80, samplesPerSec=10000, dutyCycle=0.5,
                          amplitude=0.5, recordingTime=0.012, interval=None, callback=None):
        """
        Start the asynchronous acquisition thread for continuous DAQ measurements.
        Parameters:
          wave_freq: Frequency of the square wave in Hz.
          samplesPerSec: Sampling rate (samples per second).
          dutyCycle: Duty cycle (0 to 1) of the square wave.
          amplitude: Amplitude for the square wave.
          recordingTime: Duration for each acquisition (seconds).
          interval: Time (seconds) between acquisitions (defaults to recordingTime).
          callback: Optional function to call with new data.
        """
        if not hasattr(self, "_daq_acq_thread") or self._daq_acq_thread is None:
            self._daq_acq_thread = DAQAcquisitionThread(
                daq=self,
                wave_freq=wave_freq,
                samplesPerSec=samplesPerSec,
                dutyCycle=dutyCycle,
                amplitude=amplitude,
                recordingTime=recordingTime,
                callback=callback,
                interval=interval
            )
            self._daq_acq_thread.start()

    def get_last_acquisition(self):
        """
        Retrieve the most recent measurement from the acquisition thread.
        Returns a dictionary with keys:
          "timeData", "respData", "readData", "totalResistance",
          "membraneResistance", "accessResistance", "membraneCapacitance"
        or None if no data is available.
        """
        if hasattr(self, "_daq_acq_thread") and self._daq_acq_thread:
            return self._daq_acq_thread.get_last_data()
        return None

    def stop_acquisition(self):
        """
        Stop the asynchronous DAQ acquisition thread.
        """
        if hasattr(self, "_daq_acq_thread") and self._daq_acq_thread:
            self._daq_acq_thread.stop()
            self._daq_acq_thread.join()
            self._daq_acq_thread = None
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
                if self.getCellMode():
                    (self.latestAccessResistance, self.latestMembraneResistance,
                    self.latestMembraneCapacitance) = self._getParamsfromCurrent(
                        readData, respData, timeData, amplitude * self.V_CLAMP_VOLT_PER_VOLT
                    )

                self.totalResistance = 0
                if self.getCellMode() and self.latestAccessResistance is not None and self.latestMembraneResistance is not None:
                    self.totalResistance = self.latestAccessResistance + self.latestMembraneResistance
                else:
                    self.totalResistance = self._getResistancefromCurrent(
                        respData, amplitude * self.V_CLAMP_VOLT_PER_VOLT
                    )
                    if self.totalResistance is not None:
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
            factor = 1
            startCurrentPicoAmp = round(-self.voltageMembraneCapacitance * factor, -1)
            endCurrentPicoAmp = round(self.voltageMembraneCapacitance * factor, -1)
 
            
            if startCurrentPicoAmp < -200:
                self.warning(f"starting current too great: {startCurrentPicoAmp}")
                startCurrentPicoAmp = -200
                self.info(f"starting current limited to: {startCurrentPicoAmp}")
            if endCurrentPicoAmp > 200:
                endCurrentPicoAmp = 200

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
        # self.info(f"Start Current: {startCurrent}")
        # Determine the square wave frequency (Hz)
        wave_freq = 1 / (2 * highTimeMs * 1e-3)
        samplesPerSec = 100000
        recTime = 4 * highTimeMs * 1e-3

        for i in range(num_waves - 1):
            # First, send a parameter pulse at -20 pA
            amp_pulse = (-20 * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
            wave_pulse = 1 / (2 * (highTimeMs / 2) * 1e-3)
            recording_pulse = 3 * highTimeMs / 2 * 1e-3
            # self.info('Sending param pulse at -20 pA square wave.')
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
                result = self.getDataFromSquareWave(wave_freq=40, samplesPerSec=100000, dutyCycle=0.5, amplitude=0.5, recordingTime=0.025)
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
    
    def capacitance(self):
        """
        Returns the latest membrane capacitance measurement.
        If no measurement is available, returns None.
        """
        return self.latestMembraneCapacitance

    def resistance(self):
        """
        Returns the latest total resistance measurement.
        If no measurement is available, returns None.
        """
        return self.totalResistance
    
    def accessResistance(self):
        """
        Returns the latest access resistance measurement.
        If no measurement is available, returns None.
        """
        return self.latestAccessResistance

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

    def filter_data(self, T_ms, X_mV, Y_pA):
        """
        Process the acquired data to extract the relevant segment for curve-fitting,
        using NumPy arrays instead of a pandas DataFrame.
        
        Parameters:
          T_ms: np.array of time in milliseconds (shifted so first element is 0)
          X_mV: np.array of command voltages in millivolts
          Y_pA: np.array of responses in picoamps
          
        Returns:
          sub_data: portion of Y_pA between the peak and negative peak indices
          sub_time: corresponding time points from T_ms
          sub_command: corresponding X_mV values
          plot_params: list of [peak_time, peak_current_index, negative_peak_time, negative_peak_index]
          mean_pre_peak: mean of Y_pA from start until the positive peak in dX/dT
          mean_post_peak: mean of Y_pA from the first near-zero gradient point to the negative peak
        """
        # Compute gradient of the command signal
        X_dT = np.gradient(X_mV, T_ms)
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        peak_current_index = np.argmax(Y_pA)
        peak_time = T_ms[peak_current_index]
        negative_peak_time = T_ms[negative_peak_index]
        pre_peak_current = Y_pA[:positive_peak_index+1]
        sub_data = Y_pA[peak_current_index:negative_peak_index+1]
        sub_time = T_ms[peak_current_index:negative_peak_index+1]
        sub_command = X_mV[peak_current_index:negative_peak_index+1]
        mean_pre_peak = pre_peak_current.mean()
        sub_gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(sub_gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time[zero_gradient_index]
        if zero_gradient_time is not None:
            mask = (T_ms >= zero_gradient_time) & (T_ms <= T_ms[negative_peak_index])
            mean_post_peak = Y_pA[mask].mean() if np.any(mask) else None
        else:
            mean_post_peak = None
        plot_params = [peak_time, peak_current_index, negative_peak_time, negative_peak_index]
        return sub_data, sub_time, sub_command, plot_params, mean_pre_peak, mean_post_peak

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, fit_data, I_peak_pA, I_peak_time, I_ss):
        """
        Fit the mono-exponential decay model using NumPy arrays.
        
        Parameters:
          fit_data: dict with keys 'T_ms', 'X_mV', 'Y_pA' (all numpy arrays)
          I_peak_pA: Peak current (pA) to help seed the fit
          I_peak_time: Time corresponding to the peak current (ms)
          I_ss: Steady-state current (pA)
          
        Returns:
          m, t, b: Fitted parameters of the model: m * exp(-t*x) + b
        """
        xdata = fit_data['T_ms']
        ydata = fit_data['Y_pA']
        p0 = [I_peak_pA, I_peak_time, I_ss]

        def residuals(params, x, y):
            return self.monoExp(x, *params) - y

        def jac(params, x, y):
            m, t, b = params
            exp_val = np.exp(-t * x)
            J0 = exp_val                   # d/dm
            J1 = -m * x * exp_val          # d/dt
            J2 = np.ones_like(x)           # d/db
            return np.vstack((J0, J1, J2)).T

        res = scipy.optimize.least_squares(
            residuals, p0, jac=jac, args=(xdata, ydata), max_nfev=1000000
        )
        if res.success:
            m, t, b = res.x
            return m, t, b
        else:
            return None, None, None

    def _getParamsfromCurrent(self, readData, respData, timeData, amplitude) -> tuple:
        """
        Calculate access resistance, membrane resistance, and membrane capacitance
        from the response of a voltage protocol using NumPy arrays instead of pandas.
        """
        R_a_MOhms, R_m_MOhms, C_m_pF = None, None, None
        if len(readData) and len(respData) and len(timeData):
            try:
                # Shift time to start at zero and convert units:
                T = timeData - timeData[0]           # seconds (shifted)
                T_ms = T * 1000                      # convert to milliseconds
                X_mV = readData * 1000               # volts -> millivolts
                Y_pA = respData * 1e12               # amps -> picoamps

                # Use the numpy-based filter_data function.
                filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = \
                    self.filter_data(T_ms, X_mV, Y_pA)
                self.holding_current = I_prev_pA
                peak_time, peak_index, negative_peak_time, negative_peak_index = plot_params
                if peak_index + 1 < len(Y_pA):
                    I_peak_pA = Y_pA[peak_index + 1]
                    I_peak_time = T_ms[peak_index + 1]
                else:
                    I_peak_pA = Y_pA[peak_index]
                    I_peak_time = T_ms[peak_index]
                # Prepare the data for curve fitting. Time axis zeroed.
                fit_data = {
                    'T_ms': filtered_time - filtered_time[0],
                    'X_mV': filtered_command,
                    'Y_pA': filtered_data
                }
                mean_voltage = filtered_command.mean()
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA)
                if m is not None and t is not None and b is not None:
                    tau = 1 / t
                    R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
            except Exception as e:
                return None, None, None
        else:
            return 0, 0, 0
        return R_a_MOhms, R_m_MOhms, C_m_pF

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
        # Initialize the DAQ device and set up channels, start the acquisition thread
        self.start_acquisition(wave_freq=40, samplesPerSec=100000, dutyCycle=0.5,
                                amplitude=0.5, recordingTime=0.025, interval=None)

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
