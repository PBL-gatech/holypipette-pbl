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
                
                except nidaqmx.errors.DaqError as e:
                    # Log once, then try to heal if it was a FIFO error
                    self.daq.warning(f"Acquisition error {e.error_code}: {e}")
                    if e.error_code in (-200279, -200284, -200286):   # FIFO overr/underrun
                        with self.daq._deviceLock:
                            self.daq._restartAcquisition()
                    time.sleep(0.05)       # brief pause before retry                   # Log once, then try to heal if it was a FIFO error


                except Exception as e:
                    pass
                    logging.warning(f"Error in DAQAcquisitionThread: {e}")

                time.sleep(self.interval)
            else:
                continue

    def get_last_data(self):
        """Return the most recent measurement or None if not available."""
        return dict(self._last_data_queue[-1]) if self._last_data_queue else None

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
    
    def getDataFromSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Single-cycle acquisition: fire the pre-made AO buffer, read back AI,
        then process into (time/resp, time/read, totalR, memR, accR, memC).
        """
        with self._deviceLock:
            # Device-specific wrappers (implemented in NiDAQ)
            self._sendSquareWave(wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime)
            raw = self._readAnalogInput(samplesPerSec, recordingTime)

        # Process raw 2×N array -> six-tuple
        return self._squareWaveProcessor(raw, samplesPerSec, amplitude)
      
    # --------------------------
    # ABSTRACT (DEVICE-SPECIFIC)
    # --------------------------
    
    def _setupAcquisition(self):
        """
        Abstract: set up and start ai_task & ao_task.
        """
        raise NotImplementedError("Subclasses must implement _setupAcquisition()")
    
    def _restartAcquisition(self):
        """
        Abstract: restart the acquisition, if it hangs or needs to be reset.
        This is used to reset the DAQ device when needed.
        """
        raise NotImplementedError("Subclasses must implement _restartAcquisition()")

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

    def getDataFromCurrentProtocol(self, *args, **kwargs):
        """Run the current-step protocol (I-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getDataFromHoldingProtocol(self, *args, **kwargs):
        """Acquire baseline holding data (no command output).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    def getDataFromVoltageProtocol(self, *args, **kwargs):
        """Run the voltage-step protocol (V-clamp).

        Sub-classes **must** implement this.
        """
        raise NotImplementedError("Implement in subclass.")

    # --------------------------
    # COMMON CALCULATION METHODS
    # --------------------------

    def createSquareWave(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate and cache one buffer of a square wave in self.wave.
        """
        numSamples = int(samplesPerSec * recordingTime)
        period = int(samplesPerSec / wave_freq)
        onTime = int(period * dutyCycle)
        wave = np.zeros(numSamples, dtype=float)
        for i in range(0, numSamples, period):
            wave[i:i+onTime] = amplitude
        self.wave = wave
        self._wave_rate = samplesPerSec
        self._wave_samples = numSamples

    def _squareWaveProcessor(self, raw_data, samplesPerSec, amplitude):
        """
        Cut out the pulse, convert units, compute R & C.
        Identical to our previous implementation, unchanged.
        """
        read = raw_data[0]
        resp = raw_data[1]
        N = read.size
        timeData = np.linspace(0, N/samplesPerSec, N, dtype=float)

        grad = np.gradient(read, timeData)
        max_i = np.argmax(grad)
        min_i = np.argmin(grad[max_i:]) + max_i

        left, right = 0, 300
        idx0 = max(0, max_i-left)
        idx1 = min(N, min_i+right)
        td = timeData[idx0:idx1] - timeData[idx0]
        rd = resp[idx0:idx1] * self.V_CLAMP_VOLT_PER_AMP
        cd = read[idx0:idx1] * self.V_CLAMP_VOLT_PER_VOLT

        if self.getCellMode():
            accR, memR, memC = self._getParamsfromCurrent(
                cd, rd, td, amplitude * self.V_CLAMP_VOLT_PER_VOLT
            )
            totalR = accR + memR
        else:
            totalR = self._getResistancefromCurrent(rd, amplitude * self.V_CLAMP_VOLT_PER_VOLT)
            if totalR is not None:
                totalR *= 1e-6

            accR, memR, memC = None, None, None

        return (np.array([td, rd]),
                np.array([td, cd]),
                totalR,
                memR,
                accR,
                memC)

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
                m, t, b = self.optimizer(fit_data, I_peak_pA, I_peak_time, I_post_pA) # feed in previous initial conditions if exists
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
        # 1) Pre-generate the square wave buffer
        self.createSquareWave(
            wave_freq=80, samplesPerSec=5000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025
        )

        # 2) Configure & start persistent AO/AI tasks
        self._setupAcquisition()

        # 3) Kick off the acquisition thread
        self.start_acquisition(
            wave_freq=80, samplesPerSec=5000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025
        )


    def _restartAcquisition(self):
        try:
            self.ai_task.stop(); self.ao_task.stop()
        except Exception:
            pass                           # tasks may already be stopped
        try:
            self.ai_task.close(); self.ao_task.close()
        except Exception:
            pass
        # Re-create tasks exactly as at start-up
        self._setupAcquisition()
        try:
            # read and discard one buffer length
            _ = self.ai_task.read(
                number_of_samples_per_channel=self._wave_samples, timeout=1.0)
        except nidaqmx.errors.DaqError:
            pass

        
    def _setupAcquisition(self):
        """
        Configure the continuous AI + AO pair so they start exactly like
        INIT_memtest.vi in LabVIEW:

        ▸ AI owns its own sample clock (CONTINUOUS).
        ▸ AO runs at the same rate, in regeneration mode.
        ▸ AO is armed with a digital-edge start-trigger that listens to
            the AI StartTrigger ("/<Dev>/ai/StartTrigger").
        ▸ Tasks are started in LabVIEW order:  AO first (armed) → AI starts
            (emits trigger) → AO begins automatically.
        """
        import nidaqmx
        import nidaqmx.constants as c

        # ------------------------------------------------------------------
        # 1.  ANALOG INPUT  (read + response channels)
        # ------------------------------------------------------------------
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0,
        )
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=c.TerminalConfiguration.DIFF,
            min_val=-10.0, max_val=10.0,
        )
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
        )
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size,
            self._wave_rate * 10)   # keep 10 s of slack (~200 k samples)
                # LabVIEW “Get Terminal with Device Prefix”  →  "/<device>/ai/StartTrigger"
        # On cDAQ the routable line lives on the CHASSIS, not the module.
        if "Mod" in self.readDev:                 # e.g.  "cDAQ1Mod1"
            chassis_name = self.readDev.split("Mod")[0]   # "cDAQ1"
            start_trig_term = f"/{chassis_name}/ai/StartTrigger"
        else:                                     # PCIe/X-series, USB -- use the device itself
            start_trig_term = f"/{self.readDev}/ai/StartTrigger"


        # ------------------------------------------------------------------
        # 2.  ANALOG OUTPUT  (command channel)
        # ------------------------------------------------------------------
        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(f"{self.cmdDev}/{self.cmdChannel}")
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=self._wave_rate,
            sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=self._wave_samples,
        )
        # Regeneration so a single-period buffer loops forever
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        # Arm AO with the AI StartTrigger
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            start_trig_term, trigger_edge=c.Edge.RISING
        )

        # ------------------------------------------------------------------
        # 3.  PRIME AO FIFO  (avoid -200462)
        # ------------------------------------------------------------------
        self.ao_task.write(self.wave, auto_start=False)

        # ------------------------------------------------------------------
        # 4.  START TASKS  (AO waits → AI fires trigger)
        # ------------------------------------------------------------------
        self.ao_task.start()   # armed / waiting for trigger
        self.ai_task.start()   # emits /ai/StartTrigger → AO begins

        # ------------------------------------------------------------------
        # 5.  Cache the current-wave signature
        # ------------------------------------------------------------------
        self._cur_wave_freq  = 40            # matches the pre-built buffer
        self._cur_wave_rate  = self._wave_rate
        self._cur_wave_duty  = 0.5
        self._cur_wave_amp   = 0.5
        self._cur_wave_rtime = 0.025

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Wrapper for the AI task: read exactly one bufferful.
        """
        numSamples = self._wave_samples
        data = self.ai_task.read(
            number_of_samples_per_channel=numSamples,
            timeout=10.0
        )
        return np.array(data, dtype=float)

    def _sendSquareWave(self, wave_freq, samplesPerSec,
                        dutyCycle, amplitude, recordingTime):
        """
        Update the AO buffer *without* disturbing the trigger relationship.

        • If the requested wave is identical to the one already streaming,
        do nothing (prevents DAQmx -200015).
        • If the length / rate is unchanged but the shape is different,
        rewrite the buffer **in-place** while the task is running
        (regen-mode allows this).
        • If rate or length must change, stop *both* tasks, re-configure,
        and restart them **in the same trigger order** (AO-first, AI-second).
        """
        import nidaqmx.constants as c

        same_shape = (wave_freq     == getattr(self, "_cur_wave_freq",  None) and
                    samplesPerSec == getattr(self, "_cur_wave_rate",  None) and
                    dutyCycle     == getattr(self, "_cur_wave_duty",  None) and
                    amplitude     == getattr(self, "_cur_wave_amp",   None) and
                    recordingTime == getattr(self, "_cur_wave_rtime", None))

        if same_shape:
            return  # nothing to do

        # ---------- build the new buffer ---------------------------------------
        self.createSquareWave(wave_freq, samplesPerSec,
                            dutyCycle, amplitude, recordingTime)

        # ---------- CASE 1: rate & length unchanged  ---------------------------
        if samplesPerSec == self._cur_wave_rate and self._wave_samples == len(self.wave):
            # Task is running; regen_mode lets us replace data in FIFO
            self.ao_task.write(self.wave, auto_start=False)

        # ---------- CASE 2: rate OR length changed  ----------------------------
        else:
            # 1) stop both tasks
            self.ai_task.stop()
            self.ao_task.stop()

            # 2) re-configure AO timing for new rate / buffer size
            self.ao_task.timing.cfg_samp_clk_timing(
                rate=self._wave_rate,
                sample_mode=c.AcquisitionType.CONTINUOUS,
                samps_per_chan=self._wave_samples)

            # 3) load new buffer
            self.ao_task.write(self.wave, auto_start=False)

            # 4) restart in trigger order
            self.ao_task.start()
            self.ai_task.start()

        # ---------- remember the new signature ---------------------------------
        self._cur_wave_freq  = wave_freq
        self._cur_wave_rate  = samplesPerSec
        self._cur_wave_duty  = dutyCycle
        self._cur_wave_amp   = amplitude
        self._cur_wave_rtime = recordingTime

    
    def _sendSquareWaveCurrent(self, wave_freq, samplesPerSec, dutyCycle, amplitude, recordingTime):
        """
        Generate a new current-square-wave buffer on the fly,
        write it to the always-running AO task, then restore the
        original membrane-test wave.
        """
        # 1) Save the original wave buffer
        orig_wave = self.wave.copy()
        orig_rate = self._wave_rate
        orig_samples = self._wave_samples

        # 2) Build a new wave for this current pulse
        numSamples = int(samplesPerSec * recordingTime)
        period = int(samplesPerSec / wave_freq)
        onTime = int(period * dutyCycle)
        curr_wave = np.zeros(numSamples, dtype=float)
        for i in range(0, numSamples, period):
            curr_wave[i:i+onTime] = amplitude
        # temporarily override buffer
        self.wave = curr_wave
        self._wave_rate = samplesPerSec
        self._wave_samples = numSamples

        # 3) Write it to the AO task
        self.ao_task.write(self.wave, auto_start=False)

        # 4) Restore original membrane-test wave
        self.wave = orig_wave
        self._wave_rate = orig_rate
        self._wave_samples = orig_samples

    def getDataFromCurrentProtocol(self,
                                   custom: bool = False,
                                   factor: float = None,
                                   startCurrentPicoAmp: float = None,
                                   endCurrentPicoAmp: float = None,
                                   stepCurrentPicoAmp: float = 10,
                                   highTimeMs: float = 400):
        """
        Send a series of current steps, defined by:
          - if custom=False: range ± (voltageMembraneCapacitance * factor) in 10 pA increments
          - if custom=True: from startCurrentPicoAmp to endCurrentPicoAmp in stepCurrentPicoAmp
        Each step:
          1) create & send the current‐wave
          2) read back
        Finally, restore the membrane‐test wave.
        """
        # 1) Determine pulse amplitudes (pA)
        if not custom:
            if self.voltageMembraneCapacitance is None:
                raise RuntimeError("Need voltageMembraneCapacitance from a prior V-protocol")
            factor = factor or 1.0
            span = round(self.voltageMembraneCapacitance * factor, -1)
            start = -span
            end   =  span
        else:
            if startCurrentPicoAmp is None or endCurrentPicoAmp is None:
                raise ValueError("When custom=True, startCurrentPicoAmp and endCurrentPicoAmp must be provided")
            start = startCurrentPicoAmp
            end   = endCurrentPicoAmp

        # clamp to ±200 pA
        start = max(start, -200)
        end   = min(end,   200)

        # build pulse array (always include zero)
        pulses = np.arange(start, end + stepCurrentPicoAmp, stepCurrentPicoAmp)
        if 0 not in pulses:
            pulses = np.sort(np.append(pulses, 0.))
        self.pulses = pulses
        self.pulseRange = len(pulses)

        # 2) Hold onto membrane-test buffer for later restoration
        test_wave    = self.wave.copy()
        test_rate    = self._wave_rate
        test_samples = self._wave_samples

        # 3) Prepare for acquisition
        self.isRunningProtocol     = True
        self.current_protocol_data = []

        # 4) Loop over each pulse
        for pA in pulses:
            # convert picoamp → DAQ‐units
            amp_volts = (pA * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
            freq      = 1.0 / (2 * highTimeMs * 1e-3)
            recTime   = 4 * highTimeMs * 1e-3

            with self._deviceLock:
                # send/read one pulse
                self._sendSquareWaveCurrent(freq, test_rate, 0.5, amp_volts, recTime)
                data = self._readAnalogInput(test_rate, recTime)

            # unpack
            resp = data[1] / self.C_CLAMP_VOLT_PER_VOLT
            read = data[0]
            t    = np.linspace(0, read.size / test_rate, read.size, dtype=float)
            self.current_protocol_data.append([t, resp, read])

            time.sleep(0.5)

        # 5) Restore membrane-test wave once
        self.wave         = test_wave
        self._wave_rate   = test_rate
        self._wave_samples= test_samples
        self.ao_task.write(self.wave, auto_start=False)

        self.isRunningProtocol = False
        return self.current_protocol_data, self.pulses, self.pulseRange

    def getDataFromHoldingProtocol(self):
        """
        Temporarily send zero volts (no command), read baseline,
        then resume membrane-test buffer.
        """
        # hold test buffer
        test_wave = self.wave.copy()
        test_rate = self._wave_rate
        test_samples = self._wave_samples

        self.isRunningProtocol = True
        with self._deviceLock:
            # 1) send “nothing”
            zeros = np.zeros(test_samples, dtype=float)
            self.ao_task.write(zeros, auto_start=False)

            # 2) read AI
            data = self._readAnalogInput(test_rate, test_samples / test_rate)

            # 3) restore test wave
            self.wave = test_wave
            self._wave_rate = test_rate
            self._wave_samples = test_samples
            self.ao_task.write(self.wave, auto_start=False)

        self.isRunningProtocol = False

        t = np.linspace(0, data.shape[1]/test_rate, data.shape[1])
        resp = data[1]
        read = data[0]
        self.holding_protocol_data = np.array([t, resp, read])
        return self.holding_protocol_data

    def getDataFromVoltageProtocol(self):
        """
        Sends one fresh square wave (voltage-step), reads back,
        then re-loads the continuous membrane-test wave.
        Retries up to 5× on zero capacitance.
        """
        max_attempts = 5
        attempts = 0
        # keep the membrane-test buffer on hand
        test_wave = self.wave.copy()
        test_rate = self._wave_rate
        test_samples = self._wave_samples

        try:
            self.isRunningProtocol = True
            while attempts < max_attempts:
                attempts += 1

                # 1) generate the new protocol wave
                self.createSquareWave(
                    wave_freq=40,
                    samplesPerSec=100000,
                    dutyCycle=0.5,
                    amplitude=0.5,
                    recordingTime=0.025
                )

                # 2) send & read via base-class wrapper
                result = self.getDataFromSquareWave(
                    wave_freq=40,
                    samplesPerSec=100000,
                    dutyCycle=0.5,
                    amplitude=0.5,
                    recordingTime=0.025
                )
                if result is None:
                    continue

                (self.voltage_protocol_data, self.voltage_command_data,
                 self.voltageTotalResistance, self.voltageMembraneResistance,
                 self.voltageAccessResistance, self.voltageMembraneCapacitance) = result

                # 3) restore membrane-test buffer
                self.wave = test_wave
                self._wave_rate = test_rate
                self._wave_samples = test_samples

                if self.voltageMembraneCapacitance != 0:
                    break
                else:
                    self.warning(f"Attempt {attempts}: Capacitance zero → retry")
        except Exception as e:
            self.error(f"Voltage protocol error: {e}")
            self.voltageMembraneCapacitance = None
        finally:
            self.isRunningProtocol = False

        return self.voltageMembraneCapacitance


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
