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
        self._pause_evt = daq._pause_evt     # set == allowed to run
        self._idle_evt  = daq._idle_evt      # set == thread is parked
        self.wave_freq = wave_freq
        self.samplesPerSec = samplesPerSec
        self.dutyCycle = dutyCycle
        self.amplitude = amplitude
        self.recordingTime = recordingTime
        self.callback = callback
        self.interval = interval if interval is not None else recordingTime
        self.running = True
        self._last_data_queue = collections.deque(maxlen=1)
        self._fail_streak = 0           # consecutive bad-fit counter
        self._fail_limit  = 3           # restart after 3 misses

    def run(self):
        """
        Continuous acquisition loop that

        • honours pause / resume,
        • never feeds the GUI None values (uses NaN on bad fits),
        • restarts the DAQ after three consecutive bad fits.
        """
        self._fail_streak = 0            # counts consecutive bad fits
        self._fail_limit  = 3            # restart threshold

        while self.running:
            # ---------- wait until resumed ---------------------------------
            self._pause_evt.wait()
            if not self.running:
                break

            cycle_start = time.perf_counter()

            try:
                # ---------- one square-wave cycle --------------------------
                data = self.daq.getDataFromSquareWave(
                    self.wave_freq, self.samplesPerSec,
                    self.dutyCycle, self.amplitude,
                    self.recordingTime
                )
                if not data:                 # empty FIFO read → retry
                    continue

                (t_r, respData), (_, readData), totalR, memR, accR, memC = data
                fit_ok = (totalR is not None)

                # ---------- handle failed fit ------------------------------
                if not fit_ok:
                    self._fail_streak += 1
                    totalR = accR = memR = memC = np.nan   # GUI-safe NaNs

                    if self._fail_streak >= self._fail_limit:
                        with self.daq._deviceLock:
                            self.daq._restartAcquisition()
                        self._fail_streak = 0
                else:
                    self._fail_streak = 0     # reset on success

                # ---------- enqueue for GUI / callback ---------------------
                packet = {
                    "timeData":            t_r,
                    "respData":            respData,
                    "readData":            readData,
                    "totalResistance":     totalR,
                    "membraneResistance":  memR,
                    "accessResistance":    accR,
                    "membraneCapacitance": memC
                }
                self._last_data_queue.append(packet)

                # update live attributes even if NaN
                self.daq.totalResistance           = totalR
                self.daq.latestMembraneResistance  = memR
                self.daq.latestAccessResistance    = accR
                self.daq.latestMembraneCapacitance = memC

                if self.callback:
                    self.callback(totalR, accR, memR, memC,
                                respData, readData)

            except nidaqmx.errors.DaqError as e:
                # common FIFO/buffer overruns
                self.daq.warning(f"Acquisition error {e.error_code}: {e}")
                if e.error_code in (-200279, -200284, -200286):
                    with self.daq._deviceLock:
                        self.daq._restartAcquisition()
                time.sleep(0.05)            # brief cooldown

            except Exception as e:
                logging.warning(f"Error in DAQAcquisitionThread: {e}")

            # ---------- honour requested interval --------------------------
            elapsed       = time.perf_counter() - cycle_start
            time.sleep(max(0.0, self.interval - elapsed))

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
        self._pause_evt = threading.Event(); self._pause_evt.set()   # allow run
        self._idle_evt  = threading.Event(); self._idle_evt.set()    # initially idle


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
            
    def pause_acquisition(self, timeout: float = 2.0):
        """
        Block until the acquisition thread is parked **and** both DAQmx tasks
        have been stopped, giving the caller sole ownership of the hardware.
        """
        if self._pause_evt.is_set():
            self._pause_evt.clear()                 # ask thread to park
            if not self._idle_evt.wait(timeout):
                raise TimeoutError("DAQ thread failed to become idle")

            # --- stop the streaming tasks ---------------------------------
            try:
                if hasattr(self, "ai_task"): self.ai_task.stop()
                if hasattr(self, "ao_task"): self.ao_task.stop()
            except Exception:
                pass                                # tasks may already be stopped

    def resume_acquisition(self):
        """
        Restart the continuous AI+AO pair (freshly synced) and wake the thread.
        """
        if not self._pause_evt.is_set():
            # ensure tasks are running and synchronized
            self._restartAcquisition()              # subclass implementation
            self._pause_evt.set()                   # let the thread continue
    
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


    def createSquareWave(self, wave_freq, samplesPerSec,
                        dutyCycle, amplitude, recordingTime,
                        pre_pad_ms: float = 0.5,*, store: bool = True): 
        
        """
        Build a square-wave buffer.

        Returns
        -------
        (wave: np.ndarray, rate: int, numSamples: int)

        Setting *store=False* makes the routine **side-effect–free**,
        letting protocol methods generate *temporary* buffers without
        clobbering the continuous-acquisition one.

        If *store=True* (default) the buffer is cached exactly as before
        on the instance (`self.wave`, `_wave_rate`, `_wave_samples`) so
        all existing callers continue to work unchanged.
        """     
        numSamples = int(samplesPerSec * recordingTime)
        period     = int(samplesPerSec / wave_freq)
        onTime     = int(period * dutyCycle)

        wave = np.zeros(numSamples, dtype=float)
        pad  = int(pre_pad_ms * 1e-3 * samplesPerSec)

        for i in range(0, numSamples - pad, period):
            wave[i + pad : i + pad + onTime] = amplitude

        if store:
            self.wave          = wave
            self._wave_rate    = samplesPerSec
            self._wave_samples = numSamples

        return wave, samplesPerSec, numSamples

    def createSquareWaveCurrent(self,
                                wave_freq,          # test-pulse frequency (Hz)
                                samplesPerSec,
                                dutyCycle,
                                amplitude,          # test-pulse amplitude (Volts)
                                recordingTime):
        """
        Build the composite I-clamp buffer required by the legacy protocol.

        Layout per pulse
        ----------------
            • baseline   :  @ 0 pA,   1 Hz square-wave
            • calibrator :  @ –20 pA, freq = wave_freq
            • test pulse :  @ ±pA,    freq = wave_freq

        All segments share the same sampling rate and 50 % duty-cycle.

        Returns
        -------
        wave : np.ndarray  (Volts)
        rate : int         (samples / sec)
        numSamples : int   (wave.size)
        """
        import numpy as np

        # ───── 1. baseline : 0 pA, 1 Hz ──────────────────────────────────
        base_wave, *_ = self.createSquareWave(
            wave_freq=1, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=0.0,
            recordingTime=recordingTime, pre_pad_ms=0, store=False)

        # ───── 2. calibrator : –20 pA, double test-freq ──────────────────
        calib_wave, *_ = self.createSquareWave(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle,
            amplitude=(-20e-12) / self.C_CLAMP_AMP_PER_VOLT,  # pA → Volts
            recordingTime=recordingTime, pre_pad_ms=0, store=False)

        # ───── 3. test pulse : ±pA, wave_freq ────────────────────────────
        test_wave, rate, _ = self.createSquareWave(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=amplitude,
            recordingTime=recordingTime, pre_pad_ms=0, store=False)
        
        #  ───────────────── 4. pad the base wave ───────────────────────────
        # add the base wave at the end

        # ───── concatenate and return ────────────────────────────────────
        wave = np.concatenate((base_wave, calib_wave, test_wave,base_wave))
        return wave, rate, wave.size


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

        left, right = 100,300
        idx0 = max(0, max_i-left)
        idx1 = min(N, min_i+right)
        td = timeData[idx0:idx1] - timeData[idx0]
        rd = resp[idx0:idx1] * self.V_CLAMP_VOLT_PER_AMP
        cd = read[idx0:idx1] * self.V_CLAMP_VOLT_PER_VOLT

        if self.getCellMode():
            accR, memR, memC = self._getParamsfromCurrent(
                cd, rd, td, amplitude * self.V_CLAMP_VOLT_PER_VOLT
            )
            totalR = None
            if accR is not None and memR is not None:
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


    def optimizer(self, fit_data, I_peak_pA, I_peak_time_ms, I_ss_pA):
        """
        Robust mono-exponential fit:

            I(t_ms) = m · exp(−t · t_ms) + b

        with parameter
            t = 1 / τ      [1 / ms]

        Returns
        -------
        (m, t, b) in pA, 1/ms, pA   – or (None, None, None) on failure
        """
        # -------- data -----------------------------------------------------
        x_ms = fit_data['T_ms']
        y_pA = fit_data['Y_pA']

        if not (np.isfinite(x_ms).all() and np.isfinite(y_pA).all()):
            return None, None, None

        # scale only the current to nA  (better conditioning)
        y_nA = y_pA * 1e-3

        # -------- initial guess -------------------------------------------
        m0 = I_peak_pA * 1e-3           # nA
        b0 = I_ss_pA   * 1e-3           # nA
        t0 = max(1.0 / max(I_peak_time_ms, 0.1), 0.01)   # 1/ms

        p0 = [m0, t0, b0]

        # -------- bounds ---------------------------------------------------
        # Current:   ±10 nA   (±10 000 pA)
        # Time const: t = 1/τ  → τ between 0.05 ms and 100 ms
        bounds = ([-10.0,     1/100.0, -10.0],       # lower  (nA, 1/ms, nA)
                [ 10.0,     1/0.05,  10.0])        # upper

        # -------- residual & Jacobian -------------------------------------
        def residual(p, x, y):
            return self.monoExp(x, *p) - y

        def jacobian(p, x, y):
            m, t, b = p
            e = np.exp(-t * x)
            return np.vstack((e,               # ∂/∂m
                            -m * x * e,      # ∂/∂t
                            np.ones_like(x)  # ∂/∂b
                            )).T

        # -------- bounded robust least-squares -----------------------------
        try:
            res = scipy.optimize.least_squares(
                residual, p0, jac=jacobian, args=(x_ms, y_nA),
                bounds=bounds,
                loss="soft_l1",
                f_scale=1.0,          # keeps internal (f / f_scale)**2 in range
                max_nfev=400
            )
        except Exception:
            return None, None, None

        if not (res.success and np.all(np.isfinite(res.x))):
            return None, None, None

        m_fit_nA, t_fit, b_fit_nA = res.x
        return m_fit_nA * 1e3, t_fit, b_fit_nA * 1e3   # back to pA


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
            wave_freq=40, samplesPerSec=50000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025
        )

        # 2) Configure & start persistent AO/AI tasks
        self._setupAcquisition()

        # 3) Kick off the acquisition thread
        self.start_acquisition(
            wave_freq=40, samplesPerSec=50000,
            dutyCycle=0.5, amplitude=0.5,
            recordingTime=0.025,
            interval= 0.000  # 0 ms per cycle
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
        import nidaqmx.constants as c

        # Allow the device to overwrite unread data so the fifo keeps running
        self.ai_task.in_stream.overwrite_mode = c.OverwriteMode.OVERWRITE_UNREAD_SAMPLES

        # When we call task.read(), start relative to the newest sample
        self.ai_task.in_stream.read_relative_to = c.ReadRelativeTo.MOST_RECENT_SAMPLE
        # …and back-up exactly one buffer-length so we still get a full cycle
        self.ai_task.in_stream.read_offset = -self._wave_samples

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

    def _setupAcquisitionCurrent(self, recordingTime=500):
        """
        Configure a dedicated continuous AI + AO pair for the
        current-step protocol.  The background stream must be paused first.
        """
        import nidaqmx, nidaqmx.constants as c

        samplesPerSec = 100_000
        dutyCycle     = 0.5
        recordingTime = (float(recordingTime) * 1e-3)  # convert ms to seconds
        wave_freq     = 1.0 / (recordingTime)  # 1 Hz square wave

        # ------------------------------------------------------------------
        # 1. prime buffer (0 pA test pulse)
        # ------------------------------------------------------------------
        wave, rate, numSamples = self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=0.0,
            recordingTime=recordingTime)
        # ------------------------------------------------------------------
        # 2. tasks (AI owns clock; AO waits on AI start trigger)
        # ------------------------------------------------------------------
        self.ai_task = nidaqmx.Task()
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.readDev}/{self.readChannel}",
            terminal_config=c.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.ai_task.ai_channels.add_ai_voltage_chan(
            f"{self.respDev}/{self.respChannel}",
            terminal_config=c.TerminalConfiguration.DIFF, min_val=-10.0, max_val=10.0)
        self.ai_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS)
        self.ai_task.in_stream.input_buf_size = max(
            self.ai_task.in_stream.input_buf_size, rate * 10)

        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(f"{self.cmdDev}/{self.cmdChannel}")
        self.ao_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=c.AcquisitionType.CONTINUOUS,
            samps_per_chan=numSamples)
        self.ao_task.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION

        start_trig_term = (f"/{self.readDev.split('Mod')[0] if 'Mod' in self.readDev else self.readDev}"
                           "/ai/StartTrigger")
        self.ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            start_trig_term, trigger_edge=c.Edge.RISING)

        # prime AO FIFO, then start in LabVIEW order (AO-armed → AI-fires)
        self.ao_task.write(wave, auto_start=False)
        self.ao_task.start()
        self.ai_task.start()

    def _readAnalogInput(self, samplesPerSec, recordingTime):
        """
        Wrapper for the AI task: read exactly one bufferful.
        """
        numSamples = int(samplesPerSec * recordingTime)
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

        import nidaqmx.constants as c
        # 1) create the current square wave
        wave, rate, numSamples = self.createSquareWaveCurrent(
            wave_freq=wave_freq, samplesPerSec=samplesPerSec,
            dutyCycle=dutyCycle, amplitude=amplitude,
            recordingTime=recordingTime
        )   

        # 2) write the new wave to the AO task
        self.ao_task.write(wave, auto_start=False)

    def getDataFromCurrentProtocol(self,
                                   custom: bool = False,
                                   factor: float | None = None,
                                   startCurrentPicoAmp: float | None = None,
                                   endCurrentPicoAmp: float | None = None,
                                   stepCurrentPicoAmp: float = 10,
                                   recordingTimeMs: float = 500):
        """
        Run an entire I-clamp step protocol on one continuous stream.
        Returns
        -------
        (list_of_traces, pulses_array, pulse_count)
        """
        # ───────── constants ─────────
        samplesPerSec  = 100_000
        dutyCycle      = 0.5
        recordingTime  = recordingTimeMs * 1e-3   # 0.25 s / segment
        wave_freq      = 1.0 / recordingTime      # 4 Hz
        fullRecTime    = 4 * recordingTime        # 1.0 s / train
        exp_samples    = int(samplesPerSec * fullRecTime)

        # ─ 1. build pulse list ───────
        if not custom:
            if self.voltageMembraneCapacitance is None:
                raise RuntimeError("Run getDataFromVoltageProtocol() first.")
            span = round(self.voltageMembraneCapacitance * (factor or 1.0), -1)
            startCurrentPicoAmp, endCurrentPicoAmp = -span, span

        startCurrentPicoAmp = max(startCurrentPicoAmp, -200)
        endCurrentPicoAmp   = min(endCurrentPicoAmp,   200)

        pulses = np.arange(startCurrentPicoAmp,
                           endCurrentPicoAmp + stepCurrentPicoAmp,
                           stepCurrentPicoAmp, dtype=float)
        if 0.0 not in pulses:
            pulses = np.sort(np.append(pulses, 0.0))

        self.pulses       = pulses
        self.pulseRange   = len(pulses)
        self.current_protocol_data = []
        self.info(f"I-clamp pulses (pA): {pulses}")

        # ─ 2. park background stream ─
        self.pause_acquisition()
        for t in ("ai_task", "ao_task"):
            try:
                getattr(self, t).stop(); getattr(self, t).close()
            except Exception:
                pass

        # ─ 3. start dedicated tasks (prime buffer = 0 pA test) ──────────
        self._setupAcquisitionCurrent(recordingTime * 1e3)   # expects ms

        # a) queue FIRST real pulse so it will become train #2
        next_amp_V = (pulses[0] * 1e-12) / self.C_CLAMP_AMP_PER_VOLT
        self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                    dutyCycle, next_amp_V, recordingTime)

        # b) wait 1 s and discard train #1 (0 pA prime)
        while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
            self.sleep(0.002)
        _ = self.ai_task.read(exp_samples, timeout=2.0)

        try:
            for idx, pulse in enumerate(pulses):
                # ─── queue the NEXT pulse *before* waiting ───────────────
                if idx + 1 < len(pulses):
                    future_amp_V = (pulses[idx + 1] * 1e-12) \
                                    / self.C_CLAMP_AMP_PER_VOLT
                    self._sendSquareWaveCurrent(wave_freq, samplesPerSec,
                                                dutyCycle, future_amp_V,
                                                recordingTime)

                self.info(f"Waiting for {pulse:.0f} pA train…")

                # ─── wait until exactly one full train is ready ─────────
                while self.ai_task.in_stream.avail_samp_per_chan < exp_samples:
                    self.sleep(0.002)

                # ─── read it (matches current pulse) ────────────────────
                raw = np.asarray(self.ai_task.read(exp_samples, timeout=2.0),
                                 dtype=float)
                resp = raw[1] / self.C_CLAMP_VOLT_PER_VOLT
                read = raw[0]
                tvec = np.linspace(0, fullRecTime, exp_samples, dtype=float)
                self.current_protocol_data.append([tvec, resp, read])

                self.info(f"Captured {pulse:.0f} pA pulse")

        finally:
            # ─ 4. restore background stream ────────────────────────────
            try:
                self.ai_task.stop();  self.ao_task.stop()
                self.ai_task.close(); self.ao_task.close()
            except Exception:
                pass
            self.resume_acquisition()

        return self.current_protocol_data, self.pulses, self.pulseRange

    def getDataFromHoldingProtocol(self, *, rate_hz: int = 50_000, duration_s: float = 1.0):
        """
        OLD-STYLE baseline read-out (no command output).

        • Suspends the membrane-test thread.
        • Builds one finite AI-only task at 50 kS/s for 1 s
          (same numbers as the legacy code).
        • Does *not* touch the AO task, avoiding the –200547 write error.
        • Restarts the continuous stream afterwards.

        Returns
        -------
        np.ndarray
            [time_s, resp_V, read_V]  – identical to the legacy layout.
        """
        import nidaqmx
        import nidaqmx.constants as c
        import numpy as np

        # -------------------------------------------------- 1. park background
        self.pause_acquisition()

        try:
            num_samples = int(rate_hz * duration_s)

            # ---------------------------------------------- 2. finite AI task
            ai = nidaqmx.Task()
            ai.ai_channels.add_ai_voltage_chan(
                f"{self.readDev}/{self.readChannel}",
                terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0)
            ai.ai_channels.add_ai_voltage_chan(
                f"{self.respDev}/{self.respChannel}",
                terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0)
            ai.timing.cfg_samp_clk_timing(
                rate=rate_hz,
                sample_mode=c.AcquisitionType.FINITE,
                samps_per_chan=num_samples)

            ai.start()
            raw = ai.read(
                number_of_samples_per_channel=num_samples,
                timeout=duration_s + 2.0)
            ai.stop(); ai.close()

            raw = np.array(raw, dtype=float)
            resp = raw[1]      # still in DAQ volts
            read = raw[0]
            t = np.linspace(0, duration_s, num_samples, dtype=float)

            self.holding_protocol_data = np.array([t, resp, read])
            return self.holding_protocol_data

        finally:
            # ---------------------------------------------- 3. resume stream
            self.resume_acquisition()


    def getDataFromVoltageProtocol(
            self,
            *,                       # keyword-only
            wave_freq: int = 40,
            samplesPerSec: int = 50_000,   # 50 kS/s
            dutyCycle: float = 0.5,
            amplitude: float = 0.5,
            recordingTime: float = 0.025,  # 25 ms period
            max_attempts: int = 15
    ):
        """
        • Pauses the DAQAcquisitionThread (tasks keep streaming).
        • Collects **five acceptable** traces (≤ `max_attempts` cycles).
            – First acceptable trace populates the usual voltage-protocol
              attributes (`voltage_protocol_data`, Rₐ, Rₘ, …).
            – All five capacitances are averaged → `voltageMembraneCapacitance`.
            – All five baseline currents (I_prev_pA) averaged → `holding_current_avg`.
        • Resumes the acquisition thread.

        Returns
        -------
        float | None
            Averaged capacitance (pF), or None if no acceptable trace was found.
        """
        import math, time, numpy as np

        # ─── 1. Pause the worker thread (leave AI / AO running) ─────────
        self._pause_evt.clear()
        time.sleep(recordingTime * 2)        # let current cycle pass

        cm_values      = []      # acceptable capacitances
        i_prev_values  = []      # acceptable baseline currents
        first_saved    = False
        attempts       = 0

        try:
            while attempts < max_attempts and len(cm_values) < 5:
                attempts += 1

                # ─── 2. Grab one square-wave cycle on the live stream ──
                result = self.getDataFromSquareWave(
                    wave_freq, samplesPerSec, dutyCycle,
                    amplitude, recordingTime
                )

                (prot_data, cmd_data,
                 totalR, memR, accR, memC) = result

                # _getParamsfromCurrent() has just updated self.holding_current
                i_prev_pA = getattr(self, "holding_current", None)

                # ─── 3. Acceptability check ────────────────────────────
                good_cm   = memC is not None and math.isfinite(memC) and memC > 0
                self.info(f'cm: {memC:.2f} pF, ')
                good_i    = i_prev_pA is not None and math.isfinite(i_prev_pA)
                self.info(f'i_prev: {i_prev_pA:.2f} pA, ')

                if good_cm and good_i:
                    # store the first good trace exactly as before
                    if not first_saved:
                        (self.voltage_protocol_data,
                         self.voltage_command_data,
                         self.voltageTotalResistance,
                         self.voltageMembraneResistance,
                         self.voltageAccessResistance,
                         self.voltageMembraneCapacitance) = result
                        first_saved = True

                    cm_values.append(memC)
                    i_prev_values.append(i_prev_pA)
                else:
                    self.warning(f"Voltage step attempt {attempts}: "
                                 "unacceptable fit → skipped")

                # wait one extra period so traces do not overlap in FIFO
                time.sleep(recordingTime)

            if not cm_values:
                # no acceptable trace at all
                self.error("Voltage protocol: no acceptable trace found")
                self.holding_current_avg         = None
                self.voltageMembraneCapacitance  = None
                return None

            # ─── 4. Average capacitance & baseline current ─────────────
            self.voltageMembraneCapacitance = float(np.mean(cm_values))
            self.holding_current_avg        = float(np.mean(i_prev_values))
            self.info(f"Averages: cm: {self.voltageMembraneCapacitance:.2f} pF, ")
            self.info(f"i_prev: {self.holding_current_avg:.2f} pA, ")

            if len(cm_values) < 5:
                self.warning(f"Voltage protocol collected only {len(cm_values)} "
                             f"good traces (expected 5) within {attempts} attempts")

        finally:
            # ─── 5. Resume background acquisition ──────────────────────
            self._pause_evt.set()

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
