
import time
import csv
import numpy as np
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import NiDAQ
from holypipette.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.pressurecontroller import PressureController
import collections
import logging

from holypipette.interface.patchConfig import PatchConfig

from .base import TaskController
import threading
# import locking package
from threading import Lock


class AutopatchError(Exception):
    def __init__(self, message = 'Automatic patching error'):
        self.message = message

    def __str__(self):
        return self.message


class AutoPatcher(TaskController):
    def __init__(self, amplifier: Amplifier, daq: NiDAQ, pressure: PressureController, calibrated_unit: CalibratedUnit, microscope: Microscope, calibrated_stage: CalibratedStage, config : PatchConfig):
        super().__init__()
        self.config = config
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.calibrated_unit = calibrated_unit
        self.calibrated_stage = calibrated_stage
        self.microscope = microscope
        self.safe_position = None
        self.safe_stage_position = None
        self.home_position = None
        self.home_stage_position = None
        self.cleaning_bath_position = None
        self.rinsing_bath_position = None
        self.contact_position = None
        self.initial_resistance = None
        self.vholding = None
        self.iholding = None
        self.rig_ready = False
        self.first_res = None
        self.atm = False


        self.current_protocol_graph = None
        
    def state_emitter(self, state):
        self.info(f"emitting state: {state}")

    def getHolding(self):
        """Get the holding current as measured by the DAQ."""
        self.amplifier.voltage_clamp()
        self.sleep(1)
        self.amplifier.switch_holding(False) 
        self.sleep(1)
        base1a = self.daq.holding_current
        self.sleep(1)
        base1b = self.daq.holding_current
        if base1a and base1b is not None:
            base1 = float((base1a + base1b) / 2)
        else: 
            base1 = None
        self.amplifier.switch_holding(True)
        self.sleep(1)
        base2a = self.daq.holding_current
        self.sleep(1)
        base2b = self.daq.holding_current
        # average base2a and base2b
        if base2a and base2b is not None:
            base2 = float((base2a + base2b) / 2)
        else:
            base2 = None
        if base1 is None or base2 is None:
            self.info("Holding current not set, using default value")
            return -50
        else:
            holding_current = (base2 - base1) 
            # self.info(f"Base1: {base1}, Base2: {base2}")
            self.info(f"Holding current: {holding_current} pA")
            if abs(holding_current) > 150:
                self.info("Holding current is too high, setting to default value of -50 pA")
                holding_current = -50
            return holding_current

    

    def run_protocols(self):

        self.daq.setCellMode(True)
        holding = self.getHolding()

        if self.config.voltage_protocol:
            self.run_voltage_protocol()
            self.sleep(0.25)
        if self.config.current_protocol:
            self.daq.setCellMode(False)
            self.iholding = holding
            self.run_current_protocol()
            self.sleep(0.25)
            self.daq.setCellMode(True)
        if self.config.holding_protocol:
            self.run_holding_protocol()
        # self.done = True

    def run_voltage_protocol(self):
        self.info('Running voltage protocol (membrane test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        holding = self.amplifier.get_holding()
        if holding is None:
            holding = -0.070
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        self.sleep(0.25)
        self.amplifier.switch_holding(True)
        self.info('enabled holding')
        self.sleep(0.25)
        self.info("Getting data from voltage protocol")
        self.daq.getDataFromVoltageProtocol()
        self.sleep(0.25)
        self.info('finished running voltage protocol (membrane test)')

    def run_current_protocol(self):
        self.info('Running current protocol (current clamp)')
        self.amplifier.voltage_clamp()
        self.sleep(0.1)
        self.amplifier.auto_fast_compensation()
        self.sleep(0.25)
        self.amplifier.auto_slow_compensation()
        self.sleep(0.25)
        self.info('auto capacitance compensation')  
        cap_c_double = self.amplifier.get_fast_compensation_capacitance()
        cap = float(cap_c_double.value) * 1e12 - 0.5
        cap = cap*1e-12
        self.info(f'fast compensation capacitance: {cap} pF' )
        self.sleep(0.1)
        self.amplifier.current_clamp()
        self.sleep(0.1)
        self.amplifier.set_neutralization_capacitance(cap)
        self.info('set neutralization capacitance')
        self.amplifier.set_neutralization_enable(True)
        self.info('enabled neutralization')
        self.sleep(0.1)
        self.amplifier.set_bridge_balance(True)
        self.info('auto bridge balance')
        self.amplifier.auto_bridge_balance()
        self.sleep(0.1)
        if self.iholding is None:
            current = -50

        else:
            current = (self.iholding)

        current = current * 1e-12
        self.amplifier.set_holding(current)
        self.info(f'holding at {current} pA')
        self.sleep(0.1)
        self.amplifier.switch_holding(True)
        self.info('enabled holding')
        self.sleep(0.1)
        if self.config.custom_cclamp_protocol:
            self.debug('running custom current protocol')
            self.daq.getDataFromCurrentProtocol(custom =self.config.custom_cclamp_protocol,factor= 1,startCurrentPicoAmp=(self.config.cclamp_start), endCurrentPicoAmp=(self.config.cclamp_end), stepCurrentPicoAmp=(self.config.cclamp_step), recordingTimeMs = 500)                                            
        else:
            self.debug('running default current protocol')
            self.daq.getDataFromCurrentProtocol(custom=self.config.custom_cclamp_protocol, factor=1, startCurrentPicoAmp=None, endCurrentPicoAmp=None, stepCurrentPicoAmp=10, recordingTimeMs = 500)
        self.sleep(0.1)
        self.amplifier.switch_holding(False)
        self.info('disabled holding')
        self.sleep(0.1)
        self.amplifier.voltage_clamp()
        self.info('finished running current protocol(current clamp)')

    def run_holding_protocol(self):
        self.info('Running holding protocol (E/I PSC test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        holding = self.amplifier.get_holding()
        if holding is None:
            holding = -0.070
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        self.sleep(0.25)
        self.daq.getDataFromHoldingProtocol(duration_s = self.config.hclamp_duration)
        self.sleep(0.25)
        # self.amplifier.set_holding(0)
        self.sleep(0.25)
        self.amplifier.voltage_clamp()
        self.info('finished running holding protocol (E/I PSC test)')
    
    def isrigready(self):
        try:
            if not self.calibrated_unit.calibrated:
                # # testing scenario
                # self.calibrated_unit.calibrated = True
                # print("Pipette calibrated for testing")
                raise AutopatchError("Pipette not calibrated")
            if not self.calibrated_stage.calibrated:
                raise AutopatchError("Stage not calibrated")
            if self.safe_position is None:
                raise ValueError('Safe position has not been set')
            if self.home_position is None:
                raise ValueError('Home position has not been set')
            if self.cleaning_bath_position is None:
                raise ValueError('Cleaning bath position has not been set')
            if self.microscope.floor_Z is None:
                raise AutopatchError("Cell Plane not set")
            self.rig_ready = True
        except (AutopatchError, ValueError) as e:
            self.rig_ready = False
            raise e
        
    def locate_cell(self, cell):
        '''
        Performs regional pipette localization to bring pipette above the cell.
        '''
         # regional pipette localization: 
        # move stage and pipette to safe space
        self.info("Moving to safe space")
        self.move_to_safe_space()
        self.info("Setting pressure to 100 mbar")
        self.pressure.set_pressure(100)
        # move to home space
        self.info("Moving to home space")
        self.move_to_home_space()
        # center pipette on cell xy 
        self.info("Centering pipette")
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        
        # move to cell position 
        cell_pos, cell_img = cell
        if self.config.cell_type == "Plate":
            self.config.cell_distance = 20
        elif self.config.cell_type == "Slice":
            self.config.cell_distance = 75
            

        self.info(f" Moving to Cell position: {cell_pos}") 
        # moving stage to xy position of cell
        # home position
        cell_pos_planar = np.array([cell_pos[0], cell_pos[1], 0])
        self.calibrated_stage.safe_move(np.array(cell_pos_planar))
        self.calibrated_stage.wait_until_still()
        # move pipette to xy position of cell
        stage_pos = self.calibrated_stage.pixels_to_um(self.calibrated_stage.reference_position())
        # print(f"Stage position: {stage_pos}")
        disp = np.zeros(3)
        disp[0] = stage_pos[0] - self.home_stage_position[0]
        disp[1] = stage_pos[1] - self.home_stage_position[1]
        disp[2] = 0
        # print(f"Disp: {disp}")
        pipette_disp = self.calibrated_unit.rotate(disp,2)
        self.calibrated_unit.relative_move(pipette_disp)
        self.calibrated_unit.wait_until_still() 
        # center pipette on cell xy 
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        zdist_cell = self.home_stage_position[2] - cell_pos[2]
        self.move_group_down(-zdist_cell/2)# on real rig
        self.sleep(0.1)
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        second = zdist_cell/2 + self.config.cell_distance
        self.move_group_down(-second)
        self.sleep(0.1)
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.microscope.move_to_floor()
        self.microscope.wait_until_still()
        z_pos = self.microscope.position()/5.0
        zdistleft  = z_pos - cell_pos[2]
        self.microscope.relative_move(-zdistleft)
        self.microscope.wait_until_still()
        if self.config.cell_type == "Plate": 
            self.info("centering on cell")
            self.calibrated_stage.center_on_cell(cell)
            self.calibrated_stage.wait_until_still()
            self.info(f"correcting pipette position, moving microscope by {zdistleft} um")
            self.microscope.relative_move(-self.config.cell_distance)
            self.microscope.wait_until_still()
            self.calibrated_unit.center_pipette()
            self.calibrated_unit.wait_until_still()
            self.microscope.relative_move(self.config.cell_distance)
            self.microscope.wait_until_still()
        self.info("Located Cell")

        self.amplifier.start_patch()
        self.sleep(0.1)


    def hunt_cell(self,cell = None):
        '''
        Moves the pipette down to cell plane and detects a cell using resistance measurements
        '''
        self.info("Hunting for cell")
        
        self.isrigready()

        if self.rig_ready == False:
            raise AutopatchError("Rig not ready for cell hunting")
        
        if cell is None:
            raise AutopatchError("No cell given to patch!")
        
        # # #ensure "near cell" pressure
        self.info(f"Setting pressure to {self.config.pressure_near} mbar")
        self.pressure.set_pressure(self.config.pressure_near)
        self.sleep(3) #let the resistance stabilize

        lastResDeque = collections.deque(maxlen=5)
        # get initial resistance
        daqResistance = self.daq.resistance()
        lastResDeque.append(daqResistance)

        # move pipette down at 10um/s and check resistance every 40 ms
        # get starting position 
        start_pos = self.calibrated_unit.position()

        self.first_res = self.resistanceRamp()
        self.info(f"Initial resistance: {self.first_res}")
        self.info("starting descent....")

        if self.config.mode == 'Classic':
            autoHunt = True
        else:
            autoHunt = False

        self.info(f"{self.config.mode}: starting hunt")

        if autoHunt:
            self.calibrated_unit.absolute_move_group_velocity([0, 0, -10])

        if self.config.cell_type == "Plate":
            self.config.cell_R_increase = 0.300
        elif self.config.cell_type == "Slice":
            self.config.cell_R_increase = 0.200

        while not self._isCellDetected(lastResDeque=lastResDeque,cellThreshold = self.config.cell_R_increase) and self.abort_requested == False:
            curr_pos = self.calibrated_unit.position()
            if abs(curr_pos[2] - start_pos[2]) >= (int(self.config.max_distance)):
                # we have moved expected um down and still no cell detected
                self.info("cell not detected")
                if autoHunt:
                    self.calibrated_unit.stop()
                    self.escape()
                break
            elif self._isCellDetected(lastResDeque=lastResDeque,cellThreshold=self.config.cell_R_increase):
                if autoHunt:
                    self.calibrated_unit.stop()
                self.info("Cell Detected")
                break
            #TODO will add another condition to check if cell and pipette have moved away from each other based on the mask and original image.
            self.sleep(0.04)
            lastResDeque.append(daqResistance)
            daqResistance = self.daq.resistance()

        self.calibrated_unit.stop()
        self.microscope.stop()

    def escape(self):
            self.amplifier.stop_patch()
            self.calibrated_unit.stop()
            self.microscope.stop()
            self.pressure.set_pressure(50)
            self.pressure.set_ATM(atm=False)
            self.daq.setCellMode(False)
            self.sleep(1)
            self.pressure.set_pressure(100)
            self.sleep(1)
            self.move_group_up(20)
            self.sleep(1)
            self.pressure.set_pressure(200)
            self.sleep(1)
            self.move_to_home_space()
            self.clean_pipette()
            self.sleep(1)
            self.move_to_safe_space()
            self.sleep(5)
            self.microscope.move_to_floor()
    
    def _safe_average(self, read_fn, num_measurements: int = 5, interval: float = 0.200):
        """Return the mean of *valid* samples from *read_fn*.

        * Skips any reading that is ``None`` or ``NaN``.
        * If **every** reading in a window is invalid, run ``_adjustTrace``
          **once per window** and retry.
        * Retries *max_windows* times (default = 3).  After that, raises.
        """
        max_windows = 3
        for attempt in range(max_windows):
            readings = []
            for _ in range(num_measurements):
                val = read_fn()
                # Guard against NaN/None without raising TypeError on None
                if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                    # Keep log format identical – use debug so existing info/print lines stay untouched
                    self.debug("_safe_average: invalid reading skipped (NaN/None)")
                else:
                    readings.append(val)
                self.sleep(interval)

            if readings:
                return sum(readings) / len(readings)

            # No usable sample → adjust & retry
            self.info("All readings invalid – running _adjustTrace and retrying (_safe_average)")
            self._adjustTrace()

        # Exhausted retries
        raise RuntimeError(f"All measurements from {read_fn.__name__} returned None/NaN after {max_windows} retries")
    
    def _adjustTrace(self):
        '''
        run capacitance compensation if fitting is failing.

        '''
        self.daq.setCellMode(False)
        self.amplifier.auto_fast_compensation()
        self.sleep(0.5)
        self.amplifier.auto_slow_compensation()
        self.sleep(0.5)
        self.daq.setCellMode(True)
        
    def accessRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.accessResistance, num_measurements, interval
        )

    def resistanceRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.resistance, num_measurements, interval
        )

    def capacitanceRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.capacitance, num_measurements, interval
        )


    def gigaseal(self):
        """requires **three consecutive**
        averaged-resistance windows ≥ target to declare success, reducing
        false positives from transient spikes.
        """
        if self.config.mode == 'Classic':
            autoPressure = True
        else:
            autoPressure = False
        self.info(f"{self.config.mode}: Attempting to form gigaseal...")
        self.amplifier.auto_fast_compensation()
        self.sleep(1)
        self.daq.setCellMode(True)
        self.sleep(0.1)
        self.info("Collecting baseline resistance...")

        avg_resistance = self.resistanceRamp()
        consecutive_success = 0

        self.pressure.set_ATM(atm=True)

        if self.config.cell_type == "Plate":
            self.config.Vramp_amplitude = -0.020
        elif self.config.cell_type == "Slice":
            self.config.Vramp_amplitude = -0.070

        self.sleep(10)

        if autoPressure:
            currPressure = -5
            self.pressure.set_pressure(currPressure)
            self.pressure.set_ATM(atm=False)
            prevpressure = currPressure
            speed = 1
            bad_cell_count = 0
            # this is already negative, e.g. -30 mbar
            max_pressure = self.config.pressure_ramp_max

        holding_switched = False
        last_progress_time = time.time()

        while not self.abort_requested:
            # Deadline check
            if time.time() - last_progress_time >= self.config.seal_deadline:
                raise AutopatchError("Seal unsuccessful: resistance did not improve significantly.")

            prev_resistance = avg_resistance
            avg_resistance = self.resistanceRamp()

            delta_resistance = avg_resistance - prev_resistance
            rate_mohm_per_sec = delta_resistance / (5 * 0.200)

            if delta_resistance >= self.config.gigaseal_min_delta_R:
                last_progress_time = time.time()

            print(f"goal resistance: {self.config.gigaseal_R} MΩ; "
                f"current resistance: {avg_resistance} MΩ; "
                f"rate: {rate_mohm_per_sec} MΩ/s")

            # ---------------------- auto-pressure logic ----------------------
            if autoPressure:
                # adjust currPressure by ±5 based on rate_mohm_per_sec, speed, etc.
                if -(self.config.gigaseal_R / 100) < rate_mohm_per_sec < self.config.gigaseal_R / 3000:
                    currPressure -= 5; speed = 3; max_pressure = -45
                elif self.config.gigaseal_R / 3000 <= rate_mohm_per_sec <= self.config.gigaseal_R / 10:
                    speed = 1  # maintain
                elif self.config.gigaseal_R / 10 < rate_mohm_per_sec <= self.config.gigaseal_R / 5:
                    max_pressure = self.config.pressure_ramp_max; currPressure += 5; speed = 3
                elif rate_mohm_per_sec <= -(self.config.gigaseal_R / 100):
                    currPressure += 5; currPressure = max(currPressure, -5); speed = 0.5

                # <<< replace single clamp with two lines to forbid positive pressure >>>
                # never above 0 mbar:
                currPressure = min(currPressure, 0.0)
                # never exceed deepest vacuum (e.g. -30 mbar):
                currPressure = max(currPressure, self.config.pressure_ramp_max)

                if currPressure != prevpressure:
                    self.pressure.set_pressure(currPressure)
                    prevpressure = currPressure
                    self.sleep(5 / speed)

                if currPressure <= max_pressure:
                    self.pressure.set_ATM(True)
                    self.sleep(5)
                    testresistance = self.resistanceRamp()
                    difference = testresistance - avg_resistance
                    print(f"Test resistance: {testresistance} MΩ; difference: {difference} MΩ")
                    if difference < 0:
                        bad_cell_count += 1
                        if bad_cell_count > 5:
                            raise AutopatchError("Bad cell detected")

                    currPressure = -5
                    self.pressure.set_pressure(currPressure)
                    self.pressure.set_ATM(atm=False)
            # ---------------------------------------------------------------

            # Holding potential switch
            if avg_resistance >= self.config.gigaseal_R / 12 and not holding_switched:
                self.amplifier.set_holding(self.config.Vramp_amplitude)
                self.amplifier.switch_holding(True)
                holding_switched = True

            # Success check with consecutive-hit filter
            if avg_resistance >= self.config.gigaseal_R:
                consecutive_success += 1
            else:
                consecutive_success = 0

            if consecutive_success >= 3:
                self.pressure.set_ATM(atm=True)
                self.info("Seal successful!")
                return

        # Abort request came in
        raise AutopatchError("Seal unsuccessful: gigaseal criteria not met.")

    def break_in(self):
        """
        Attempts whole-cell break-in.

        NEW LOGIC
        ---------
        * Measure access resistance first each loop.
        * On every sub-threshold reading, **pause** all pressure/zap activity and
        skip the slow resistance & capacitance ramps.
        * Require three consecutive good readings to confirm success.
        * The moment a reading is above threshold, reset the streak and fall back
        to the full pressure/zap/ramp cycle.
        """
        from collections import deque  # local import keeps patch minimal

        # ---------- initial setup (unchanged) ----------
        self.daq.setCellMode(True)
        autoPressure = (self.config.mode == 'Classic')
        self.info(f"{self.config.mode}: Attempting Break in...")
        self.sleep(3)
        self.pressure.set_pressure(self.config.pulse_pressure_break_in)
        self.amplifier.set_zap_duration(25 * 1e-6)

        measuredAccessResistance = self.accessRamp()
        measuredResistance       = self.resistanceRamp()
        measuredCapacitance      = self.capacitanceRamp()
        self.info(
            f"Initial Resistance: {measuredResistance}; "
            f"Initial Capacitance: {measuredCapacitance},  "
            f"Initial Access Resistance: {measuredAccessResistance}")
        self.info(
            f"Target Resistance: {self.config.max_cell_R}; "
            f"Target Capacitance: {self.config.min_cell_C}, "
            f"Target Access Resistance: {self.config.max_access_R}")

        # ---------- loop variables ----------
        trials        = 0
        speed         = 3
        good_count    = 0
        threshold_AR  = self.config.max_access_R      # adjust here if units differ

        # ---------- main loop ----------
        while True:
            # ---- 1) quick access-R check ----
            r_ax = self.accessRamp()
            self.debug(f"Access-R check: {r_ax:.2f} Ω (good_count={good_count})")

            if r_ax <= threshold_AR:
                good_count += 1
                if good_count >= 3:        # success: 3 good hits in a row
                    measuredAccessResistance = r_ax
                    break
                # pause all actions; go straight to next access-R check
                continue
            else:
                good_count = 0             # reset streak on failure

            # ---- 2) full break-in cycle (runs only after a “bad” access-R) ----
            if autoPressure:
                trials += 1
                self.debug(f"Trial: {trials}")

                speedosc = trials % 5
                if speedosc == 0:
                    speed = 2
                self.pressure.set_ATM(atm=False)
                self.sleep(1 / speed)
                self.pressure.set_ATM(atm=True)
                self.sleep(0.75)
                speed = 3

                osc = trials % 3
                if self.config.zap and osc == 0:
                    self.info("zapping")
                    self.amplifier.zap(); self.sleep(0.5)
                    self.amplifier.zap(); self.sleep(0.5)

                self.sleep(1)

            # slow ramps (only if previous access-R was “bad”)
            measuredResistance  = self.resistanceRamp()
            measuredCapacitance = self.capacitanceRamp()

            if autoPressure:
                self.info(
                    f"Trial {trials}: Running Avg Membrane Resistance: "
                    f"{measuredResistance}; Membrane Capacitance: "
                    f"{measuredCapacitance}, Access Resistance: {r_ax}")

                if trials > 15:
                    self.info("Break-in unsuccessful")
                    raise AutopatchError("Break-in unsuccessful")

        # ---------- success ----------
        self.info("Successful break-in, Running Avg Access Resistance = "
                f"{measuredAccessResistance:.2f}")

    def _isCellDetected(self, lastResDeque, cellThreshold = 0.15):
        '''Given a list of three resistance readings, do we think there is a cell where the pipette is?
        '''
        # print(lastResDeque)

        # Ensure there are five readings before checking
        if len(lastResDeque) < 5:
            return False

        # Criteria 1: the last three readings must be increasing
        # if not lastResDeque[0] < lastResDeque[1] < lastResDeque[2]:
        #     # show the last three resistances
        #     self.debug(f"Last three resistances: {lastResDeque}")
        #     return False  # Last three resistances must be ascending
        
        # Criteria 2: there must be an increase by at least the cellThreshold
        r_delta = (lastResDeque[4] - self.first_res)

        # self.info(f"Cell detected, resistance: {r_delta}")
        detected = cellThreshold <= r_delta
        if detected:
            self.info(f"Cell detected: {detected}; resistance: {r_delta}")
            self.calibrated_unit.stop()

        return cellThreshold <= r_delta

    def patch(self, cell=None):
        '''
        Runs the automatic patch-clamp algorithm, including manipulator movements.
        '''

        # ------ rig preparation -------------------------------#
        
        #check for stage and pipette calibration

        self.isrigready()
        if self.rig_ready == False:
            raise AutopatchError("Rig not ready for patching")
        
        if cell is None:
            raise AutopatchError("No cell given to patch!")

        self.info("Starting patching process")

        #! Phase 0: locate cell
        self.locate_cell(cell)
        self.sleep(3)
        #! phase 1: hunt for cell
        self.hunt_cell(cell)
        self.sleep(3)
        # move a bit further down to make sure we're at the cell
        # self.calibrated_unit.relative_move(1, axis=2)
        #! phase 2: attempt to form a gigaseal
        self.gigaseal()
        self.sleep(10)
        #! Phase 3: break into cell
        self.break_in()
        self.info("Whole cell Acheived, resting for 60 seconds")
        self.sleep(60)
        #! Phase 4: run protocols
        self.info("Running protocol 1")
        self.run_protocols()
        self.sleep(20)
        self.info("Running protocol 2")
        self.run_protocols()
        self.sleep(20)
        self.info("Running protocol 3")
        self.run_protocols()
        self.sleep(5)
        #! Phase 5: clean pipette
        self.info("Data collection complete, cleaning pipette")
        self.escape()


    def move_to_safe_space(self):
        '''
        Moves the pipette to the safe space.
        '''
        if self.safe_position is None:
            raise ValueError('Safe position has not been set')


        try:
            # Extract individual coordinates from the safe position
            safe_x, safe_y, safe_z = self.safe_position
            safe_stage_x, safe_stage_y, safe_microscope_z = self.safe_stage_position
            self.info(f"Moving to safe space: {safe_x}, {safe_y}, {safe_z}")

            # Step 0: Move the microscope to the safe position
            logging.debug(f'Moving microscope to safe position value: Z={safe_microscope_z}')
            self.microscope.absolute_move(safe_microscope_z)
            self.microscope.wait_until_still()  # Ensure movement completes
            # Step 1: Move the stage to the safe position
            self.calibrated_stage.absolute_move([safe_stage_x,safe_stage_y])
            self.calibrated_stage.wait_until_still()
            # # step 1.5: move pipette up if at cleaning position:
            # if self.cleaning_bath_position is not None and self.calibrated_unit.position()[2] == self.cleaning_bath_position[2]:
            #     self.calibrated_unit.relative_move(-500, axis=2)
            # Step 2: Move Y axis first to align with the safe position value
            logging.debug(f'Moving Y axis to safe position value: {safe_y}')
            self.calibrated_unit.absolute_move(safe_y, axis=1)
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

            # Step 3: Simultaneously move X and Z axes to reach the safe position
            logging.debug(f'Moving X and Z axes to safe position values: X={safe_x}, Z={safe_z}')
            self.calibrated_unit.absolute_move_group([safe_x,safe_y,safe_z], [0,1,2])
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

        finally:
            pass
        
    def move_to_home_space(self):
        '''
        Moves the pipette and stage to the home space.
        '''
        if self.home_position is None:
            raise ValueError('Home position has not been set')

        try:
            # # Extract individual coordinates from the home position
            home_x, home_y, home_z = self.home_position
            stage_home_x, stage_home_y, microscope_home_z = self.home_stage_position
            # step 0: move the microscope to the home position
            logging.debug(f'Moving microscope to home position value: Z={microscope_home_z}')
            self.microscope.absolute_move(microscope_home_z)
            self.microscope.wait_until_still()
            # self.sleep(0.5)
            # Step 1: Move the stage to the home position
            logging.debug(f'Moving stage to home position values: X={stage_home_x}, Y={stage_home_y}')
            self.calibrated_stage.absolute_move([stage_home_x,stage_home_y])
            self.calibrated_stage.wait_until_still()
            # # step 1.5: move pipette up if at cleaning position:
            # if self.cleaning_bath_position is not None and self.calibrated_unit.position()[2] == self.cleaning_bath_position[2]:
            #     self.calibrated_unit.relative_move(-500, axis=2)
            # Step 2: Move Y axis first to align with the home position value
            logging.debug(f'Moving Y axis to home position value: {home_y}')
            self.calibrated_unit.absolute_move(home_y, axis=1)
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

            # Step 3: Simultaneously move X and Z axes to reach the home position
            logging.debug(f'Moving X and Z axes to home position values: X={home_x}, Z={home_z}')
            self.calibrated_unit.absolute_move_group([home_x,home_y,home_z], [0,1,2])
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

        finally:
            pass

    def move_group_down(self,dist = 100):
        '''
        Moves the microsope and manipulator down by input distance in the z axis
        '''

        self.info('MOVING GROUP DOWN')

        try:
            self.calibrated_unit.relative_move(dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
            self.microscope.relative_move(dist)
            self.microscope.wait_until_still()
            # end = time.perf_counter_ns()
            # print(f"Time taken to move down: {(end-start)/1e6} ms")
        finally:
            pass

    def move_group_up(self,dist = 100):
        '''
        Moves the microscope and manipulator up by input distance in the z axis
        '''
    
        try:
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
            self.microscope.relative_move(-dist)
            self.microscope.wait_until_still()
        finally:
            pass
    
    def move_pipette_up(self, dist = 5000):
        '''
        Moves the pipette up by input distance in the z axis
        '''
        try:
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
        finally:
            pass

    def clean_pipette(self):
        if self.cleaning_bath_position is None:
            raise ValueError('Cleaning bath position has not been set')

        if self.safe_position is None:
            raise ValueError('Safe position has not been set')
        # TODO: implement an abort mechanism
        try:
            start_x, start_y, start_z = self.calibrated_unit.position()
            safe_x, safe_y, safe_z = self.safe_position
            # Step 1: Move to the safe space
            self.move_to_safe_space()
            clean_x, clean_y, clean_z = self.cleaning_bath_position
            # Step 2: Move the pipette above the cleaning bath in the x and y directions
            self.calibrated_unit.absolute_move(clean_y, axis=1)
            self.calibrated_unit.wait_until_still(1)
            self.calibrated_unit.absolute_move(clean_x, axis=0)
            self.calibrated_unit.wait_until_still(0)
            # Step 3: Move the pipette down to the cleaning bath
            self.calibrated_unit.absolute_move(clean_z, axis=2)
            self.calibrated_unit.wait_until_still(2)

            # Step 4: Cleaning
            # Fill up with the Alconox
            self.pressure.set_ATM(atm=False)
            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.75)
                self.pressure.set_pressure(1000)
                self.sleep(0.75)

            # Step 5: Drying
            # move pipette back to safe space in reverse
            self.calibrated_unit.absolute_move(safe_z, axis=2)
            self.calibrated_unit.wait_until_still(2)
            self.calibrated_unit.absolute_move(safe_x, axis=0)
            self.calibrated_unit.wait_until_still(0)
            self.calibrated_unit.absolute_move(safe_y, axis=1)
            self.calibrated_unit.wait_until_still(1)

            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.75)
                self.pressure.set_pressure(1000)
                self.sleep(0.75)
            self.pressure.set_pressure(50)
  
            # Step 6: Move back to start from safespace
            self.calibrated_unit.absolute_move_group([start_x,safe_y,start_z], [0,1,2])
            self.calibrated_unit.wait_until_still()
            self.calibrated_unit.absolute_move(start_y, axis=1)
            self.calibrated_unit.wait_until_still() # Ensure movement completes
        finally:
            pass

    def test_movement(self, path: str, target_frequency: int = 12):
        """
        Moves the pipette and stage based on parsed data, updating positions every
        ``1/target_frequency`` seconds — all without relying on *pandas*.

        The file must be semicolon‑delimited with a header row::

            timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z
        """
        # --- Step 1 — Rapid parse with csv.DictReader -------------------------
        data = {h: [] for h in (
            'timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z'
        )}
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                for key in data:
                    data[key].append(float(row[key]))

        # --- Step 2 — Down‑sample --------------------------------------------
        filtered = self._downsample_data(data, target_frequency)

        # --- Step 3 — Move to the initial position ---------------------------
        self.calibrated_stage.absolute_move([
            filtered['st_x'][0], filtered['st_y'][0], filtered['st_z'][0]
        ])
        self.calibrated_unit.absolute_move_group(
            [filtered['pi_x'][0], filtered['pi_y'][0], filtered['pi_z'][0]], [0, 1, 2]
        )

        self.stop_event = threading.Event()
        self.movement_thread = threading.Thread(
            target=self._movement_loop, args=(filtered,)
        )
        self.info('Movement Test started')
        self.movement_thread.start()

    def _downsample_data(self, data: dict, target_frequency: int = 12) -> dict:
        """Return a *new* dict containing rows sampled at ``target_frequency`` Hz."""
        timestamps = data['timestamp']
        t0 = timestamps[0]
        # Normalise to start at 0 seconds
        rel_time = [t - t0 for t in timestamps]

        interval = 1.0 / target_frequency
        max_time = rel_time[-1]
        target_times = [i * interval for i in range(int(max_time // interval) + 1)]

        filtered = {k: [] for k in data}
        idx = 0
        n = len(rel_time)

        for t in target_times:
            # Advance until the first record >= target time ("forward" merge rule)
            while idx < n and rel_time[idx] < t:
                idx += 1
            if idx == n:
                break
            for key in data:
                filtered[key].append(data[key][idx])

        # Replace timestamp column with zero‑based times
        filtered['timestamp'] = [ts - t0 for ts in filtered['timestamp']]
        self.info(f"Filtered data to {len(filtered['timestamp'])} rows")
        return filtered

    def _movement_loop(self, data: dict):
        """Executes calibrated moves at each timestamp in *data*."""
        start = time.perf_counter()
        count = len(data['timestamp'])

        for i in range(count):
            if self.stop_event.is_set():
                self.info('Movement Test stopped')
                break

            # Uncomment if stage moves also needed
            # self.calibrated_stage.absolute_move([
            #     data['st_x'][i], data['st_y'][i], data['st_z'][i]
            # ])
            self.calibrated_unit.absolute_move_group(
                [data['pi_x'][i], data['pi_y'][i], data['pi_z'][i]], [0, 1, 2]
            )

            target_time = data['timestamp'][i]
            while time.perf_counter() < start + target_time:
                if self.stop_event.is_set():
                    break

        self.info('Movement Test completed')

    def stop_movement(self):
        """Requests the movement loop to halt and waits for the thread to finish."""
        if getattr(self, 'stop_event', None):
            self.stop_event.set()
        if getattr(self, 'movement_thread', None):
            self.movement_thread.join()
