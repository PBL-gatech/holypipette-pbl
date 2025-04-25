

import time

import numpy as np
import cv2
import ctypes
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import NiDAQ
from holypipette.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.pressurecontroller import PressureController
import collections
import logging

from holypipette.interface.patchConfig import PatchConfig

from .base import TaskController


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

    def run_protocols(self):

        self.daq.setCellMode(True)
        if self.config.voltage_protocol:
            self.run_voltage_protocol()
            self.sleep(0.25)
        if self.config.current_protocol:
            self.iholding = self.daq.holding_current
            # self.iholding = 0
            logging.debug(f"custom_current_protocol state: {self.config.custom_protocol}")
            logging.debug(f"start cclamp current: {self.config.cclamp_start}")
            logging.debug(f"end cclamp current: {self.config.cclamp_end}")
            logging.debug(f"step cclamp current: {self.config.cclamp_step}")
            self.run_current_protocol()
            self.sleep(0.25)
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
        self.amplifier.set_bridge_balance(True)
        self.info('auto bridge balance')
        self.amplifier.auto_bridge_balance()
        self.sleep(0.1)
        self.amplifier.set_neutralization_capacitance(cap)
        self.info('set neutralization capacitance')
        self.amplifier.set_neutralization_enable(True)
        self.info('enabled neutralization')
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
        if self.config.custom_protocol:
            self.debug('running custom current protocol')
            self.daq.getDataFromCurrentProtocol(custom =self.config.custom_protocol,factor= 1,startCurrentPicoAmp=(self.config.cclamp_start), endCurrentPicoAmp=(self.config.cclamp_end), stepCurrentPicoAmp=(self.config.cclamp_step), highTimeMs=400)                                            
        else:
            self.debug('running default current protocol')
            self.daq.getDataFromCurrentProtocol(custom=self.config.custom_protocol, factor=1, startCurrentPicoAmp=None, endCurrentPicoAmp=None, stepCurrentPicoAmp=10, highTimeMs=400)
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
        self.daq.getDataFromHoldingProtocol()
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
        self.info("Setting pressure to 200 mbar")
        self.pressure.set_pressure(200)
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
            self.config.cell_distance = 50

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
            self.config.cell_R_increase = 0.150

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

    def accessRamp(self, num_measurements=3, interval=0.2):
        measurements = []
        for _ in range(num_measurements):
            measurements.append(self.daq.accessResistance())
            self.sleep(interval)
        return sum(measurements) / len(measurements)
    
    def resistanceRamp(self, num_measurements=3, interval=0.2):
        measurements = []
        for _ in range(num_measurements):
            measurements.append(self.daq.resistance())
            self.sleep(interval)
        return sum(measurements) / len(measurements)
    
    def capacitanceRamp(self, num_measurements=3, interval=0.2):
        measurements = []
        for _ in range(num_measurements):
            measurements.append(self.daq.capacitance())
            self.sleep(interval)
        return sum(measurements) / len(measurements)


    def gigaseal(self):
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
            max_pressure = self.config.pressure_ramp_max

        holding_switched = False
        cap_switched = False
        last_progress_time = time.time()

        while avg_resistance < self.config.gigaseal_R and not self.abort_requested:
            if time.time() - last_progress_time >= self.config.seal_deadline:
                raise AutopatchError("Seal unsuccessful: resistance did not improve significantly.")

            prev_resistance = avg_resistance
            avg_resistance = self.resistanceRamp()

            delta_resistance = avg_resistance - prev_resistance
            rate_mohm_per_sec = delta_resistance / (5 * 0.2)

            if delta_resistance >= self.config.gigaseal_min_delta_R:
                last_progress_time = time.time()
                
            print(f"goal resistance: {self.config.gigaseal_R} MΩ; current resistance: {avg_resistance} MΩ; rate: {rate_mohm_per_sec} MΩ/s")
            if autoPressure:
                if -(self.config.gigaseal_R /100) < rate_mohm_per_sec < self.config.gigaseal_R/3000: # less than 0.33, or negative 10? 
                    print(f"Rate under 330kohm/s: {rate_mohm_per_sec} MΩ/s")
                    currPressure -= 5
                    speed = 3
                    max_pressure = -45
                    self.config.pressure_ramp_max = max_pressure
                elif self.config.gigaseal_R/3000 <= rate_mohm_per_sec <= self.config.gigaseal_R/10: # between 0.33 and 100
                    speed = 1   # Maintain current pressure
                elif self.config.gigaseal_R/10 < rate_mohm_per_sec <= self.config.gigaseal_R/5: # between 100 and 200
                    print(f"Rate over 100mohm/s : {rate_mohm_per_sec} MΩ/s")
                    max_pressure = self.config.pressure_ramp_max
                    currPressure += 5
                    speed = 3
                elif rate_mohm_per_sec <= -(self.config.gigaseal_R/100): # less than -10 mohm/s
                    print(f"Rate too negative: {rate_mohm_per_sec} MΩ/s")
                    currPressure += 5
                    if currPressure > -5:
                        currPressure = -5
                    speed = 0.5
                
                currPressure = max(currPressure, max_pressure)
                if currPressure != prevpressure:
                    self.pressure.set_pressure(currPressure)
                    prevpressure = currPressure
                    self.sleep(5/speed)

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

            if avg_resistance >= self.config.gigaseal_R/12 and not holding_switched:
                self.amplifier.set_holding(self.config.Vramp_amplitude)
                self.amplifier.switch_holding(True)
                holding_switched = True

            if avg_resistance >= self.config.gigaseal_R/3 and  not cap_switched:
                self.daq.setCellMode(False)
                self.amplifier.auto_fast_compensation()
                self.sleep(1)
                self.daq.setCellMode(True)
                cap_switched = True


            if avg_resistance >= self.config.gigaseal_R:
                self.pressure.set_ATM(atm=True)
                self.info(f"Seal successful!")
                return

        raise AutopatchError("Seal unsuccessful: gigaseal criteria not met.")
    
    def break_in(self):
        '''
        Performs cell membrane Break in. The pipette must be in cell-attached Mode.
        '''
        self.daq.setCellMode(True)
        if self.config.mode == 'Classic':
            autoPressure = True
        else:
            autoPressure = False
        self.info(f"{self.config.mode}: Attempting Break in...")
        # self.pressure.set_ATM(atm=True)
        self.sleep(3)
        self.pressure.set_pressure(self.config.pulse_pressure_break_in)
        self.amplifier.set_zap_duration(25*1e-6)

        measuredAccessResistance = self.accessRamp()
        measuredResistance = self.resistanceRamp()
        measuredCapacitance = self.capacitanceRamp()
        self.info(f"Initial Resistance: {measuredResistance}; Initial Capacitance: {measuredCapacitance},  Initial Access Resistance: {measuredAccessResistance}")
        self.info(f"target Resistance: {self.config.max_cell_R}; Target Capacitance: {self.config.min_cell_C}, Target Access Resistance: {self.config.max_access_R}")

        # # Check if the gigaseal is lost
        # if measuredResistance is not None and measuredResistance < self.config.max_cell_R*1e-6 and measuredCapacitance is not None and measuredCapacitance > self.config.min_cell_C*1e-12 or measuredAccessResistance > self.config.max_access_R*1e-6:
        #     # self.done = True
        #     raise AutopatchError("Seal lost")
        
        trials = 0
        speed = 3
        while measuredAccessResistance > self.config.max_access_R*1e-6:


            if autoPressure:
                trials += 1
                self.debug(f"Trial: {trials}")
                
                # Apply pressure pulses
                speedosc = trials % 5
                if speedosc == 0:
                    speed = 2
                self.pressure.set_ATM(atm=False)
                self.sleep(1/speed)
                self.pressure.set_ATM(atm=True)
                self.sleep(0.75)
                # Optional zapping after a few trials every 3rd trial
                speed = 3
                osc = trials % 3
                if self.config.zap and osc == 0:
                    self.info("zapping")
                    self.amplifier.zap()
                    self.sleep(0.5)
                    self.amplifier.zap()
                    self.sleep(0.5)

                self.sleep(1)
            
            # Take new measurements using ramp functions to compute running averages.
            measuredResistance = self.resistanceRamp()
            measuredAccessResistance = self.accessRamp()
            measuredCapacitance = self.capacitanceRamp()

            if autoPressure:
                self.info(f"Trial {trials}: Running Avg Membrane Resistance: {measuredResistance}; Membrane Capacitance: {measuredCapacitance}, Access Resistance: {measuredAccessResistance}")
                
                # Fail after too many attempts.
                if trials > 15:

                    self.info("Break-in unsuccessful")
                    raise AutopatchError("Break-in unsuccessful")
        
        self.info("Successful break-in, Running Avg Access Resistance = " + str(measuredAccessResistance))

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
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
            self.microscope.relative_move(dist)
            self.microscope.wait_until_still()
        finally:
            pass
    def move_group_up(self,dist = 100):
        '''
        Moves the microscope and manipulator up by input distance in the z axis
        '''
    
        try:
            self.calibrated_unit.relative_move(dist, axis=2)
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
            self.pressure.set_pressure(200)
  
            # Step 6: Move back to start from safespace
            self.calibrated_unit.absolute_move_group([start_x,safe_y,start_z], [0,1,2])
            self.calibrated_unit.wait_until_still()
            self.calibrated_unit.absolute_move(start_y, axis=1)
            self.calibrated_unit.wait_until_still() # Ensure movement completes
        finally:
            pass
