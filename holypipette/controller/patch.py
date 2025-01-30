import time

import numpy as np
import cv2
import ctypes
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import DAQ
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
    def __init__(self, amplifier: Amplifier, daq: DAQ, pressure: PressureController, calibrated_unit: CalibratedUnit, microscope: Microscope, calibrated_stage: CalibratedStage, config : PatchConfig):
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
        

        self.current_protocol_graph = None
        
    def state_emitter(self, state):
        self.info(f"emitting state: {state}")

    def run_protocols(self):
        # TODO : implement an abort mechanism
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
            self.daq.getDataFromCurrentProtocol(custom=self.config.custom_protocol, factor=2, startCurrentPicoAmp=None, endCurrentPicoAmp=None, stepCurrentPicoAmp=10, highTimeMs=400)
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
            if self.microscope.floor_Z is None:
                raise AutopatchError("Cell Plane not set")
            self.rig_ready = True
        except (AutopatchError, ValueError) as e:
            self.rig_ready = False
            raise e

    
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
        
        # move stage and pipette to safe space
        print("Moving to safe space")
        self.move_to_safe_space()
        self.calibrated_stage.wait_until_still()
        self.calibrated_unit.wait_until_still()
        # set pipette pressure to 200 mbar
        print("Setting pressure to 200 mbar")
        self.pressure.set_pressure(200)
        # move to home space
        logging.debug("Moving to home space")
        self.move_to_home_space()
        self.calibrated_stage.wait_until_still()
        self.calibrated_unit.wait_until_still()
        # center pipette and stage on cell XY
        cell_pos, cell_img = cell
        # print(f"Cell img: {cell_img}")

        print(f" Moving to Cell position: {cell_pos}") # in um i think?
        # moving stage to xy position of cell
        # home position
        cell_pos = np.array([cell_pos[0], cell_pos[1], 0])
        self.calibrated_stage.safe_move(np.array(cell_pos))
        self.calibrated_stage.wait_until_still()

        # move pipette to xy position of cell
        stage_pos = self.calibrated_stage.pixels_to_um(self.calibrated_stage.reference_position())
        disp = np.zeros(3)
        disp[0] = stage_pos[0] - self.home_stage_position[0]
        disp[1] = stage_pos[1] - self.home_stage_position[1]
        disp[2] = 0
        curr = self.calibrated_unit.position()
        desired = curr + disp
        self.calibrated_unit.absolute_move(desired)
        self.calibrated_unit.wait_until_still()
        
        # # move stage and pipette to 20 um above cell plane
        print("Moving to cell plane")

        zdist =  stage_pos[2] - self.microscope.floor_Z + 20

        self.calibrated_unit.absolute_move(np.array([0,0, zdist]))
        self.calibrated_unit.wait_until_still()
        self.calibrated_stage.absolute_move(np.array[0,0, self.microscope.floor_Z + 20])
        self.calibrated_stage.wait_until_still()
        
        # #ensure "near cell" pressure
        self.pressure.set_pressure(self.config.pressure_near)

        lastResDeque = collections.deque(maxlen=3)
        # get initial resistance
        daqResistance = self.daq.resistance()
        lastResDeque.append(daqResistance)

        # move pipette down at 1um/s and check reistance every 50 ms
        # get starting position 
        start_pos = self.calibrated_unit.position()

        # self.calibrated_unit.absolute_move_group_velocity(1, [2])
        # self.calibrated_stage.microscope.absolute_move_velocity(1)
        # while not self._isCellDetected(lastResDeque):
        #     curr_pos = self.calibrated_unit.position()
        #     if self.abort_requested():
        #         # stop the movement
        #         self.calibrated_unit.stop()
        #         self.calibrated_stage.microscope.stop()
        #         self.info("Hunt cancelled")
        #         break
        #     elif abs(curr_pos[2] - start_pos[2]) >= int(self.config.max_distance):
            #     # we have moved 25 um down and still no cell detected
            #     self.calibrated_unit.stop()
            #     self.calibrated_stage.microscope.stop()
            #     self.info("No cell detected")
            #     break
            # elif self._isCellDetected(lastResDeque):
            #     self.info("Cell detected")
            #     self.calibrated_unit.stop()
            #     self.calibrated_stage.microscope.stop()
            #     break
            # #TODO will add another condition to check if cell and pipette have moved away from each other based on the mask and original image.
            # self.sleep(0.05)
            # lastResDeque.append(daqResistance)
            # daqResistance = self.daq.resistance()

    def gigaseal(self):
        lastResDeque = collections.deque(maxlen=3)
        # Release pressure
        self.info("Cell Detected, Lowering pressure")
        currPressure = 0
        self.pressure.set_pressure(currPressure)
        self.amplifier.set_holding(self.config.Vramp_amplitude)
        
        self.sleep(10)
        t0 = time.time()
        while daqResistance < self.config.gigaseal_R:
            t = time.time()
            if currPressure < -40:
                currPressure = 0
            self.pressure.set_pressure(currPressure)
                
            if t - t0 >= self.config.seal_deadline:
                # Time timeout for gigaseal
                self.amplifier.stop_patch()
                self.pressure.set_pressure(20)
                raise AutopatchError("Seal unsuccessful")
            
            # did we reach gigaseal?
            daqResistance = self.daq.resistance()
            lastResDeque.append(daqResistance)
            if daqResistance > self.config.gigaseal_R or len(lastResDeque) == 3 and all([lastResDeque == None for x in lastResDeque]):
                success = True
                break
            
            # else, wait a bit and lower pressure
            self.sleep(5)
            currPressure -= 10

        if not success:
            self.pressure.set_pressure(20)
            raise AutopatchError("Seal unsuccessful")

        self.info("Seal successful, R = " + str(self.daq.resistance() / 1e6))

    def break_in(self):
        '''
        Breaks in. The pipette must be in cell-attached mode
        '''
        self.info("Breaking in")
        measuredResistance = self.daq.resistance()
        # if R is not None and R < self.config.gigaseal_R:
        #     raise AutopatchError("Seal lost")

        pressure = 0
        trials = 0
        while measuredResistance is None or self.daq.resistance() > self.config.max_cell_R:  # Success when resistance goes below 300 MOhm
            trials += 1
            self.debug(f"Trial: {trials}")
            pressure += self.config.pressure_ramp_increment
            if abs(pressure) > abs(self.config.pressure_ramp_max):
                raise AutopatchError("Break-in unsuccessful")
            if self.config.zap:
                self.debug('zapping')
                self.amplifier.zap()
            self.pressure.ramp(amplitude=pressure, duration=self.config.pressure_ramp_duration)
            self.sleep(1.3)

        self.info("Successful break-in, R = " + str(self.daq.resistance() / 1e6))

    def _verify_resistance(self):
        return # * just for testing TODO: remove

        R = self.daq.resistance()

        if R < self.config.min_R:
            # print("Resistance is too low (broken tip?)")
            raise AutopatchError("Resistance is too low (broken tip?)")
        elif self.config.max_R < R:
            # print("Resistance is too high (obstructed?)")
            raise AutopatchError("Resistance is too high (obstructed?)")
        
    def _isCellDetected(self, lastResDeque, cellThreshold = 0.3*10**6):
        '''Given a list of three resistance readings, do we think there is a cell where the pipette is?
        '''
        print(lastResDeque)

        #criteria 1: the last three readings must be increasing
        if not lastResDeque[0] < lastResDeque[1] < lastResDeque[2]:
            return False #last three resistances must be ascending
        
        print('ascending')
        
        #criteria 2: there must be an increase of at least 0.3 mega ohms
        r_delta = lastResDeque[2] - lastResDeque[0]
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

        #setup amp for patching
        self.amplifier.start_patch()
        #! phase 1: hunt for cell
        self.hunt_cell(cell)
        # move a bit further down to make sure we're at the cell
        self.calibrated_unit.relative_move(1, axis=2)
        #! phase 2: attempt to form a gigaseal
        self.gigaseal()

        #! Phase 3: break into cell
        self.break_in()

        #! Phase 4: run protocols
        self.run_protocols()

        #! Phase 5: clean pipette
        # move set pipette pressure to 25 mbar
        self.pressure.set_pressure(25)
        # move pipette and stage up 25 um
        self.move_group_up(dist=25)
        self.sleep(0.1)
        self.clean_pipette()

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
            # Step 1: Move the stage to the safe position
            self.calibrated_stage.absolute_move([safe_stage_x,safe_stage_y])

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
        Moves the pipette to the home space.
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
            self.sleep(0.1)
            # Step 1: Move the stage to the home position
            logging.debug(f'Moving stage to home position values: X={stage_home_x}, Y={stage_home_y}')
            self.calibrated_stage.absolute_move([stage_home_x,stage_home_y])
            self.calibrated_stage.wait_until_still()

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

        try:
            self.calibrated_unit.relative_move(dist, axis=2)
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
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
            self.microscope.relative_move(-dist)
            self.microscope.wait_until_still()
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
            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.625)
                self.pressure.set_pressure(1000)
                self.sleep(0.375)

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
                self.sleep(0.625)
                self.pressure.set_pressure(1000)
                self.sleep(0.375)
            self.pressure.set_pressure(200)
  
            # Step 6: Move back to start from safespace
            self.calibrated_unit.absolute_move_group([start_x,safe_y,start_z], [0,1,2])
            self.calibrated_unit.wait_until_still()
            self.calibrated_unit.absolute_move(start_y, axis=1)
            self.calibrated_unit.wait_until_still() # Ensure movement completes
        finally:
            pass
