import time

import numpy as np
import cv2
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.devices.amplifier.DAQ import DAQ
from holypipette.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.pressurecontroller import PressureController
import collections

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
        self.cleaning_bath_position = None
        self.rinsing_bath_position = None
        self.contact_position = None
        self.initial_resistance = None

        self.current_protocol_graph = None
        
    def state_emitter(self, state):
        self.info(f"emitting state: {state}")

    def run_protocols(self):
        if self.config.voltage_protocol:
            print('Running voltage protocol')
            self.run_voltage_protocol()
            self.sleep(0.25)
        if self.config.current_protocol:
            self.run_current_protocol()
            self.sleep(0.25)
        if self.config.holding_protocol:
            self.run_holding_protocol()

    def run_voltage_protocol(self):
        self.info('Running voltage protocol (membrane test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        self.info("Getting data from voltage protocol")
        self.daq.getDataFromVoltageProtocol()
        self.sleep(0.25)
        self.info('finished running voltage protocol (membrane test)')

    def run_current_protocol(self):
        self.info('Running current protocol (current clamp)')
        self.amplifier.current_clamp()
        self.sleep(0.25)
        self.daq.getDataFromCurrentProtocol()
        self.sleep(0.25)
        self.amplifier.voltage_clamp()
        self.info('finished running current protocol(current clamp)')

    def run_holding_protocol(self):
        self.info('Running holding protocol (E/I PSC test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        self.amplifier.set_holding(-0.070)
        self.sleep(0.25)
        self.daq.getDataFromHoldingProtocol()
        self.sleep(0.25)
        # self.amplifier.set_holding(0)
        self.sleep(0.25)
        self.amplifier.voltage_clamp()
        self.info('finished running holding protocol (E/I PSC test)')

        
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

        #verify that the rig is calibrated

        #check for stage and pipette calibration
        if not self.calibrated_unit.calibrated:
            raise AutopatchError("Pipette not calibrated")
        if not self.calibrated_stage.calibrated:
            raise AutopatchError("Stage not calibrated")
        if self.microscope.floor_Z is None:
            raise AutopatchError("Cell Plane not set")
        
        if cell is None:
            raise AutopatchError("No cell given to patch!")

        cell_pos, cell_img = cell
        
        lastResDeque = collections.deque(maxlen=3)

        #setup amp for patching
        self.amplifier.start_patch()

        #ensure "near cell" pressure
        self.pressure.set_pressure(self.config.pressure_near)

        # Check initial resistance
        self.amplifier.auto_pipette_offset()
        self.sleep(1) # TODO is this needed?
        self.amplifier.voltage_clamp()

        # set amplifier to resistance mode
        daqResistance = self.daq.resistance()
        # lastResDeque.append(R)
        self.debug("Resistance:" + str(daqResistance/1e6))

        #ensure good pipette (not broken or clogged)
        # self._verify_resistance()

        #center stage on the cell we want to patch 
        self.calibrated_stage.safe_move(np.array([cell_pos[0], cell_pos[1], 0]))
        self.calibrated_stage.wait_until_still()

        print('centered stage on the cell!')
        self.sleep(1) #just for testing

        #focus on the cell plane
        self.microscope.absolute_move(self.microscope.floor_Z)
        self.microscope.wait_until_still()

        print('moved cell plane into focus!')

        #get image
        img = self.calibrated_unit.camera.get_16bit_image()

        img_mins = np.min([np.min(img), np.min(cell_img)])
        img_maxs = np.max([np.max(img), np.max(cell_img)])

        img = (img - img_mins)/(img_maxs - img_mins)
        cell_img = (cell_img - img_mins)/(img_maxs - img_mins)

        img = (img*255).astype(np.uint8)
        cell_img = (cell_img*255).astype(np.uint8)

        #find via template matching
        res = cv2.matchTemplate(img, cell_img, cv2.TM_CCOEFF_NORMED) 
        threshold = 0.8

        max_pos = np.unravel_index(np.argmax(res), res.shape)
        max_val = np.max(res)

        print(max_pos, max_val)

        if max_val > threshold:
            max_pos = (max_pos[0] + cell_img.shape[0] // 2, max_pos[0] + cell_img.shape[1] // 2)
            self.calibrated_unit.camera.show_point(max_pos, color=(255, 0, 0), radius=15, duration=5)
        else:
            print('couldn\'t find cell in image!')

        print('taking resistance readings...')
        for i in range(3):
            self.sleep(1)
            daqResistance = self.daq.resistance()
            lastResDeque.append(daqResistance)
            print(f"resistance: {daqResistance}")

        #move above cell plane (where we're about to move the pipette to)
        self.microscope.absolute_move(self.microscope.floor_Z + self.config.cell_distance * 5)
        self.microscope.wait_until_still()
        self.microscope.fix_backlash()
        print('moved pipette setpoint plane into focus!')
        self.sleep(1) #just for testing

        #move the pipette to the correct spot (config distance above the cell plane)
        
        self.calibrated_unit.safe_move(np.array([0, 0, self.microscope.position()]))
        self.calibrated_unit.wait_until_still()
        print('moved pipette to cell plane!')


        self.sleep(1) #just for testing

        # create a pipette setpoint in stage coordinates
        # cell_distance = self.calibrated_unit.um_to_pixels_relative(np.array([0, 0, -self.config.cell_distance]))
        # cell_distance = cell_distance[2]
        # pipette_setpoint = np.array([0, 0, self.microscope.position() + cell_distance])
        # print('moving pipette 30um above cell pos: ({})'.format(pipette_setpoint))
        # self.calibrated_unit.safe_move(pipette_setpoint, yolo_correction=False)
        # self.calibrated_unit.wait_until_still()

        # Check resistance again
        self._verify_resistance()
        daqResistance = self.daq.resistance()
        # lastResDeque.append(R)

        # recal pipette offset in multiclamp
        self.amplifier.auto_pipette_offset()
        self.sleep(2)

        # Approach and make the seal
        self.info("Approaching the cell")
        success = False


        # phase 1: hunt for the cell
        cellFound = False
        for _ in range(int(self.config.max_distance)):  # move 15 um down
            # move by 1 um down
            self.calibrated_unit.relative_move(1, axis=2, speed=100)  # *calibrated_unit.up_position[2]
            self.abort_if_requested()
            self.calibrated_unit.wait_until_still(2)
            self.sleep(1)
            self.amplifier.voltage_clamp()
            daqResistance = self.daq.resistance()
            lastResDeque.append(daqResistance)

            self.info("R = " + str(self.daq.resistance() / 1e6))
            if self._isCellDetected(lastResDeque):
                cellFound = True
                break #we found a cell!

        if not cellFound:
            self.amplifier.stop_patch()
            self.pressure.set_pressure(20)
            raise AutopatchError("Couldn't detect a cell")
        
        # move a bit further down to make sure we're at the cell
        self.calibrated_unit.relative_move(1, axis=2)


        # phase 2: attempt to form a gigaseal
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

        # Phase 3: break into cell
        self.break_in()

    def clean_pipette(self):
        if self.cleaning_bath_position is None:
            raise ValueError('Cleaning bath position has not been set')
        # if self.rinsing_bath_position is None:
        #     raise ValueError('Rinsing bath position has not been set')
        try:
            start_position = self.calibrated_unit.position()
            # Move the pipette to the washing bath.
            self.calibrated_unit.absolute_move(0, 2)
            print('moving up to 0...')
            self.calibrated_unit.wait_until_still(2)
            self.calibrated_unit.absolute_move(self.cleaning_bath_position[0], 0)
            print('moving to cleaning bath x, y...')
            self.calibrated_unit.wait_until_still(0)
            self.calibrated_unit.absolute_move(self.cleaning_bath_position[1], 1)
            self.calibrated_unit.wait_until_still(1)
            print('moving to cleaning bath z...')
            self.calibrated_unit.absolute_move(self.cleaning_bath_position[2], 2)
            self.calibrated_unit.wait_until_still(2)
            # Fill up with the Alconox
            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.625)
                self.pressure.set_pressure(1000)
                self.sleep(0.375)

            # # Step 2: Rinsing.
            # # Move the pipette to the rinsing bath.
            # self.calibrated_unit.absolute_move(self.rinsing_bath_position[2] - 5000, 2)
            # self.calibrated_unit.wait_until_still(2)
            # self.calibrated_unit.absolute_move(self.rinsing_bath_position[1], 1)
            # self.calibrated_unit.wait_until_still(1)
            # self.calibrated_unit.absolute_move(self.rinsing_bath_position[0], 0)
            # self.calibrated_unit.wait_until_still(0)
            # self.calibrated_unit.absolute_move(self.rinsing_bath_position[2], 2)
            # self.calibrated_unit.wait_until_still(2)
            # # Expel the remaining Alconox
            # self.pressure.set_pressure(1000)
            # self.sleep(6)

            # Step 3: Move back.
            self.calibrated_unit.absolute_move(0, 2)
            self.calibrated_unit.wait_until_still(2)
            self.calibrated_unit.absolute_move(start_position[1], 1)
            self.calibrated_unit.wait_until_still(1)
            self.calibrated_unit.absolute_move(start_position[0], 0)
            self.calibrated_unit.wait_until_still(0)
            self.pressure.set_pressure(self.config.pressure_near)
            self.calibrated_unit.absolute_move(start_position[2], 2)
            self.calibrated_unit.wait_until_still(2)
            
        finally:
            pass
