# coding=utf-8
'''
Control of automatic patch clamp algorithm
'''
import numpy as np

from holypipette.interface import TaskInterface, command, blocking_command
from holypipette.controller import AutoPatcher
from holypipette.devices.pressurecontroller.BasePressureController import PressureController
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.interface.pipettes import PipetteInterface
from holypipette.devices.amplifier.DAQ import DAQ
from .patchConfig import PatchConfig
from PyQt5 import QtCore
import time

__all__ = ['AutoPatchInterface']

class AutoPatchInterface(TaskInterface):
    '''
    A class to run automatic patch-clamp
    '''
    def __init__(self, amplifier: Amplifier, daq: DAQ, pressure: PressureController, pipette_interface: PipetteInterface):
        super().__init__()
        self.config = PatchConfig(name='Patch')
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.pipette_controller = pipette_interface
        autopatcher = AutoPatcher(amplifier, daq, pressure, self.pipette_controller.calibrated_unit,
                                    self.pipette_controller.calibrated_unit.microscope,
                                    calibrated_stage=self.pipette_controller.calibrated_stage,
                                    config=self.config)
        self.current_autopatcher = autopatcher

        self.is_selecting_cells = False
        self.cells_to_patch = []

        #call update_camera_cell_list every 0.1 seconds using a QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_cell_list)
        self.timer.start(50)
        

    @blocking_command(category='Patch', description='Break into the cell',
                      task_description='Breaking into the cell')
    def break_in(self):
        self.execute(self.current_autopatcher.break_in)

    def start_selecting_cells(self):
        self.is_selecting_cells = True

    def remove_last_cell(self):
        if len(self.cells_to_patch) > 0:
            self.cells_to_patch = self.cells_to_patch[:-1]

    @blocking_command(category='Cell Sorter',
            description='Move the cell sorter to a cell',
            task_description='Move the cell sorter to a cell')
    def move_cellsorter_to_cell(self):
        #grab cell from list
        cellx = self.cells_to_patch[0][0][0]
        celly = self.cells_to_patch[0][0][1]
        cellz = self.pipette_controller.calibrated_unit.microscope.floor_Z

        #move cell sorter to cell
        self.execute(self.pipette_controller.calibrated_cellsorter.center_cellsorter_on_point, argument=[cellx, celly, cellz])

    @blocking_command(category='DAQ',
            description='Run Protocols on the Cell',
            task_description='Run Protocols on the Cell')
    def run_protocols(self):
        self.execute(self.current_autopatcher.run_protocols)
    

    @command(category='Patch', description='Add a mouse position to the list of cells to patch')
    def add_cell(self, position):
        #add half the size of the camera image to the position to get the center of the cell
        position = np.array(position)

        position[0] += self.current_autopatcher.calibrated_unit.camera.width / 2
        position[1] += self.current_autopatcher.calibrated_unit.camera.height / 2
        print(f'adding cell... {self.is_selecting_cells}')
        if self.is_selecting_cells:
            print('Adding cell at', position, 'to list of cells to patch')
            stage_pos_pixels = self.current_autopatcher.calibrated_stage.reference_position()
            stage_pos_pixels[0:2] -= position
            #take a 256x256 image centered on the cell
            img = self.current_autopatcher.calibrated_unit.camera.get_16bit_image()
            img = img[int(position[1]-128):int(position[1]+128), int(position[0]-128):int(position[0]+128)]
            if img is None or img.shape != (256, 256):
                raise RuntimeError('Cell too Close to edge!')
            
            #save the image
            # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # cv2.imwrite(f'cell_{len(self.cells_to_patch)}.png', img)
            self.cells_to_patch.append((np.array(stage_pos_pixels), img))
            self.is_selecting_cells = False

    def update_camera_cell_list(self) -> None:
        self.current_autopatcher.calibrated_unit.camera.cell_list = []
        for cell, img in self.cells_to_patch:
            camera_pos = -cell + self.current_autopatcher.calibrated_stage.reference_position()
            self.current_autopatcher.calibrated_unit.camera.cell_list.append(camera_pos[0:2].astype(int))
            

    @blocking_command(category='Patch', description='Move to cell and patch it',
                      task_description='Moving to cell and patching it')
    def patch(self) -> None:
        cell, img = self.cells_to_patch[0]
        self.execute(self.current_autopatcher.patch,
                     argument=(cell, img))
        time.sleep(2)
        self.cells_to_patch = self.cells_to_patch[1:]
        
    @command(category='Patch',
             description='Store the position of the washing bath',
             success_message='Cleaning path position stored')
    def store_cleaning_position(self) -> None:
        self.current_autopatcher.cleaning_bath_position = self.pipette_controller.calibrated_unit.position()

    @command(category='Patch',
             description='Store the position of the rinsing bath',
             success_message='Rinsing bath position stored')
    def store_rinsing_position(self) -> None:
        self.current_autopatcher.rinsing_bath_position = self.pipette_controller.calibrated_unit.position()

    @blocking_command(category='Patch',
                      description='Clean the pipette (wash and rinse)',
                      task_description='Cleaning the pipette')
    def clean_pipette(self):
        self.execute(self.current_autopatcher.clean_pipette)

    @blocking_command(category='Patch',
                      description='Sequential patching and cleaning for multiple cells',
                      task_description='Sequential patch clamping')
    def sequential_patching(self):
        self.execute(self.current_autopatcher.sequential_patching)

    @blocking_command(category='Patch',
                      description='Moving down the calibrated manipulator to detect the contact point with the coverslip',
                      task_description='Contact detection')
    def contact_detection(self):
        # TODO: Figure out what this should link to
        self.execute(self.current_autopatcher.contact_detection)

    
    def set_pressure_near(self) -> None:
        '''puts the pipette under positive pressure to prevent blockages
        '''
        self.pressure.set_pressure(self.config.pressure_near)