# coding=utf-8
'''
Control of automatic patch clamp algorithm
'''
import numpy as np

from holypipette.interface import TaskInterface, command, blocking_command
from holypipette.controller import AutoPatcher
from holypipette.utils import EPhysLogger, RecordingStateManager
from holypipette.devices.pressurecontroller.BasePressureController import PressureController
from holypipette.devices.amplifier.amplifier import Amplifier
from holypipette.interface.pipettes import PipetteInterface
from holypipette.devices.amplifier.DAQ import NiDAQ
from .patchConfig import PatchConfig
from PyQt5 import QtCore
import time

__all__ = ['AutoPatchInterface']

class AutoPatchInterface(TaskInterface):
    '''
    A class to run automatic patch-clamp
    '''
    def __init__(self, amplifier: Amplifier, daq: NiDAQ, pressure: PressureController, pipette_interface: PipetteInterface, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.config = PatchConfig(name='Patch')
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.pipette_controller = pipette_interface
        self.recording_state_manager = recording_state_manager

        self.ephys_logger = EPhysLogger(recording_state_manager=self.recording_state_manager, ephys_filename="CellMetadata")
        autopatcher = AutoPatcher(amplifier, daq, pressure, self.pipette_controller.calibrated_unit,
                                    self.pipette_controller.calibrated_unit.microscope,
                                    calibrated_stage=self.pipette_controller.calibrated_stage,
                                    config=self.config)
        self.current_autopatcher = autopatcher

        self.is_selecting_cells = False
        self.cells_to_patch = []
        # self.done = self.current_autopatcher.done

        #call update_camera_cell_list every 0.05 seconds using a QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_cell_list)
        self.timer.start(50)
        
    
    @blocking_command(category='Patch', description='Break into the cell',
                      task_description='Breaking into the cell')
    def break_in(self):
        self.execute(self.current_autopatcher.break_in)

    @blocking_command(category='Patch', description='GigaSeal the cell',
                      task_description='GigaSealing the cell')
    def gigaseal(self):
        self.execute(self.current_autopatcher.gigaseal)

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
        index = self.recording_state_manager.sample_number
        if self.cells_to_patch:
            stage_coords, img = self.cells_to_patch[0]
            self.ephys_logger.save_cell_metadata(index, stage_coords, img)
        self.execute(self.current_autopatcher.run_protocols)
    

    @command(category='Patch', description='Add a mouse position to the list of cells to patch')
    def add_cell(self, position):
        #add half the size of the camera image to the position to get the center of the cell
        position = np.array(position)

        position[0] += self.current_autopatcher.calibrated_unit.camera.width / 2
        position[1] += self.current_autopatcher.calibrated_unit.camera.height / 2
        # add the z position of the microscope
        z_pos = self.current_autopatcher.calibrated_unit.microscope.position()/5
        self.info(f'z position of the microscope: {z_pos}')
        print(f'adding cell... {self.is_selecting_cells}')
        if self.is_selecting_cells:
            print('Adding cell at', position, 'to list of cells to patch in pixels')
            stage_pos_pixels = self.current_autopatcher.calibrated_stage.reference_position()
            stage_pos_pixels[0:2] -= position
            # display stage position
            # add the z_pos to the stage position as a third dimension in the np array

            stage_pos_pixels = np.array([stage_pos_pixels[0], stage_pos_pixels[1], z_pos])


            print(f'Stage position dimensions: {np.size(stage_pos_pixels)}')
            print(f'Stage um position: {stage_pos_pixels}')
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

    # Update the cell list to store both cell coordinates and image.
    def update_camera_cell_list(self) -> None:
        self.current_autopatcher.calibrated_unit.camera.cell_list = []
        for cell, img in self.cells_to_patch:
            camera_pos = -cell + self.current_autopatcher.calibrated_stage.reference_position()
            # Append a tuple of (coordinates, image)
            self.current_autopatcher.calibrated_unit.camera.cell_list.append((camera_pos[0:2].astype(int), img))


    @command(category='Patch',
                description='emit the states to logger that are being attempted in manual mode',
                success_message='state emitted')
    def state_emitter(self,state):
        self.execute(self.current_autopatcher.state_emitter, argument=(state))
                
    @blocking_command(category='Patch', description='Move to cell and patch it',
                      task_description='Moving to cell and patching it')
    def patch(self) -> None:
        cell, img = self.cells_to_patch[0]
        self.execute(self.current_autopatcher.patch,
                     argument=(cell, img))
        time.sleep(2)
        self.cells_to_patch = self.cells_to_patch[1:]

    @blocking_command(category='Patch',
                        description='Locate the cell',
                        task_description='Moving to the cell')
    def locate_cell(self):
        cell, img = self.cells_to_patch[0]
        self.execute(self.current_autopatcher.locate_cell,
                      argument = (cell, img))
        time.sleep(2)
 
    @blocking_command(category='Stage',
                     description = 'Center the stage on cell',
                      task_description='Centering the stage on cell')
    def center_on_cell(self):
        cell, img = self.cells_to_patch[0]
        # print( f"patch.py: centering on cell {cell} with image {img.shape}")
        self.execute(self.current_autopatcher.calibrated_stage.center_on_cell,
                      argument = (cell, img))

    @blocking_command(category='Patch',
                        description='Hunt the cell',
                        task_description='Moving to the cell and detecting it ')
    def hunt_cell(self):
        cell, img = self.cells_to_patch[0]
        self.execute(self.current_autopatcher.hunt_cell,
                      argument = (cell, img))
        time.sleep(2)
        # self.cells_to_patch = self.cells_to_patch[1:]

    @blocking_command(category='Patch',
                        description='escape the cell',
                        task_description='Moving away from the cell, cleaning pipette and moving to home space')
    def escape_cell(self):
        self.execute(self.current_autopatcher.escape)
        time.sleep(2)
        self.cells_to_patch = self.cells_to_patch[1:]


    @command(category='Patch',
             description='Store the position of the washing bath',
             success_message='Cleaning path position stored')
    def store_cleaning_position(self) -> None:
        self.current_autopatcher.cleaning_bath_position = self.pipette_controller.calibrated_unit.position()

    @command(category='Patch',
                description='Store the position of the safe space',
                success_message='Safe space position stored')
    def store_safe_position(self) -> None:
        self.current_autopatcher.safe_position = self.pipette_controller.calibrated_unit.position()
        x,y = self.pipette_controller.calibrated_stage.position()
        z = float(self.pipette_controller.calibrated_unit.microscope.position()/5.0)
        self.current_autopatcher.safe_stage_position = [x,y,z]
        self.info(f'safe space position stored: {self.current_autopatcher.safe_position} and {self.current_autopatcher.safe_stage_position}')

    @command(category='Patch',
                description='Store the position of the home space',
                success_message='Home position stored')
    def store_home_position(self) -> None:
        # modifying to store the safe position of the and stage as well.
        self.current_autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        x,y = self.pipette_controller.calibrated_stage.position()
        z = float(self.pipette_controller.calibrated_unit.microscope.position()/5.0)
        self.current_autopatcher.home_stage_position = [x,y,z]
        self.info(f'safe home position stored: {self.current_autopatcher.home_position} and {self.current_autopatcher.home_stage_position}')


    @command(category='Patch',
                description='Store the position of the home and safe spaces',
                success_message='Home position stored')
    def store_calibration_positions(self) -> None:
        # modifying to store the safe position of the and stage as well.
        self.current_autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        angle = np.deg2rad(25)
        delta = -18000
        x_pip, y_pip,z_pip  = self.current_autopatcher.home_position
        self.current_autopatcher.safe_position = np.array([x_pip + delta*np.cos(angle), y_pip , z_pip +delta*np.sin(angle)])
        x,y = self.pipette_controller.calibrated_stage.position()
        z = float(self.pipette_controller.calibrated_unit.microscope.position()/5.0)
        self.current_autopatcher.home_stage_position = [x,y,z]
        self.current_autopatcher.safe_stage_position = self.current_autopatcher.home_stage_position
        self.info(f'safe home position stored: {self.current_autopatcher.home_position} and {self.current_autopatcher.home_stage_position}')
        self.info(f'safe space position stored: {self.current_autopatcher.safe_position} and {self.current_autopatcher.safe_stage_position}')
    
    # @command(category='Recording',
    #          description='Check to see if one of the patch methods is complete, whether failed or successful',
    #          success_message='Patch method complete')
    # def check_done(self) -> bool:
    #     self.done = self.current_autopatcher.done
    #     if self.done:
    #         self.current_autopatcher.done = False
    #     return self.done

    @command(category='Patch',
             description='Store the position of the rinsing bath',
             success_message='Rinsing bath position stored')
    def store_rinsing_position(self) -> None:
        self.current_autopatcher.rinsing_bath_position = self.pipette_controller.calibrated_unit.position()

    @command(category='Patch',
             description='clear all stored positions',
                success_message='All positions cleared')
    def clear_positions(self) -> None:
        self.current_autopatcher.cleaning_bath_position = None
        self.current_autopatcher.rinsing_bath_position = None
        self.current_autopatcher.home_position = None
        self.current_autopatcher.safe_position = None
        self.current_autopatcher.home_stage_position = None
        self.current_autopatcher.safe_stage_position = None
        self.info('All positions cleared')

    @blocking_command(category='Patch',
                      description='Clean the pipette (wash and rinse)',
                      task_description='Cleaning the pipette')
    def clean_pipette(self):
        self.execute(self.current_autopatcher.clean_pipette)

    @blocking_command(category='Patch',
                      description= 'Move to safe space',
                      task_description='Moving to safe space')
    def move_to_safe_space(self):
        self.execute(self.current_autopatcher.move_to_safe_space)

    @blocking_command(category='Patch',
                        description='Move to home position',
                        task_description='Moving to home position')
    def move_to_home_space(self):
        self.execute(self.current_autopatcher.move_to_home_space)

    @blocking_command(category='Patch',
                        description='Move the group down',
                        task_description='Moving the group down')
    def move_group_down(self):
        self.execute(self.current_autopatcher.move_group_down)

    @blocking_command(category='Patch',
                        description='Move the group up',
                        task_description='Moving the group up')
    def move_group_up(self):
        self.execute(self.current_autopatcher.move_group_up)
    @blocking_command(category='Patch',
                      description='Move the pipette up',
                      task_description='Moving the pipette up')
    def move_pipette_up(self):
        self.execute(self.current_autopatcher.move_pipette_up)

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