import threading
import h5py
import numpy as np


class HDF5Logger(threading.Thread):
    def __init__(self, recording_state_manager, stage, microscope, camera, amplifier, daq, pressure, pipette_controller, graphs):
        super().__init__()

        self.recording_state_manager = recording_state_manager
        self.is_recording = self.recording_state_manager.is_recording_enabled()

        self.stage = stage
        self.microscope = microscope
        self.camera = camera
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.pipette_controller = pipette_controller
        self.graphs = graphs

        self.logging_active = True

        self.start_stop_logging()

    def start_stop_logging(self):
        if self.logging_active:
            self.record_timestep()
            threading.Timer(1, self.start_stop_logging).start()

    def record_timestep(self):
        if self.recording_state_manager.is_recording_enabled():
            if self.is_recording == False:
                # New Recording Started
                with h5py.File('experiments/Datasets/dataset_1.hdf5', 'a') as hf:
                    # Create a demo within the dataset_1
                    print(hf['data'])
                    demo_number = hf['data'].attrs['num_demos']
                    demo = hf['data'].create_group(f'demo_{demo_number}')
                    demo.attrs['num_samples'] = 0

                    demo.create_dataset('actions', data=np.array([]), maxshape=(None,))
                    demo.create_dataset('dones', data=np.array([]))
                    demo.create_dataset('rewards', data=np.array([]))
                    demo.create_dataset('states', data=np.array([]))

                    #Observations Group
                    observations = demo.create_group('obs')
                    observations.create_dataset('pressure', data=np.array([]))
                    observations.create_dataset('resistance', data=np.array([]))
                    observations.create_dataset('current', data=np.array([]))
                    observations.create_dataset('voltage', data=np.array([]))
                    observations.create_dataset('camera_image', data=np.array([]))

                    #Next Observations Group
                    next_observations = demo.create_group('next_obs')
                    next_observations.create_dataset('pressure', data=np.array([]))
                    next_observations.create_dataset('resistance', data=np.array([]))
                    next_observations.create_dataset('current', data=np.array([]))
                    next_observations.create_dataset('voltage', data=np.array([]))
                    next_observations.create_dataset('camera_image', data=np.array([]))


                    hf['data'].attrs['num_demos'] = hf['data'].attrs['num_demos'] + 1


            self.is_recording = True

            #OBSERVATIONS
            # Readings (pressure, resistance, current, voltage)
            pressure = self.pressure.get_pressure()
            resistance = self.daq.resistance()
            current = None
            if len(self.graphs.lastrespData) > 0:
                current = self.graphs.lastrespData[1, :]
            voltage = None
            if len(self.graphs.lastReadData) > 0:
                voltage = self.graphs.lastReadData[1, :]

            #Images
            camera_image = self.camera.raw_snap()
            #dataset = f.create_dataset("image_data", data=image_data, compression="gzip", complevel=9)


            # STATE
            # Position (microscope, stage, pipette)
            microscope_true_z_pos = self.microscope.position()

            stage_pos = self.stage.position()
            stage_pos_x = stage_pos[0]
            stage_pos_y = stage_pos[1]

            pipette_pos = self.pipette_controller.calibrated_unit.position()
            pipette_pos_x = pipette_pos[0]
            pipette_pos_y = pipette_pos[1]
            pipette_pos_z = pipette_pos[2]

            state_vector = [microscope_true_z_pos, stage_pos_x, stage_pos_y, pipette_pos_x, pipette_pos_y, pipette_pos_z]

            with h5py.File('experiments/Datasets/dataset_1.hdf5', 'a') as hf:
                # Create a demo within the dataset_1
                print(hf['data'])
                demo_number = hf['data'].attrs['num_demos'] - 1
                curr_demo = f'demo_{demo_number}'

                #demo.create_dataset('actions', data=[])
                #demo.create_dataset('dones', data=[])
                #demo.create_dataset('rewards', data=[])
                print(hf['data'][curr_demo])
                print(hf['data'][curr_demo]['states'])
                hf['data'][curr_demo]['states'].append(state_vector)

                #Observations Group
                #observations = demo.create_group('obs')
                #observations.create_dataset('pressure', data=[])
                #observations.create_dataset('resistance', data=[])
                #observations.create_dataset('current', data=[])
                #observations.create_dataset('voltage', data=[])
                #observations.create_dataset('camera_image', data=[])

                #Next Observations Group
                #next_observations = demo.create_group('next_obs')
                #next_observations.create_dataset('pressure', data=[])
                #next_observations.create_dataset('resistance', data=[])
                #next_observations.create_dataset('current', data=[])
                #next_observations.create_dataset('voltage', data=[])
                #next_observations.create_dataset('camera_image', data=[])


            print('Recording')
        else:
            if self.is_recording == True:
                #Recording Stopped
                with h5py.File('experiments/Datasets/dataset_1.hdf5', 'a') as hf:
                    demo_number = hf['data'].attrs['num_demos'] - 1
                    curr_demo = f'demo_{demo_number}'

                    hf['data'][curr_demo].attrs['num_samples'] = hf['data'][curr_demo]['states'].shape[0]

            self.is_recording = False
            print('Not Recording')

    
    def close(self):
        self.logging_active = False
