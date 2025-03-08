import os
import h5py
import pandas as pd
import numpy as np
import datetime
import json
from PIL import Image

class DatasetBuilder():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        if dataset_name not in os.listdir('./experiments/Datasets'):
            with h5py.File(f'experiments/Datasets/{dataset_name}', 'w') as hf:
                group = hf.create_group('data')
                group.attrs['num_demos'] = 0

        
    def add_demo(self, demo_file_path, record_to_file=False):
        include_camera = True

        graph_recording_file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv')
        graph_values = pd.read_csv(graph_recording_file, delimiter=';').to_numpy()
        movement_recording_file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv')
        movement_values = pd.read_csv(movement_recording_file, delimiter=';').to_numpy()
        log_file = open(f'./experiments/Data/log_data/logs_{demo_file_path[:10]}.csv')
        log_values = pd.read_csv(log_file, on_bad_lines='skip')

        first_timestep = graph_values[0][0] - 1
        last_timestep = graph_values[-1][0] + 1
        print(first_timestep)
        print(last_timestep)

        # print('movement values:')
        # print(movement_values)
        # print('graph values:')
        # print(graph_values)
        # print(graph_values.shape)


        #DONES
        #"Cell detected: True" in message
        #Controller/patch.py has success messages for different methods
        #   -patch method has all of the different methods, will record when the button is clicked to call this method
        #Others in patch.py in gui
        #Start recording when patch method started
            #Want to separate into datasets for:
                #cell hunt
                    #"Cell detected: True"
                #gigaseal
                    #"Seal successful"
                #break-in
                    #"Successful break-in"

        #Get Hunt Cell Start Log Message
        hunt_cell_start_log_messages = log_values['Message'].str.contains('Hunting for cell')
        hunt_cell_start_log_indices = np.where(hunt_cell_start_log_messages == True)[0]
        hunt_cell_start_logs = log_values.iloc[hunt_cell_start_log_indices]
        hunt_cell_start_logs['Full Time'] = (pd.to_datetime(hunt_cell_start_logs['Time(HH:MM:SS)'] + '.' + hunt_cell_start_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=5)).apply(lambda x: x.timestamp())


        curr_demo_hunt_cell_start_logs = hunt_cell_start_logs[hunt_cell_start_logs['Full Time'] > first_timestep]
        curr_demo_hunt_cell_start_logs = curr_demo_hunt_cell_start_logs[curr_demo_hunt_cell_start_logs['Full Time'] < last_timestep]
        filtered_hunt_cell_start_logs = curr_demo_hunt_cell_start_logs.drop_duplicates()

        #Get Hunt Cell Done Log Message
        hunt_cell_done_log_messages = log_values['Message'].str.contains('Cell detected: True')
        hunt_cell_done_log_indices = np.where(hunt_cell_done_log_messages == True)[0]
        hunt_cell_done_logs = log_values.iloc[hunt_cell_done_log_indices]
        hunt_cell_done_logs['Full Time'] = (pd.to_datetime(hunt_cell_done_logs['Time(HH:MM:SS)'] + '.' + hunt_cell_done_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=5)).apply(lambda x: x.timestamp())

        curr_demo_hunt_cell_done_logs = hunt_cell_done_logs[hunt_cell_done_logs['Full Time'] > first_timestep]
        curr_demo_hunt_cell_done_logs = curr_demo_hunt_cell_done_logs[curr_demo_hunt_cell_done_logs['Full Time'] < last_timestep]
        filtered_hunt_cell_done_logs = curr_demo_hunt_cell_done_logs.drop_duplicates()

        #Truncate Graph Values to be Between Hunt Cell Start and Done Logs
        first_graph_value_index = -1
        last_graph_value_index = -1
        cont = True
        for i in range(len(graph_values)):
            if cont:
                curr_timestamp = graph_values[i][0]
                if i == len(graph_values) - 1:
                    timestamp_range = abs(curr_timestamp - graph_values[i-1][0])
                else:
                    timestamp_range = abs(curr_timestamp - graph_values[i+1][0])

                #Check if first graph value
                if first_graph_value_index == -1:
                    for j in range(filtered_hunt_cell_start_logs.shape[0]):
                        if abs(curr_timestamp - filtered_hunt_cell_start_logs.iloc[j]['Full Time']) < timestamp_range:
                            first_graph_value_index = i
                            break

                #Check if last graph value
                for j in range(filtered_hunt_cell_done_logs.shape[0]):
                    if abs(curr_timestamp - filtered_hunt_cell_done_logs.iloc[j]['Full Time']) < timestamp_range:
                        last_graph_value_index = i + 1
                        cont = False
                        break


        dones = np.zeros(last_graph_value_index - first_graph_value_index)
        dones[-1] = 1

        truncated_graph_values = graph_values[first_graph_value_index:last_graph_value_index]

        #REWARDS -Not Using
        #rewards = np.copy(done_values) * 100


        #OBSERVATIONS
        #Curr Observations

        #Pressure
        pressure_values = truncated_graph_values[:, 1].astype(np.uint8)

        #Resistance
        resistance_values = truncated_graph_values[:, 2].astype(np.uint8)

        #Current
        current_values = truncated_graph_values[:, 3]
        current_values_list = []
        max_current_list_len = 0
        for i in range(len(current_values)):
            curr_current_values = json.loads(current_values[i])
            current_values_list.append(curr_current_values)
            if len(curr_current_values) > max_current_list_len:
                max_current_list_len = len(curr_current_values)

        for i in range(len(current_values_list)):
            while (len(current_values_list[i])) < max_current_list_len:
                current_values_list[i].append(current_values_list[i][-1])

        current_values = np.array(current_values_list)

        #Voltage
        voltage_values = truncated_graph_values[:, 4]
        voltage_values_list = []
        max_voltage_list_len = 0
        for i in range(len(voltage_values)):
            curr_voltage_values = json.loads(voltage_values[i])
            voltage_values_list.append(curr_voltage_values)
            if len(curr_voltage_values) > max_voltage_list_len:
                max_voltage_list_len = len(curr_voltage_values)

        for i in range(len(voltage_values_list)):
            while (len(voltage_values_list[i])) < max_voltage_list_len:
                voltage_values_list[i].append(voltage_values_list[i][-1])

        voltage_values = np.array(voltage_values_list)

        #Pipette and Stage Positions
        stage_positions = []
        pipette_positions = []
        curr_stage_position = list(movement_values[0][1:4])
        curr_pipette_position = list(movement_values[0][4:])
        for i in range(len(truncated_graph_values)):
            curr_timestamp = truncated_graph_values[i][0]
            if i == len(truncated_graph_values) - 1:
                timestamp_range = abs(curr_timestamp - truncated_graph_values[i-1][0])
            else:
                timestamp_range = abs(curr_timestamp - truncated_graph_values[i+1][0])

            for j in range(len(movement_values)):
                if abs(curr_timestamp - movement_values[j][0]) < timestamp_range:
                    curr_stage_position = list(movement_values[j][1:4])
                    curr_pipette_position = list(movement_values[j][4:])
                    break

            stage_positions.append(curr_stage_position)
            pipette_positions.append(curr_pipette_position)

        stage_positions = np.array(stage_positions)
        pipette_positions = np.array(pipette_positions)

        #Camera
        if include_camera:
            camera_files = os.listdir(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames')
            camera_files.sort()
            frames_list = []
            curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[0]}')).tolist()
            
            for i in range(len(truncated_graph_values)):
                #print(f'Timestep: {i}')
                curr_timestamp = truncated_graph_values[i][0]
                if i == len(truncated_graph_values) - 1:
                    timestamp_range = abs(curr_timestamp - truncated_graph_values[i-1][0])
                else:
                    timestamp_range = abs(curr_timestamp - truncated_graph_values[i+1][0])

                closest_camera_index = 0

                for j in range(len(camera_files)):
                    camera_file = camera_files[j]
                    underscore_index = camera_file.find("_")
                    last_period_index = camera_file.rfind(".")
                    camera_timestep = camera_files[j][underscore_index+1:last_period_index]
                    if abs(curr_timestamp - float(camera_timestep)) < timestamp_range:
                        closest_camera_index = j
                        break

                #print(closest_camera_index)
                #curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[closest_camera_index]}')).tolist()
                curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[closest_camera_index]}').resize((85, 85)))[:, :, 0]
                frames_list.append(curr_frame)

            camera_frames = np.array(frames_list)

        


        #Next Observations - Not using
        # graph_values_rolled = np.roll(truncated_graph_values, -1, axis=0)
        # next_pressure_values = graph_values_rolled[:, 1].astype(np.uint8)
        # next_resistance_values = graph_values_rolled[:, 2].astype(np.uint8)

        # next_current_values = np.roll(current_values, -1, axis=0)

        # next_voltage_values = np.roll(voltage_values, -1, axis=0)

        # if include_camera:
        #     next_camera_frames = np.roll(camera_frames, -1, axis=0)
        



        #ACTIONS
        #Low-level Actions
        movement_actions_list = list(np.diff(movement_values[:, 1:], axis=0))
        movement_actions_list.insert(0, list(np.zeros(movement_values.shape[1] - 1)))
        movement_actions = np.array(movement_actions_list)

        #High-level Actions (Not used for hunt cell datasets)
        # log_messages = log_values['Message'].str.contains('Executing command')
        # action_log_indices = np.where(log_messages == True)[0]
        # action_logs = log_values.iloc[action_log_indices]
        # action_logs['Full Time'] = (pd.to_datetime(action_logs['Time(HH:MM:SS)'] + '.' + action_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=5)).apply(lambda x: x.timestamp())

        # curr_demo_actions_logs = action_logs[action_logs['Full Time'] > first_timestep]
        # curr_demo_actions_logs = curr_demo_actions_logs[curr_demo_actions_logs['Full Time'] < last_timestep]
        # filtered_action_logs = curr_demo_actions_logs.drop_duplicates()

        actions_list = []
        for i in range(len(truncated_graph_values)):
            curr_timestamp = truncated_graph_values[i][0]
            if i == len(truncated_graph_values) - 1:
                timestamp_range = abs(curr_timestamp - truncated_graph_values[i-1][0])
            else:
                timestamp_range = abs(curr_timestamp - truncated_graph_values[i+1][0])

            low_level_actions = list(np.zeros(movement_actions.shape[1]))
            #high_level_action = hash('None')

            for j in range(len(movement_values)):
                if abs(curr_timestamp - movement_values[j][0]) < timestamp_range:
                    low_level_actions = list(movement_actions[j])
                    break
            # for j in range(filtered_action_logs.shape[0]):
            #     if abs(curr_timestamp - filtered_action_logs.iloc[j]['Full Time']) < timestamp_range:
            #         high_level_action = hash(filtered_action_logs.iloc[j]['Message'][19:])
            #         #print(high_level_action)
            #         break

            actions = low_level_actions
            #actions.append(high_level_action)
            actions_list.append(actions)

        demo_actions = np.array(actions_list)

        #STATES - not using (prev states of pipette and stage position now observations)
        # states_list = []
        # curr_state = list(movement_values[0][1:])
        # for i in range(len(truncated_graph_values)):
        #     curr_timestamp = truncated_graph_values[i][0]
        #     if i == len(truncated_graph_values) - 1:
        #         timestamp_range = abs(curr_timestamp - truncated_graph_values[i-1][0])
        #     else:
        #         timestamp_range = abs(curr_timestamp - truncated_graph_values[i+1][0])

        #     for j in range(len(movement_values)):
        #         if abs(curr_timestamp - movement_values[j][0]) < timestamp_range:
        #             curr_state = list(movement_values[j][1:])
        #             break

        #     states_list.append(curr_state)

        # demo_states = np.array(states_list)


        
        


        #Add to dataset
        if record_to_file:
            with h5py.File(f'experiments/Datasets/{self.dataset_name}', 'a') as hf:
                # Create a demo within the dataset_1
                demo_number = hf['data'].attrs['num_demos']
                demo = hf['data'].create_group(f'demo_{demo_number}')
                demo.attrs['num_samples'] = truncated_graph_values.shape[0]

                demo.create_dataset('actions', data=demo_actions)
                demo.create_dataset('dones', data=dones)
                #demo.create_dataset('rewards', data=rewards)
                #demo.create_dataset('states', data=demo_states)

                #Observations Group
                observations = demo.create_group('obs')
                observations.create_dataset('pressure', data=pressure_values)
                observations.create_dataset('resistance', data=resistance_values)
                observations.create_dataset('current', data=current_values)
                observations.create_dataset('voltage', data=voltage_values)
                observations.create_dataset('stage_positions', data=stage_positions)
                observations.create_dataset('pipette_positions', data=pipette_positions)
                observations.create_dataset('camera_image', data=camera_frames)

                #Next Observations Group
                # next_observations = demo.create_group('next_obs')
                # next_observations.create_dataset('pressure', data=next_pressure_values)
                # next_observations.create_dataset('resistance', data=next_resistance_values)
                # next_observations.create_dataset('current', data=next_current_values)
                # next_observations.create_dataset('voltage', data=next_voltage_values)
                # next_observations.create_dataset('camera_image', data=next_camera_frames)


                hf['data'].attrs['num_demos'] = hf['data'].attrs['num_demos'] + 1



    def convert_graph_recording_csv_to_new_format(self, demo_file_path):
        graph_recording_file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv')
        graph_values = pd.read_csv(graph_recording_file, delimiter=':') #.to_numpy()

        #converted_graph_values_df = pd.DataFrame(columns=['timestamp', 'pressure', 'resistance', 'current', 'voltage'])

        converted_file_strings = ['timestamp;pressure;resistance;current;voltage\n']
        
        for i in range(len(graph_values)):
            timestamp = graph_values.iloc[i][1][:-10]
            pressure = graph_values.iloc[i][2][:-12]
            resistance = graph_values.iloc[i][3][:-9]
            current = graph_values.iloc[i][4][:-8]
            voltage = graph_values.iloc[i][5]

            graph_datapoint_string = f'{timestamp};{pressure};{resistance};{current};{voltage}\n'

            converted_file_strings.append(graph_datapoint_string)

        #print(converted_file_strings[0])

        with open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv', 'w') as f:
            for i in range(len(converted_file_strings)):
                f.write(converted_file_strings[i])


        file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv')
        graph_values_new = pd.read_csv(file, delimiter=';')

        print(graph_values_new)

    def convert_movement_recording_csv_to_new_format(self, demo_file_path):
        movement_recording_file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv')
        movement_values = pd.read_csv(movement_recording_file, delimiter=':') #.to_numpy()

        converted_graph_values_df = pd.DataFrame(columns=['timestamp', 'pressure', 'resistance', 'current', 'voltage'])

        converted_file_strings = ['timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z\n']
        
        for i in range(len(movement_values)):
            timestamp = movement_values.iloc[i][1][:-6]
            st_x = movement_values.iloc[i][2][:-6]
            st_y = movement_values.iloc[i][3][:-6]
            st_z = movement_values.iloc[i][4][:-6]
            pi_x = movement_values.iloc[i][5][:-5]
            pi_y = movement_values.iloc[i][6][:-5]
            pi_z = movement_values.iloc[i][7]

            movement_datapoint_string = f'{timestamp};{st_x};{st_y};{st_z};{pi_x};{pi_y};{pi_z}\n'

            converted_file_strings.append(movement_datapoint_string)

        #print(converted_file_strings[0])

        with open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv', 'w') as f:
            for i in range(len(converted_file_strings)):
                f.write(converted_file_strings[i])


        file = open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv')
        movement_values_new = pd.read_csv(file, delimiter=';')

        print(movement_values_new)
        




