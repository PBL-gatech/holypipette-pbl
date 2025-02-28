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

        
    def add_demo(self, demo_file_path):
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


        #OBSERVATIONS
        #Curr Observations
        pressure_values = graph_values[:, 1].astype(np.uint8)
        resistance_values = graph_values[:, 2].astype(np.uint8)
        current_values = graph_values[:, 3]
        current_values_list = []
        for i in range(len(current_values)):
            current_values_list.append(json.loads(current_values[i]))
        current_values = np.array(current_values_list)

        voltage_values = graph_values[:, 4]
        voltage_values_list = []
        for i in range(len(voltage_values)):
            voltage_values_list.append(json.loads(voltage_values[i]))
        voltage_values = np.array(voltage_values_list)

        #Curr Camera
        camera_files = os.listdir(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames')
        camera_files.sort()
        frames_list = []
        curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[0]}')).tolist()
        
        for i in range(len(graph_values)):
            #print(f'Timestep: {i}')
            curr_timestamp = graph_values[i][0]
            if i == len(graph_values) - 1:
                timestamp_range = abs(curr_timestamp - graph_values[i-1][0])
            else:
                timestamp_range = abs(curr_timestamp - graph_values[i+1][0])

            closest_camera_index = 0
            camera_timestamps = np.array(camera_files[:, 4:-5], dtype=float)
            print(camera_timestamps)

            for j in range(len(camera_files)):
                camera_timestep = camera_files[j][4:-5]
                if abs(curr_timestamp - float(camera_timestep)) < timestamp_range:
                    closest_camera_index = j
                    break

            print(closest_camera_index)
            #curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[closest_camera_index]}')).tolist()
            curr_frame = np.array(Image.open(f'./experiments/Data/rig_recorder_data/{demo_file_path}/camera_frames/{camera_files[closest_camera_index]}').resize((85, 85)))
            frames_list.append(curr_frame)

        camera_frames = np.array(frames_list)


        #Next Observations
        graph_values_rolled = np.roll(graph_values, -1, axis=0)
        #graph_values_rolled[-1] = np.z
        next_pressure_values = graph_values_rolled[:, 1].astype(np.uint8)
        next_resistance_values = graph_values_rolled[:, 2].astype(np.uint8)
        next_current_values = graph_values_rolled[:, 3]
        next_current_values_list = []
        for i in range(len(next_current_values)):
            next_current_values_list.append(json.loads(next_current_values[i]))
        next_current_values = np.array(next_current_values_list)

        next_voltage_values = graph_values_rolled[:, 4]
        next_voltage_values_list = []
        for i in range(len(next_voltage_values)):
            next_voltage_values_list.append(json.loads(next_voltage_values[i]))
        next_voltage_values = np.array(next_voltage_values_list)

        next_camera_frames = np.roll(camera_frames, -1, axis=0)
        



        #ACTIONS
        #Low-level Actions
        movement_actions_list = list(np.diff(movement_values[:, 1:], axis=0))
        movement_actions_list.insert(0, list(np.zeros(movement_values.shape[1] - 1)))
        movement_actions = np.array(movement_actions_list)

        #High-level Actions
        log_messages = log_values['Message'].str.contains('Executing command')
        action_log_indices = np.where(log_messages == True)[0]
        action_logs = log_values.iloc[action_log_indices]
        action_logs['Full Time'] = (pd.to_datetime(action_logs['Time(HH:MM:SS)'] + '.' + action_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=5)).apply(lambda x: x.timestamp())

        curr_demo_actions_logs = action_logs[action_logs['Full Time'] > first_timestep]
        curr_demo_actions_logs = curr_demo_actions_logs[curr_demo_actions_logs['Full Time'] < last_timestep]
        filtered_action_logs = curr_demo_actions_logs.drop_duplicates()
        #print(filtered_action_logs['Message'].iloc[0])

        actions_list = []
        for i in range(len(graph_values)):
            curr_timestamp = graph_values[i][0]
            if i == len(graph_values) - 1:
                timestamp_range = abs(curr_timestamp - graph_values[i-1][0])
            else:
                timestamp_range = abs(curr_timestamp - graph_values[i+1][0])

            low_level_actions = list(np.zeros(movement_actions.shape[1]))
            high_level_action = hash('None')

            for j in range(len(movement_values)):
                if abs(curr_timestamp - movement_values[j][0]) < timestamp_range:
                    low_level_actions = list(movement_actions[j])
                    break
            for j in range(filtered_action_logs.shape[0]):
                if abs(curr_timestamp - filtered_action_logs.iloc[j]['Full Time']) < timestamp_range:
                    high_level_action = hash(filtered_action_logs.iloc[j]['Message'][19:])
                    break

            actions = low_level_actions
            actions.append(high_level_action)
            actions_list.append(actions)

        demo_actions = np.array(actions_list)

        #STATES
        states_list = []
        curr_state = list(movement_values[0][1:])
        for i in range(len(graph_values)):
            curr_timestamp = graph_values[i][0]
            if i == len(graph_values) - 1:
                timestamp_range = abs(curr_timestamp - graph_values[i-1][0])
            else:
                timestamp_range = abs(curr_timestamp - graph_values[i+1][0])

            for j in range(len(movement_values)):
                if abs(curr_timestamp - movement_values[j][0]) < timestamp_range:
                    curr_state = list(movement_values[j][1:])
                    break

            states_list.append(curr_state)

        demo_states = np.array(states_list)


        #DONES - Not Implemented
        dones = np.zeros(graph_values.shape[0])

        #REWARDS -Not Implemented
        rewards = np.zeros(graph_values.shape[0])
        


        #Add to dataset
        with h5py.File(f'experiments/Datasets/{self.dataset_name}', 'a') as hf:
            # Create a demo within the dataset_1
            demo_number = hf['data'].attrs['num_demos']
            demo = hf['data'].create_group(f'demo_{demo_number}')
            demo.attrs['num_samples'] = graph_values.shape[0]

            demo.create_dataset('actions', data=demo_actions)
            demo.create_dataset('dones', data=dones)
            demo.create_dataset('rewards', data=rewards)
            demo.create_dataset('states', data=demo_states)

            #Observations Group
            observations = demo.create_group('obs')
            observations.create_dataset('pressure', data=pressure_values)
            observations.create_dataset('resistance', data=resistance_values)
            observations.create_dataset('current', data=current_values)
            observations.create_dataset('voltage', data=voltage_values)
            observations.create_dataset('camera_image', data=camera_frames)

            #Next Observations Group
            next_observations = demo.create_group('next_obs')
            next_observations.create_dataset('pressure', data=next_pressure_values)
            next_observations.create_dataset('resistance', data=next_resistance_values)
            next_observations.create_dataset('current', data=next_current_values)
            next_observations.create_dataset('voltage', data=next_voltage_values)
            next_observations.create_dataset('camera_image', data=next_camera_frames)


            hf['data'].attrs['num_demos'] = hf['data'].attrs['num_demos'] + 1

        




