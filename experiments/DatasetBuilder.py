import os
import h5py
import pandas as pd
import numpy as np
import datetime
import json
from PIL import Image


ATL_TO_UTC_TIME_DELTA = 4 #March 9 - Nov 1: 4 hours, otherwise 5 hours


class DatasetBuilder():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.zero_values = True

        if dataset_name not in os.listdir('experiments/Datasets'):
            with h5py.File(f'experiments/Datasets/{dataset_name}', 'w') as hf:
                group = hf.create_group('data')
                group.attrs['num_demos'] = 0



    def convert_graph_recording_csv_to_new_format(self, demo_file_path):
        graph_recording_file = open(f'experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv')
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

        with open(f'experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv', 'w') as f:
            for i in range(len(converted_file_strings)):
                f.write(converted_file_strings[i])


        file = open(f'experiments/Data/rig_recorder_data/{demo_file_path}/graph_recording.csv')
        graph_values_new = pd.read_csv(file, delimiter=';')

        print(graph_values_new)

    def convert_movement_recording_csv_to_new_format(self, demo_file_path):
        movement_recording_file = open(f'experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv')
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

        with open(f'experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv', 'w') as f:
            for i in range(len(converted_file_strings)):
                f.write(converted_file_strings[i])


        file = open(f'experiments/Data/rig_recorder_data/{demo_file_path}/movement_recording.csv')
        movement_values_new = pd.read_csv(file, delimiter=';')

        print(movement_values_new)

    def load_experiment_data(self, rig_recorder_data_folder):
        """
        Returns graph and movement values from respective files in the rig recorder folder
            -New rig recorder folder created upon every execution when there is a recording during that execution
                -Can be multiple recordings in a single folder 
                -Can be multiple neuron hunt attempts in a single recording
        Returns the logs from the log file associated with the rig recorder folder
            -New log file created by day (single log file for whole day)
        Author(s): Kaden Stillwagon

        args:
            rig_recorder_data_folder (string): the filename of the rig recorder folder containing the experiment data

        Returns:
            graph_values (pd.Dataframe): dataframe containing the graph values within the rig_recorder_data_folder
            movement_values (pd.Dataframe): dataframe containing the movement values within the rig_recorder_data_folder
            log_values (pd.Dataframe): dataframe containing the log messages from the log file that are within the rig_recorder_data_folder's time frame

        """
        graph_recording_file = open(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/graph_recording.csv')
        graph_values = pd.read_csv(graph_recording_file, delimiter=';').to_numpy()
        movement_recording_file = open(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/movement_recording.csv')
        movement_values = pd.read_csv(movement_recording_file, delimiter=';').to_numpy()
        log_file = open(f'experiments/Data/log_data/logs_{rig_recorder_data_folder[:10]}.csv')
        log_values = pd.read_csv(log_file, on_bad_lines='skip')

        return graph_values, movement_values, log_values
    
    def get_timestamps_for_all_experiment_recordings(self, log_values, experiment_first_timestamp, experiment_last_timestamp):
        """
        Returns a list of tuples representing the starting and ending timestamps of each recording in the rig recorder folder
            1) Finds each of the "Recording started" and "Recording stopped" messages in the log file 
            2) Filters messages to only includes those that occur within the time range of the current rig recorder folder
            3) Associates the "Recording started" and "Recording stopped" messages for a single recording to get the start and end times for the recording
        Author(s): Kaden Stillwagon

        args:
            log_values (pd.Dataframe): dataframe containing the log messages from the log file that are within the rig_recorder_data_folder's time frame

        Returns:
            recording_time_ranges (list): list of tuples representing the starting and ending times of each recording in the current rig recorder folder

        """
        #Get Recording Started Messages within Experiment
        recording_started_log_messages = log_values['Message'].str.contains('Recording started')
        recording_started_log_indices = np.where(recording_started_log_messages == True)[0]
        recording_started_logs = log_values.iloc[recording_started_log_indices].copy()
        recording_started_logs.loc[:, 'Full Time'] = (pd.to_datetime(recording_started_logs['Time(HH:MM:SS)'] + '.' + recording_started_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)).apply(lambda x: x.timestamp())

        curr_experiment_recording_started_logs = recording_started_logs[recording_started_logs['Full Time'] > experiment_first_timestamp]
        curr_experiment_recording_started_logs = curr_experiment_recording_started_logs[curr_experiment_recording_started_logs['Full Time'] < experiment_last_timestamp]
        filtered_recording_started_logs = curr_experiment_recording_started_logs.drop_duplicates()

        #Get Recording Ended Messages within Experiment
        recording_ended_log_messages = log_values['Message'].str.contains('Recording stopped')
        recording_ended_log_indices = np.where(recording_ended_log_messages == True)[0]
        recording_ended_logs = log_values.iloc[recording_ended_log_indices].copy()
        recording_ended_logs.loc[:, 'Full Time'] = (pd.to_datetime(recording_ended_logs['Time(HH:MM:SS)'] + '.' + recording_ended_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)).apply(lambda x: x.timestamp())

        curr_experiment_recording_ended_logs = recording_ended_logs[recording_ended_logs['Full Time'] > experiment_first_timestamp]
        curr_experiment_recording_ended_logs = curr_experiment_recording_ended_logs[curr_experiment_recording_ended_logs['Full Time'] < experiment_last_timestamp]
        filtered_recording_ended_logs = curr_experiment_recording_ended_logs.drop_duplicates()

        #Get Individual Recording Start/End Timestamps
        recording_start_times = []
        for recording_started_message_timestamp in filtered_recording_started_logs['Full Time']:
            recording_start_times.append(recording_started_message_timestamp)

        recording_time_ranges = []
        for recording_ended_message_timestamp in filtered_recording_ended_logs['Full Time']:
            for i in range(len(recording_start_times)):
                if i < len(recording_start_times) - 1:
                    if recording_ended_message_timestamp > recording_start_times[i] and recording_ended_message_timestamp < recording_start_times[i+1]:
                        recording_time_ranges.append((recording_start_times[i], recording_ended_message_timestamp))
                else:
                    if recording_ended_message_timestamp > recording_start_times[i] and recording_ended_message_timestamp < experiment_last_timestamp:
                        recording_time_ranges.append((recording_start_times[i], recording_ended_message_timestamp))

        return recording_time_ranges
    
    def get_timestamps_for_all_successful_hunt_cell_attempts(self, log_values, recording_timestamp_ranges):
        """
        Returns a list of tuples representing the starting and ending timestamps of each successful hunt cell attempt in each recording,
        using "Initial resistance:" as the start indicator and "Cell detected: True" as the end indicator.
            1) Finds each of the "Initial resistance:" and "Cell detected: True" messages in the log file 
            2) Filters messages to only include those that occur within the time range of the current recording
            3) Associates the "Initial resistance:" and "Cell detected: True" messages for a single attempt to get the start and end times
        Author(s): Kaden Stillwagon

        args:
            log_values (pd.DataFrame): dataframe containing the log messages from the log file that are within the rig_recorder_data_folder's time frame
            recording_timestamp_ranges (list): list of tuples representing the starting and ending times of each recording in the current rig recorder folder

        Returns:
            successful_hunt_cell_time_ranges (list): list of tuples representing the starting and ending times of each successful hunt cell attempt in the current rig recorder folder
        """

        successful_hunt_cell_time_ranges = []  # Not separating by recordings now, but could easily

        for recording_timestamps in recording_timestamp_ranges:
            start_timestamp = recording_timestamps[0]
            end_timestamp = recording_timestamps[1]

            # ----------------------------------------------------------
            # Get "Initial resistance:" messages within the recording (start events)
            # ----------------------------------------------------------
            initial_resistance_mask = log_values['Message'].str.contains('Initial resistance:')
            initial_resistance_indices = np.where(initial_resistance_mask == True)[0]
            initial_resistance_logs = log_values.iloc[initial_resistance_indices].copy()
            initial_resistance_logs.loc[:, 'Full Time'] = (
                pd.to_datetime(
                    initial_resistance_logs['Time(HH:MM:SS)'] + '.' + initial_resistance_logs['Time(ms)'].astype(str),
                    format='%Y-%m-%d %H:%M:%S.%f'
                ) + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())

            curr_recording_initial_resistance_logs = initial_resistance_logs[initial_resistance_logs['Full Time'] > start_timestamp]
            curr_recording_initial_resistance_logs = curr_recording_initial_resistance_logs[curr_recording_initial_resistance_logs['Full Time'] < end_timestamp]
            filtered_initial_resistance_logs = curr_recording_initial_resistance_logs.drop_duplicates()
            print(f"Filtered Initial Resistance Logs: {len(filtered_initial_resistance_logs)} found between {start_timestamp} and {end_timestamp}")

            # ----------------------------------------------------------
            # Get "Cell detected: True" messages within the recording (end events)
            # ----------------------------------------------------------
            cell_detected_mask = log_values['Message'].str.contains('Cell detected: True')
            cell_detected_indices = np.where(cell_detected_mask == True)[0]
            cell_detected_logs = log_values.iloc[cell_detected_indices].copy()
            cell_detected_logs.loc[:, 'Full Time'] = (
                pd.to_datetime(
                    cell_detected_logs['Time(HH:MM:SS)'] + '.' + cell_detected_logs['Time(ms)'].astype(str),
                    format='%Y-%m-%d %H:%M:%S.%f'
                ) + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())

            curr_recording_cell_detected_logs = cell_detected_logs[cell_detected_logs['Full Time'] > start_timestamp]
            curr_recording_cell_detected_logs = curr_recording_cell_detected_logs[curr_recording_cell_detected_logs['Full Time'] < end_timestamp]
            filtered_cell_detected_logs = curr_recording_cell_detected_logs.drop_duplicates()

            # ----------------------------------------------------------
            # Get Individual Attempt Start/End Timestamps by pairing
            # ----------------------------------------------------------
            initial_resistance_start_times = []
            for initial_resistance_message_timestamp in filtered_initial_resistance_logs['Full Time']:
                initial_resistance_start_times.append(initial_resistance_message_timestamp)

            for cell_detected_message_timestamp in filtered_cell_detected_logs['Full Time']:
                for i in range(len(initial_resistance_start_times)):
                    if i < len(initial_resistance_start_times) - 1:
                        if cell_detected_message_timestamp > initial_resistance_start_times[i] and cell_detected_message_timestamp < initial_resistance_start_times[i+1]:
                            successful_hunt_cell_time_ranges.append((initial_resistance_start_times[i], cell_detected_message_timestamp))
                    else:
                        if cell_detected_message_timestamp > initial_resistance_start_times[i] and cell_detected_message_timestamp < end_timestamp:
                            successful_hunt_cell_time_ranges.append((initial_resistance_start_times[i], cell_detected_message_timestamp))

            # No additional adjustments are made as we now simply use the "Initial resistance:" message as the start event

        return successful_hunt_cell_time_ranges

    def truncate_graph_values(self, graph_values, first_timestamp, last_timestamp):
        '''
        Finds the rows of graph values dataframe that have the closest timestamps to the first and last timestamp of the attempt
        Truncates the graph values to be only between these rows and returns this truncated dataframe of graph values

        Author(s): Kaden Stillwagon

        args:
            graph_values (np.array): array containing the graph values within the rig_recorder_data_folder
            first_timestamp (float): timestamp representing the start of the neuron hunt attempt
            last_timestamp (float): timestamp representing the end of the neuron hunt attempt

        Returns:
            truncated_graph_values (np.array): array containing the graph values, truncated to only values within the first and last timestamps
        '''
        first_graph_value_index = -1
        last_graph_value_index = -1
        cont = True
        for i in range(len(graph_values)):
            if cont:
                curr_graph_timestamp = graph_values[i][0]
                if i == len(graph_values) - 1:
                    timestamp_range = abs(curr_graph_timestamp - graph_values[i-1][0])
                else:
                    timestamp_range = abs(curr_graph_timestamp - graph_values[i+1][0])

                #Check if first graph value
                if first_graph_value_index == -1:
                    if abs(curr_graph_timestamp - first_timestamp) < timestamp_range:
                        first_graph_value_index = i
                        continue

                #Check if last graph value
                if abs(curr_graph_timestamp - last_timestamp) < timestamp_range:
                    last_graph_value_index = i + 1
                    cont = False
                    break
        
        truncated_graph_values = graph_values[first_graph_value_index:last_graph_value_index]

        return truncated_graph_values
    
    def associate_attempt_movement_and_graph_values(self, attempt_graph_values, movement_values):
        '''
        Finds the rows of the movement values that have the closest timestamps to the graph value rows' timestamps
        Returns an array of movement value rows associated with each graph value row

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values within the current attempt
            movement_values (np.array): timestamp representing the start of the neuron hunt attempt

        Returns:
            attempt_movement_values (np.array): array containing the movement value rows associated with the graph value rows
        '''
        attempt_movement_values = []
        
        last_index = 0
        for i in range(len(attempt_graph_values)):
            target_timestamp = attempt_graph_values[i][0]
            if i == len(attempt_graph_values) - 1:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i-1][0])
            else:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i+1][0])

            valid_movement_indices = []
            for j in range(last_index, len(movement_values)):
                if abs(target_timestamp - movement_values[j][0]) < timestamp_range:
                    valid_movement_indices.append(j)
                else:
                    if len(valid_movement_indices) > 0:
                        break

            min_timestamp_diff = 1000000
            min_timestamp_diff_indice = 0
            for j in valid_movement_indices:
                timestamp_diff = abs(target_timestamp - movement_values[j][0])
                if timestamp_diff < min_timestamp_diff:
                    min_timestamp_diff = timestamp_diff
                    min_timestamp_diff_indice = j
       
            attempt_movement_values.append(movement_values[min_timestamp_diff_indice])
            last_index = min_timestamp_diff_indice - 1

        attempt_movement_values = np.array(attempt_movement_values)

        return attempt_movement_values

    def get_attempt_dones(self, attempt_graph_values):
        '''
        Simply creates a numpy array of zeros with a one at the end since only success attempts are looked at
        Includes commentary for how to expand this to methods other than hunt cell

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            dones (np.array): numpy array containing done marker values
        '''
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


        dones = np.zeros(len(attempt_graph_values))
        dones[-1] = 1

        return dones
    

    def get_attempt_pressure_values(self, attempt_graph_values):
        '''
        Returns pressure values for current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            pressure_values (np.array): numpy array containing pressure values for the current attempt
        '''
        pressure_values = attempt_graph_values[:, 1].astype(np.float64)

        return pressure_values
    

    def get_attempt_resistance_values(self, attempt_graph_values):
        '''
        Returns resistance values for current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            resistance_values (np.array): numpy array containing resistance values for the current attempt
        '''
        resistance_values = attempt_graph_values[:, 2].astype(np.float64)
        if self.zero_values:
            # normalize resistance values to start at zero
            resistance_values[:] -= resistance_values[0]

        return resistance_values
    

    def get_attempt_current_values(self, attempt_graph_values):
        '''
        Returns current values for current attempt
            -Ensures all current lists are of equal length (or else cannot be put into numpy array)

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            current_values (np.array): numpy array containing current values for the current attempt
        '''
        current_values = attempt_graph_values[:, 3]
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

        return current_values


    def get_attempt_voltage_values(self, attempt_graph_values):
        '''
        Returns voltage values for current attempt
            -Ensures all voltage lists are of equal length (or else cannot be put into numpy array)

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            voltage_values (np.array): numpy array containing voltage values for the current attempt
        '''
        voltage_values = attempt_graph_values[:, 4]
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

        return voltage_values
    
    def get_attempt_stage_positions(self, attempt_movement_values):
        '''
        Returns stage positions for current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            state_positions (np.array): numpy array containing stage positions for the current attempt

        '''
        stage_positions = attempt_movement_values[:, 1:].astype(np.float64) #Note: for hunt cell, only need z-coordinate for stage positions (must use [:, 1:] to get all 3 stage coordinates)
        # stage_positions = attempt_movement_values[:, 3].astype(np.float64) #Note: for hunt cell, only need z-coordinate for stage positions (must use [:, 1:] to get all 3 stage coordinates)
        if self.zero_values:
            # normalize stage positions to start at zero for all 3 axes
            # stage_positions[:] -= stage_positions[0] # z-axis only
            stage_positions[:, 0] -= stage_positions[0, 0]  # x-axis
            stage_positions[:, 1] -= stage_positions[0, 1]  # y-axis
            stage_positions[:, 2] -= stage_positions[0, 2]  # z-axis
        
        return stage_positions
    
    def get_attempt_pipette_positions(self, attempt_movement_values):
        '''
        Returns pipette positions for current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            pipette_positions (np.array): numpy array containing pipette positions for the current attempt
        '''
        pipette_positions = attempt_movement_values[:, 4:].astype(np.float64)
        if self.zero_values: 
            # normalize pipette positions to start at zero for all 3 axes
            pipette_positions[:, 0] -= pipette_positions[0, 0]  # x-axis
            pipette_positions[:, 1] -= pipette_positions[0, 1]
            pipette_positions[:, 2] -= pipette_positions[0, 2]  # z-axis


        
        return pipette_positions
    
    def get_attempt_camera_frames(self, rig_recorder_data_folder, attempt_graph_values):
        '''
        Returns camera frames for current attempt
            -Associates each camera frame with a row of the graph values
            -Converts camera frame into numpy array

        Author(s): Kaden Stillwagon

        args:
            rig_recorder_data_folder (string): the filename of the rig recorder folder containing the experiment data
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            camera_frames (np.array): numpy array containing camera frames for the current attempt
        '''
        camera_files = os.listdir(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/camera_frames')
        camera_files.sort()
        frames_list = []
        curr_frame = np.array(Image.open(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/camera_frames/{camera_files[0]}')).tolist()
        
        last_index = 0
        for i in range(len(attempt_graph_values)):
            #print(f'Timestep: {i}')
            target_timestamp = attempt_graph_values[i][0]
            if i == len(attempt_graph_values) - 1:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i-1][0])
            else:
                timestamp_range = abs(target_timestamp - attempt_graph_values[i+1][0])

            valid_camera_indices = []
            valid_camera_timestamps = []
            for j in range(last_index, len(camera_files)):
                camera_file = camera_files[j]
                underscore_index = camera_file.find("_")
                last_period_index = camera_file.rfind(".")
                camera_timestamp = camera_files[j][underscore_index+1:last_period_index]

                if abs(target_timestamp - float(camera_timestamp)) < timestamp_range:
                    valid_camera_indices.append(j)
                    valid_camera_timestamps.append(float(camera_timestamp))
                else:
                    if len(valid_camera_indices) > 0:
                        break

            min_timestamp_diff = 1000000
            min_timestamp_diff_indice = 0
            for j in range(len(valid_camera_timestamps)):
                timestamp_diff = abs(target_timestamp - valid_camera_timestamps[j])
                if timestamp_diff < min_timestamp_diff:
                    min_timestamp_diff = timestamp_diff
                    min_timestamp_diff_indice = valid_camera_indices[j]
       
            curr_frame = np.array(Image.open(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/camera_frames/{camera_files[min_timestamp_diff_indice]}').resize((85, 85)))[:, :,: ]
            frames_list.append(curr_frame)

            last_index = min_timestamp_diff_indice - 1

        camera_frames = np.array(frames_list)

        return camera_frames


    def get_attempt_observations(self, attempt_graph_values, attempt_movement_values, rig_recorder_data_folder, include_camera=True):
        '''
        Retrieves and returns all observations (pressure, resistance, current, volatge, stage/pipette positions, and camera frames) for each timestamp of the current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt
            attempt_movement_values (np.array): array containing the movement value rows associated with the graph value rows
            rig_recorder_data_folder (string): the filename of the rig recorder folder containing the experiment data
            include_camera (boolean): boolean determining whether to include camera frames in the observations

        Returns:
            pressure_values (np.array): numpy array containing pressure values for the current attempt
            resistance_values (np.array): numpy array containing resistance values for the current attempt
            current_values (np.array): numpy array containing current values for the current attempt
            voltage_values (np.array): numpy array containing voltage values for the current attempt
            state_positions (np.array): numpy array containing stage positions for the current attempt
            pipette_positions (np.array): numpy array containing pipette positions for the current attempt
            camera_frames (np.array): numpy array containing camera frames for the current attempt
        '''
        #Pressure
        pressure_values = self.get_attempt_pressure_values(attempt_graph_values)

        #Resistance
        resistance_values = self.get_attempt_resistance_values(attempt_graph_values)

        #Current
        current_values = self.get_attempt_current_values(attempt_graph_values)

        #Voltage
        voltage_values = self.get_attempt_voltage_values(attempt_graph_values)

        #State and Pipette Positions
        stage_positions = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions = self.get_attempt_pipette_positions(attempt_movement_values)

        #Camera Frames
        if include_camera:
            camera_frames = self.get_attempt_camera_frames(rig_recorder_data_folder, attempt_graph_values)
            # cast camera values into rgb if they are grayscale

            return pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames

        return pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions
    

    def get_attempt_next_observations(self, attempt_graph_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames, include_next_obs=False, include_camera=True):
        '''
        Rolls all observations (pressure, resistance, current, volatge, stage/pipette positions, and camera frames) to the next timestamp of the current attempt
        and returns as next observations

        Return nones if not including next observations

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt
            current_values (np.array): numpy array containing current values for the current attempt
            voltage_values (np.array): numpy array containing voltage values for the current attempt
            state_positions (np.array): numpy array containing stage positions for the current attempt
            pipette_positions (np.array): numpy array containing pipette positions for the current attempt
            camera_frames (np.array): numpy array camera frames for the current attempt
            include_next_obs (boolean): boolean determining whether to include next observations or not (return nones if not)
            include_camera (boolean): boolean determining whether to include camera frames in the observations

        Returns:
            next_pressure_values (np.array): numpy array containing the next pressure values for the current attempt
            next_resistance_values (np.array): numpy array containing the next resistance values for the current attempt
            next_current_values (np.array): numpy array containing the next current values for the current attempt
            next_voltage_values (np.array): numpy array containing the next voltage values for the current attempt
            next_state_positions (np.array): numpy array containing the next stage positions for the current attempt
            next_pipette_positions (np.array): numpy array containing the next pipette positions for the current attempt
            next_camera_frames (np.array): numpy array containing the next camera frames for the current attempt
        '''
        #If not including next observations, return nones
        if not include_next_obs:
            return None, None, None, None, None, None, None

        graph_values_rolled = np.roll(attempt_graph_values, -1, axis=0)
        next_pressure_values = graph_values_rolled[:, 1].astype(np.uint8)
        next_resistance_values = graph_values_rolled[:, 2].astype(np.uint8)

        next_current_values = np.roll(current_values, -1, axis=0)
        next_voltage_values = np.roll(voltage_values, -1, axis=0)
        next_stage_positions = np.roll(stage_positions, -1, axis=0)
        next_pipette_positions = np.roll(pipette_positions, -1, axis=0)

        if include_camera:
            next_camera_frames = np.roll(camera_frames, -1, axis=0)

            return next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions, next_camera_frames

        return next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions
    
    def get_attempt_actions(self, attempt_movement_values, attempt_graph_values, log_values, include_high_level_actions=False):
        '''
        Retrieves and returns all movement and high-level actions (if include_high_level_actions is True) for the current attempt
        Computes movement actions as the difference between current stage and pipette positions and previous stage and pipette positions

        Author(s): Kaden Stillwagon

        args:
            attempt_movement_values (np.array): array containing the movement value rows associated with the graph value rows
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt
            log_values (pd.Dataframe): dataframe containing the log messages from the log file that are within the rig_recorder_data_folder's time frame
            include_high_level_actions (boolean): boolean determining whether to include high-level actions in the actions

        Returns:
            actions (np.array): numpy array containing all actions for the current attempt
        '''
        #Low-level Actions
        movement_actions_list = list(np.diff(attempt_movement_values[:, 3:], axis=0)) #Note: for hunt cell only need z stage coordinate (must be [:, 1:] to get all 3 stage coordinates)
        movement_actions_list.insert(0, list(np.zeros(attempt_movement_values.shape[1] - 3))) #should be - 1 if need all 3 stage coordinates
        movement_actions = np.array(movement_actions_list)

        #High-level Actions (Not used for hunt cell datasets)
        if include_high_level_actions:
            log_messages = log_values['Message'].str.contains('Executing command')
            action_log_indices = np.where(log_messages == True)[0]
            action_logs = log_values.iloc[action_log_indices]
            action_logs['Full Time'] = (pd.to_datetime(action_logs['Time(HH:MM:SS)'] + '.' + action_logs['Time(ms)'].astype(str), format='%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)).apply(lambda x: x.timestamp())

            curr_demo_actions_logs = action_logs[action_logs['Full Time'] > attempt_movement_values[0][0]]
            curr_demo_actions_logs = curr_demo_actions_logs[curr_demo_actions_logs['Full Time'] < attempt_movement_values[-1][0]]
            filtered_action_logs = curr_demo_actions_logs.drop_duplicates()
            
            actions_list = []
            for i in range(len(attempt_graph_values)):
                curr_timestamp = attempt_graph_values[i][0]
                if i == len(attempt_graph_values) - 1:
                    timestamp_range = abs(curr_timestamp - attempt_graph_values[i-1][0])
                else:
                    timestamp_range = abs(curr_timestamp - attempt_graph_values[i+1][0])

                high_level_action = hash('None')

                for j in range(filtered_action_logs.shape[0]):
                    if abs(curr_timestamp - filtered_action_logs.iloc[j]['Full Time']) < timestamp_range:
                        high_level_action = hash(filtered_action_logs.iloc[j]['Message'][19:])
                        break

                actions = list(movement_actions[i])
                actions.append(high_level_action)
                actions_list.append(actions)
        else:
            actions_list = list(movement_actions)

        actions = np.array(actions_list)

        return actions
    

    def add_attempt_demo_to_dataset(self, num_samples, actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames, next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions, next_camera_frames, include_next_obs=False, include_camera=True):
        '''
        Adds attempt to the dataset hdf5 file as a demo
        Incremenets number of demos in the dataset

        Return nones if not including next observations

        Author(s): Kaden Stillwagon

        args:
            num_samples (int): integer representing the number of samples (or timestamps) in the attempt demo
            current_values (np.array): numpy array containing current values for the current attempt
            voltage_values (np.array): numpy array containing voltage values for the current attempt
            state_positions (np.array): numpy array containing stage positions for the current attempt
            pipette_positions (np.array): numpy array containing pipette positions for the current attempt
            camera_frames (np.array): numpy array camera frames for the current attempt
            next_pressure_values (np.array): numpy array containing the next pressure values for the current attempt
            next_resistance_values (np.array): numpy array containing the next resistance values for the current attempt
            next_current_values (np.array): numpy array containing the next current values for the current attempt
            next_voltage_values (np.array): numpy array containing the next voltage values for the current attempt
            next_state_positions (np.array): numpy array containing the next stage positions for the current attempt
            next_pipette_positions (np.array): numpy array containing the next pipette positions for the current attempt
            next_camera_frames (np.array): numpy array containing the next camera frames for the current attempt
            include_next_obs (boolean): boolean determining whether to include next observations or not (return nones if not)
            include_camera (boolean): boolean determining whether to include camera frames in the observations
        '''
        with h5py.File(f'experiments/Datasets/{self.dataset_name}', 'a') as hf:
            # Create a demo within the dataset_1
            demo_number = hf['data'].attrs['num_demos']
            demo = hf['data'].create_group(f'demo_{demo_number}')
            demo.attrs['num_samples'] = num_samples

            demo.create_dataset('actions', data=actions)
            demo.create_dataset('dones', data=dones)
            #demo.create_dataset('rewards', data=rewards)
            #demo.create_dataset('states', data=demo_states)

            #Observations Group
            observations = demo.create_group('obs')
            # observations.create_dataset('pressure', data=pressure_values)
            observations.create_dataset('resistance', data=resistance_values)
            # observations.create_dataset('current', data=current_values)
            # observations.create_dataset('voltage', data=voltage_values)
            observations.create_dataset('stage_positions', data=stage_positions)
            observations.create_dataset('pipette_positions', data=pipette_positions)
            if include_camera:
                observations.create_dataset('camera_image', data=camera_frames)

            #Next Observations Group
            if include_next_obs:
                next_observations = demo.create_group('next_obs')
                next_observations.create_dataset('pressure', data=next_pressure_values)
                next_observations.create_dataset('resistance', data=next_resistance_values)
                next_observations.create_dataset('current', data=next_current_values)
                # next_observations.create_dataset('voltage', data=next_voltage_values)
                if include_camera:
                    next_observations.create_dataset('camera_image', data=next_camera_frames)

            hf['data'].attrs['num_demos'] = hf['data'].attrs['num_demos'] + 1
            print(f"Added demo_{demo_number} to dataset '{self.dataset_name}' with {num_samples} samples.")
            

        
    def add_demo(self, rig_recorder_data_folder, record_to_file=False):
        '''
        Creates a demo entry into the dataset for each successful attempt in each recording within the rig_recorder_data_folder

        Author(s): Kaden Stillwagon

        args:
            rig_recorder_data_folder (string): the filename of the rig recorder folder containing the experiment data
            record_to_file (boolean): boolean determining whether to record the attempts into the dataset or not
        '''
        print(f"Adding demos from rig_recorder_data_folder: {rig_recorder_data_folder}")
        #Parameters
        include_next_obs = False
        include_camera = True
        include_high_level_actions = False

        #Load experiment data from rig recorder and log files
        graph_values, movement_values, log_values = self.load_experiment_data(rig_recorder_data_folder=rig_recorder_data_folder)

        #Get first and last timestamp within rig recorder folder
        experiment_first_timestamp = graph_values[0][0] - 1
        experiment_last_timestamp = graph_values[-1][0] + 1
        # print(experiment_first_timestamp)
        # print(experiment_last_timestamp)

        #Get starting and ending timestamps of each recording in the rig recorder folder
        experiment_recordings_timestamp_ranges = self.get_timestamps_for_all_experiment_recordings(log_values, experiment_first_timestamp, experiment_last_timestamp)

        #Get starting and ending timestamps of each successful hunt cell attempt in each recording
        experiment_successful_hunt_cell_timestamp_ranges = self.get_timestamps_for_all_successful_hunt_cell_attempts(log_values, experiment_recordings_timestamp_ranges)
        #print(experiment_successful_hunt_cell_timestamp_ranges)
        
        #Create dataset entry for each successful hunt cell attempt
        for hunt_cell_timestamps in experiment_successful_hunt_cell_timestamp_ranges:
            attempt_first_timestamp = hunt_cell_timestamps[0]
            attempt_last_timestamp = hunt_cell_timestamps[1]

            #Get the graph values within the current attempt
            attempt_graph_values = self.truncate_graph_values(graph_values, attempt_first_timestamp, attempt_last_timestamp)

            #Get movement values with current attempt and associate to graph value timestamps
            attempt_movement_values = self.associate_attempt_movement_and_graph_values(attempt_graph_values, movement_values)


            #~~~~Collect Attempt Data~~~~~:

            #DONES
            dones = self.get_attempt_dones(attempt_graph_values)

            #REWARDS -Not Using
            #rewards = np.copy(done_values) * 100

            #OBSERVATIONS
            pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames = self.get_attempt_observations(attempt_graph_values, attempt_movement_values, rig_recorder_data_folder, include_camera=include_camera)
            
            #NEXT OBSERVATIONS - Not Using (returns Nones now)
            next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions, next_camera_frames = self.get_attempt_next_observations(attempt_graph_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames, include_next_obs=include_next_obs, include_camera=include_camera)
            
            #ACTIONS
            actions = self.get_attempt_actions(attempt_movement_values, attempt_graph_values, log_values, include_high_level_actions=include_high_level_actions)
            
            #STATES - Not Using (prev states of pipette and stage position now observations)



            #~~~~Add Attempt to Dataset~~~~
            if record_to_file:
                self.add_attempt_demo_to_dataset(
                    num_samples=attempt_graph_values.shape[0], 
                    actions=actions, 
                    dones=dones, 
                    pressure_values=pressure_values, 
                    resistance_values=resistance_values, 
                    current_values=current_values, 
                    voltage_values=voltage_values, 
                    stage_positions=stage_positions, 
                    pipette_positions=pipette_positions, 
                    camera_frames=camera_frames,
                    next_pressure_values=next_pressure_values, 
                    next_resistance_values=next_resistance_values, 
                    next_current_values=next_current_values, 
                    next_voltage_values=next_voltage_values, 
                    next_stage_positions=next_stage_positions, 
                    next_pipette_positions=next_pipette_positions, 
                    next_camera_frames=next_camera_frames,
                    include_next_obs=include_next_obs,
                    include_camera=include_camera
                    )


if __name__ == '__main__':
    # dataset_name = '2025_03_20-15_19_dataset.hdf5'
    dataset_name = 'HEK_dataset_fullStage.hdf5'  # For initial training dataset, uncomment this line to overwrite the existing dataset

    # rig_recorder_data_folder_set =  [
    #     "2025_03_11-16_01",
    #     "2025_03_11-16_32",
    #     "2025_03_11-16_49"
    # ]

    rig_recorder_data_folder_set =  [
        "2025_03_11-16_01",
        "2025_03_11-16_32",
        "2025_03_11-16_49",
        "2025_03_11-17_50",
        "2025_03_11-18_00",
        "2025_03_11-18_09",
        "2025_03_11-18_15",
        "2025_03_11-18_25",
        "2025_03_17-16_27",
        "2025_03_17-16_57",
        "2025_03_17-17_21",
        "2025_03_17-17_26",
        "2025_03_17-18_10",
        "2025_03_20-13_52",
        "2025_03_20-14_01",
        "2025_03_20-14_42",
        "2025_03_20-14_53",
        "2025_03_20-15_04",
        "2025_03_20-15_19",
        "2025_03_20-15_34",
        "2025_03_20-15_45",
        "2025_03_20-16_07",
        "2025_03_20-16_15",
        "2025_03_20-16_35",
        "2025_03_20-16_59",
        "2025_03_20-17_23",
        "2025_03_20-17_53",
        "2025_03_20-18_01",
        "2025_03_20-18_13",
        "2025_03_20-18_25",
        "2025_03_20-18_49",
        "2025_03_20-19_03",
        "2025_03_20-19_13",
        "2025_03_25-16_42",
        "2025_03_25-16_12",
        "2025_03_25-15_34",
        "2025_03_25-14_48",
        "2025_03_25-14_32",
        "2025_03_25-14_17",
        "2025_04_07-14_32", 
        "2025_04_07-14_50", 
        "2025_04_07-15_50", 
        "2025_04_07-18_04"
        ] # this is most recent HEK DATA with NO overlays. some manual some automatic. (3/11/2025 - 4/7/2025)

    for folder in rig_recorder_data_folder_set:
        print(f"Processing folder: {folder}")
        rig_recorder_data_folder = folder
        record_to_file = True

        datasetBuilder = DatasetBuilder(dataset_name=dataset_name)
        datasetBuilder.add_demo(rig_recorder_data_folder=rig_recorder_data_folder, record_to_file=record_to_file)


#Cases to Consider:
#Each graph_recording/movement folder made with time when program started
    #Can have multiple recordings in one of these folders (look for "Recording Started") in the log file
#Can also have multiple hunt cell attempts within a single recording
    #Can have multiple success and some failures in

#NOTE
    #Neuron hunt demo should start when "starting descent....."

