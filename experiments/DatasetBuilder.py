import os
import h5py
import pandas as pd
import numpy as np
import datetime
import json
from PIL import Image


ATL_TO_UTC_TIME_DELTA = 4 #March 9 - Nov 1: 4 hours, otherwise 5 hours
class DatasetBuilder():
    def __init__(self,
                 dataset_name,
                 val_ratio: float = 1 / 6,
                 omit_stage_movement: bool = True,
                 random_seed: int = 0,
                 rotate_valid: bool = True,
                 *,                              # ── NEW: keyword-only below
                 stage_y_axis_flip: bool = True, # flip the stage Y axis
                 pipette_rotation_deg: float = -60.75 # θ ─ rotate pipette XY
                 ):
        """
        Parameters
        ----------
        val_ratio : float
            Fraction of demos reserved for validation.  If set to 0 there will
            be *no* validation split and no “mask” group will be written.
        """
        self.dataset_name = dataset_name
        self.zero_values = False
        self.center_crop = True
        self.rotate = True                 # train-time augmentation
        self.rotate_valid = rotate_valid
        self.inaction = 3
        self.val_ratio = val_ratio
        self.omit_stage_movement = omit_stage_movement
        self.rng = np.random.default_rng(random_seed)
        self.stage_y_axis_flip = stage_y_axis_flip
        self.pipette_rotation_deg = pipette_rotation_deg

        # Only keep bookkeeping for the splits that will exist
        if self.val_ratio == 0:
            self._split_keys = {"train": []}
        else:
            self._split_keys = {"train": [], "valid": []}

        # create dataset file stub if needed
        if dataset_name not in os.listdir('experiments/Datasets'):
            with h5py.File(f'experiments/Datasets/{dataset_name}', 'w') as hf:
                group = hf.create_group('data')
                group.attrs['num_demos'] = 0

        # ──────────────────────────────────────────────────────────────────────
    def _transform_pipette_positions(self,
                                     stage_positions: np.ndarray,
                                     pipette_positions: np.ndarray
                                     ) -> np.ndarray:
        """
        Express the pipette manipulator coordinates in the **stage frame**.

        Steps  
        1.  Optionally flip the stage Y axis.  
        2.  Rotate raw pipette XY by `self.pipette_rotation_deg`.  
        3.  Add the transformed stage coordinates to the rotated pipette
            coordinates.

        Returns
        -------
        np.ndarray      shape (N, 3)
        """
        stage_adj = stage_positions.copy()
        if self.stage_y_axis_flip:
            stage_adj[:, 1] = -stage_adj[:, 1]

        pip_rot = self._rotate_positions(pipette_positions,
                                         self.pipette_rotation_deg)

        # combine -- only XY are coupled to the stage; Z stays independent
        pip_rot[:, :2] += stage_adj[:, :2]
        return pip_rot           # shape (N, 3)



    def _write_split_masks(self):
        """
        Create / overwrite the ‘mask’ group that stores UTF-8 demo keys for each
        split.  When `val_ratio` is zero the group is omitted entirely.
        """
        # Skip mask creation if no validation split was requested
        if self.val_ratio == 0:
            print("✨  val_ratio is 0 — dataset contains only training demos; "
                  "skipping mask creation.")
            return

        with h5py.File(f'experiments/Datasets/{self.dataset_name}', 'a') as hf:
            if "mask" in hf:
                del hf["mask"]                    # wipe old masks if rerun
            mask_grp = hf.create_group("mask")
            for name in ("train", "valid"):
                keys = np.asarray(self._split_keys[name], dtype='S')
                mask_grp.create_dataset(name, data=keys)

        print(f"✨  wrote split masks: {len(self._split_keys['train'])} train | "
              f"{len(self._split_keys['valid'])} valid")


    def _rotate_positions(self, positions, angle_degrees):
        """
        Rotate x and y coordinates in a positions array by angle_degrees.
        Assumes positions is an array of shape (n, 3) where column 0 is x and column 1 is y.
        """
        rad = np.deg2rad(angle_degrees)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        rotated = positions.copy()
        rotated[:, 0] = cos_val * positions[:, 0] - sin_val * positions[:, 1]
        rotated[:, 1] = sin_val * positions[:, 0] + cos_val * positions[:, 1]
        return rotated

    def _rotate_actions(self, actions, angle_degrees):
        """
        Rotates the (x, y) components of each action vector by the specified angle.
        Assumes actions is an (N, D) array where columns 0 is x, 1 is y, 2 is z (z unchanged).
        Any extra dimensions are passed through unchanged.
        """
        rad = np.deg2rad(angle_degrees)
        cos_val = np.cos(rad)
        sin_val = np.sin(rad)
        actions_rot = actions.copy()
        x = actions[:, 0]
        y = actions[:, 1]
        actions_rot[:, 0] = cos_val * x - sin_val * y
        actions_rot[:, 1] = sin_val * x + cos_val * y
    # -------------------  ADD THIS BLOCK  --------------------
        # rotate PIPETTE x / y  (cols 3, 4)  — only if they exist
        if actions.shape[1] >= 5:          # ensure the columns are present
            x_pip = actions[:, 3]
            y_pip = actions[:, 4]
            actions_rot[:, 3] = cos_val * x_pip - sin_val * y_pip
            actions_rot[:, 4] = sin_val * x_pip + cos_val * y_pip
    # ---------------------------------------------------------
        return actions_rot

    def filter_inactive_actions(self, actions, *arrays):
        """
        Vectorised run-length detection of ≥ self.inaction consecutive zero-rows.
        Much faster than the original while-loop; behaviour is unchanged.
        """
        if self.inaction == 0:
            return (actions,) + tuple(arrays)

        inactive = (actions.sum(axis=1) == 0).astype(np.int8)
        # run-length encode: diff==1 -> start, diff==-1 -> end
        diff = np.diff(np.concatenate(([0], inactive, [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        keep = np.ones_like(inactive, dtype=bool)
        for s, e in zip(starts, ends):
            if e - s >= self.inaction:
                keep[s:e] = False

        filtered = (actions[keep],) + tuple(arr[keep] for arr in arrays)
        return filtered

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
        """
        Optimised by replacing the row-by-row scan with two np.searchsorted calls
        (binary search) on the monotonic timestamp column.  Output is identical.

        Alternative considered: np.argmin(abs(timestamps-ts)) for each boundary
        (still O(N)); rejected in favour of searchsorted.
        """
        ts = graph_values[:, 0]

        # index of element ≥ first_timestamp  (left boundary)
        i0 = np.searchsorted(ts, first_timestamp, side="left")
        if i0 == len(ts):                       # beyond the array → clamp
            i0 = len(ts) - 1
        elif i0 and abs(ts[i0-1] - first_timestamp) < abs(ts[i0] - first_timestamp):
            i0 -= 1                             # neighbour on the left is nearer

        # index of last element ≤ last_timestamp (right boundary, inclusive)
        i1 = np.searchsorted(ts, last_timestamp, side="right") - 1
        if i1 < 0:
            i1 = 0
        elif i1 + 1 < len(ts) and abs(ts[i1+1] - last_timestamp) < abs(ts[i1] - last_timestamp):
            i1 += 1                             # neighbour on the right is nearer

        return graph_values[i0:i1 + 1]

    def associate_attempt_movement_and_graph_values(self, attempt_graph_values, movement_values):
        """
        Uses np.searchsorted to find, for each graph timestamp, the closest movement
        timestamp in O(T log M) total.  Eliminates the nested Python loops while
        returning identical alignments.
        """
        g_ts = attempt_graph_values[:, 0]
        m_ts = movement_values[:, 0]

        right = np.searchsorted(m_ts, g_ts)                      # index of first ≥
        left  = np.clip(right - 1, 0, len(m_ts) - 1)            # previous element
        right = np.clip(right,      0, len(m_ts) - 1)

        choose_right = np.abs(m_ts[right] - g_ts) < np.abs(m_ts[left] - g_ts)
        indices = np.where(choose_right, right, left)

        return movement_values[indices]

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
        """
        Parses JSON once per row via list-comprehension, pads in NumPy
        (no per-element .append loops).  Output exactly matches the original.
        """
        lists = [json.loads(s) for s in attempt_graph_values[:, 3]]
        max_len = max(len(lst) for lst in lists)
        return np.asarray([lst + [lst[-1]] * (max_len - len(lst)) for lst in lists],
                        dtype=np.float64)

    def get_attempt_voltage_values(self, attempt_graph_values):
        """
        Same optimisation as for current values: fast JSON parse & vectorised pad.
        """
        lists = [json.loads(s) for s in attempt_graph_values[:, 4]]
        max_len = max(len(lst) for lst in lists)
        return np.asarray([lst + [lst[-1]] * (max_len - len(lst)) for lst in lists],
                        dtype=np.float64)
   
    def get_attempt_stage_positions(self, attempt_movement_values):
        '''
        Returns stage positions for current attempt

        Author(s): Kaden Stillwagon

        args:
            attempt_graph_values (np.array): array containing the graph values, truncated to only values within the current attempt

        Returns:
            state_positions (np.array): numpy array containing stage positions for the current attempt

        '''
        stage_positions = attempt_movement_values[:, 1:4].astype(np.float64) #Note: for hunt cell, only need z-coordinate for stage positions (must use [:, 1:] to get all 3 stage coordinates)
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
    
    def get_attempt_camera_frames(self, rig_recorder_data_folder, attempt_graph_values, rotation_angle=None):
        camera_files = os.listdir(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/camera_frames')
        camera_files.sort()
        frames_list = []
        last_index = 0
        for i in range(len(attempt_graph_values)):
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
                camera_timestamp = camera_file[underscore_index+1:last_period_index]
                if abs(target_timestamp - float(camera_timestamp)) < timestamp_range:
                    valid_camera_indices.append(j)
                    valid_camera_timestamps.append(float(camera_timestamp))
                else:
                    if len(valid_camera_indices) > 0:
                        break
            min_timestamp_diff = 1e6
            min_timestamp_diff_indice = 0
            for j in range(len(valid_camera_timestamps)):
                timestamp_diff = abs(target_timestamp - valid_camera_timestamps[j])
                if timestamp_diff < min_timestamp_diff:
                    min_timestamp_diff = timestamp_diff
                    min_timestamp_diff_indice = valid_camera_indices[j]
            # Open the image from the selected camera frame.
            pil_image = Image.open(f'experiments/Data/rig_recorder_data/{rig_recorder_data_folder}/camera_frames/{camera_files[min_timestamp_diff_indice]}')
            # Apply rotation augmentation before cropping and resizing if requested.
            if rotation_angle is not None:
                pil_image = pil_image.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
            if self.center_crop:
                pil_image = self.crop_image_center(pil_image)
            curr_frame = np.array(pil_image.resize((85, 85)))
            frames_list.append(curr_frame)
            last_index = min_timestamp_diff_indice - 1
        camera_frames = np.array(frames_list)
        return camera_frames

    def crop_image_center(self,pil_image):
        """
        Center crops the input PIL image by one half its original dimensions.
        
        For example, if the image is 400x400, the resulting crop will be 100x100 
        from the center of the image.
        
        Args:
            pil_image (PIL.Image.Image): The input image.
            
        Returns:
            PIL.Image.Image: The center-cropped image.
        """
        width, height = pil_image.size
        new_width = width // 2
        new_height = height // 2
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return pil_image.crop((left, top, right, bottom))

    def get_attempt_observations(self, attempt_graph_values, attempt_movement_values, rig_recorder_data_folder, include_camera=True, rotation_angle=None):
        # Pressure
        pressure_values = self.get_attempt_pressure_values(attempt_graph_values)
        # Resistance
        resistance_values = self.get_attempt_resistance_values(attempt_graph_values)
        # Current
        current_values = self.get_attempt_current_values(attempt_graph_values)
        # Voltage
        voltage_values = self.get_attempt_voltage_values(attempt_graph_values)
        # State and Pipette Positions
        stage_positions = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions = self.get_attempt_pipette_positions(attempt_movement_values)


        # ‼️ NEW: express pipette coords in the stage frame
        pipette_positions = self._transform_pipette_positions(
            stage_positions, pipette_positions
        )

        # If a rotation angle is provided, rotate the position data (only x and y).
        if rotation_angle is not None:
            stage_positions = self._rotate_positions(stage_positions, rotation_angle)
            pipette_positions = self._rotate_positions(pipette_positions, rotation_angle)
        # Camera Frames
        if include_camera:
            camera_frames = self.get_attempt_camera_frames(rig_recorder_data_folder, attempt_graph_values, rotation_angle=rotation_angle)
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
    
    def get_attempt_actions(self,
                            attempt_movement_values,
                            attempt_graph_values,
                            log_values,
                            include_high_level_actions: bool = False):
        """
        Low-level actions  (Δstage + Δpipette)  
        • Stage-Y may be flipped.  
        • Pipette-XY is first rotated by `self.pipette_rotation_deg`
          *then* added to the (optionally flipped) stage-XY delta.  
        • Pipette-Z **never** receives any stage-Z contribution.

        Optionally appends one extra column with a hashed high-level command.
        """
        # ───────────────────── raw frame deltas ────────────────────────
        movement_actions_list = list(
            np.diff(attempt_movement_values[:, 1:], axis=0)
        )
        movement_actions_list.insert(
            0, list(np.zeros(attempt_movement_values.shape[1] - 1))
        )
        movement_actions = np.array(movement_actions_list)          # (N, 6)

        # ─────────── transform pipette part into stage frame ───────────
        stage_delta        = movement_actions[:, :3].copy()
        pip_raw_delta      = movement_actions[:, 3:6].copy()

        # 1 ─ flip stage-Y if requested
        if self.stage_y_axis_flip:
            stage_delta[:, 1] = -stage_delta[:, 1]

        # 2 ─ rotate pipette XY
        pip_rot_delta = self._rotate_positions(
            pip_raw_delta, self.pipette_rotation_deg
        )

        # 3 ─ write back (XY coupled, Z independent)
        movement_actions[:, :3]  = stage_delta                      # updated stage
        movement_actions[:, 3:5] = stage_delta[:, :2] + pip_rot_delta[:, :2]
        movement_actions[:, 5]   = pip_rot_delta[:, 2]              # pipette Z only

        # ─────────────── optional high-level actions ──────────────────
        if include_high_level_actions:
            log_messages = log_values['Message'].str.contains('Executing command')
            action_logs  = log_values.loc[log_messages].copy()
            action_logs['Full Time'] = (
                pd.to_datetime(
                    action_logs['Time(HH:MM:SS)'] + '.' +
                    action_logs['Time(ms)'].astype(str),
                    format='%Y-%m-%d %H:%M:%S.%f'
                ) + datetime.timedelta(hours=ATL_TO_UTC_TIME_DELTA)
            ).apply(lambda x: x.timestamp())

            # keep only those inside this attempt
            mask = (action_logs['Full Time'] > attempt_movement_values[0, 0]) & \
                   (action_logs['Full Time'] < attempt_movement_values[-1, 0])
            action_logs = action_logs.loc[mask].drop_duplicates()

            hi_lvl = np.full((movement_actions.shape[0], 1),
                             hash('None'), dtype=np.int64)
            for ts, msg in zip(action_logs['Full Time'], action_logs['Message']):
                idx = np.argmin(np.abs(attempt_graph_values[:, 0] - ts))
                hi_lvl[idx, 0] = hash(msg[19:])          # strip prefix

            actions = np.hstack([movement_actions, hi_lvl])
        else:
            actions = movement_actions

        return actions

    def add_attempt_demo_to_dataset(self, num_samples, actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames, next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions, next_camera_frames, include_next_obs=False, include_camera=True, split_label="train"):
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
        # --- NEW CODE: call the filtering method before writing to file ---
        # We base the filtering on the actions array and apply the same removal to all other arrays.
        if include_next_obs:
            if include_camera:
                (actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames,
                 next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions, next_camera_frames) = \
                    self.filter_inactive_actions(actions, dones, pressure_values, resistance_values, current_values, voltage_values,
                                                 stage_positions, pipette_positions, camera_frames,
                                                 next_pressure_values, next_resistance_values, next_current_values, next_voltage_values,
                                                 next_stage_positions, next_pipette_positions, next_camera_frames)
            else:
                (actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions,
                 next_pressure_values, next_resistance_values, next_current_values, next_voltage_values, next_stage_positions, next_pipette_positions) = \
                    self.filter_inactive_actions(actions, dones, pressure_values, resistance_values, current_values, voltage_values,
                                                 stage_positions, pipette_positions,
                                                 next_pressure_values, next_resistance_values, next_current_values, next_voltage_values,
                                                 next_stage_positions, next_pipette_positions)
        else:
            if include_camera:
                (actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions, camera_frames) = \
                    self.filter_inactive_actions(actions, dones, pressure_values, resistance_values, current_values, voltage_values,
                                                 stage_positions, pipette_positions, camera_frames)
            else:
                (actions, dones, pressure_values, resistance_values, current_values, voltage_values, stage_positions, pipette_positions) = \
                    self.filter_inactive_actions(actions, dones, pressure_values, resistance_values, current_values, voltage_values,
                                                 stage_positions, pipette_positions)

        # Update the num_samples based on the filtered actions array:
        num_samples = actions.shape[0]
        # --- END NEW CODE ---
        
        with h5py.File(f'experiments/Datasets/{self.dataset_name}', 'a') as hf:
            # Create a demo within the dataset_1
            demo_number = hf['data'].attrs['num_demos']
            demo = hf['data'].create_group(f'demo_{demo_number}')
            demo.attrs['num_samples'] = num_samples
            demo.attrs['split'] = split_label

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
            print(f"Added {split_label} demo_{demo_number} to dataset '{self.dataset_name}' with {num_samples} samples.")
            return f"demo_{demo_number}"  

    def add_demo(self, rig_recorder_data_folder, record_to_file=False):
        """
        Parse one rig-recorder folder, extract each successful hunt-cell attempt,
        and store them as demos with a train / valid split.  Rotation augmentation
        is optional for validation demos, controlled by `self.rotate_valid`.
        """
        print(f"Adding demos from rig_recorder_data_folder: {rig_recorder_data_folder}")

        include_next_obs = False
        include_camera = True
        include_high_level_actions = False

        # ─── load raw CSV data ────────────────────────────────────────────────────
        graph_values, movement_values, log_values = self.load_experiment_data(
            rig_recorder_data_folder=rig_recorder_data_folder
        )

        experiment_first_timestamp = graph_values[0][0] - 1
        experiment_last_timestamp  = graph_values[-1][0] + 1

        rec_ranges = self.get_timestamps_for_all_experiment_recordings(
            log_values, experiment_first_timestamp, experiment_last_timestamp
        )
        hunt_ranges = self.get_timestamps_for_all_successful_hunt_cell_attempts(
            log_values, rec_ranges
        )

        # ─── iterate through each successful hunt-cell attempt ────────────────────
        for attempt_first_timestamp, attempt_last_timestamp in hunt_ranges:

            attempt_graph_values = self.truncate_graph_values(
                graph_values, attempt_first_timestamp, attempt_last_timestamp
            )
            attempt_movement_values = self.associate_attempt_movement_and_graph_values(
                attempt_graph_values, movement_values
            )

            # ─── build observations & actions ─────────────────────────────────────
            dones = self.get_attempt_dones(attempt_graph_values)

            (pressure_values, resistance_values, current_values, voltage_values,
            stage_positions, pipette_positions, camera_frames) = \
                self.get_attempt_observations(
                    attempt_graph_values, attempt_movement_values,
                    rig_recorder_data_folder,
                    include_camera=include_camera, rotation_angle=None
                )

            next_obs = self.get_attempt_next_observations(
                attempt_graph_values, current_values, voltage_values,
                stage_positions, pipette_positions, camera_frames,
                include_next_obs=include_next_obs, include_camera=include_camera
            )

            actions = self.get_attempt_actions(
                attempt_movement_values, attempt_graph_values, log_values,
                include_high_level_actions=include_high_level_actions
            )

            # ─── optional: skip demos with stage XYZ motion ───────────────────────
            if self.omit_stage_movement and np.any(actions[:, :3]):
                print("  skipped – demo contains stage movement")
                continue

            # ─── decide train / valid split BEFORE augmentation ───────────────────
            split_lbl = "valid" if self.rng.random() < self.val_ratio else "train"

            # ─── write ORIGINAL demo ──────────────────────────────────────────────
            if record_to_file:
                demo_key = self.add_attempt_demo_to_dataset(
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
                    next_pressure_values=next_obs[0] if next_obs else None,
                    next_resistance_values=next_obs[1] if next_obs else None,
                    next_current_values=next_obs[2] if next_obs else None,
                    next_voltage_values=next_obs[3] if next_obs else None,
                    next_stage_positions=next_obs[4] if next_obs else None,
                    next_pipette_positions=next_obs[5] if next_obs else None,
                    next_camera_frames=next_obs[6] if next_obs and include_camera else None,
                    include_next_obs=include_next_obs,
                    include_camera=include_camera,
                    split_label=split_lbl
                )
                self._split_keys[split_lbl].append(demo_key)
                print(f"  added original {split_lbl} demo")

            # ─── rotation augmentation (train always; valid if rotate_valid) ─────
            if self.rotate and record_to_file and (split_lbl == "train" or
                                                (split_lbl == "valid" and self.rotate_valid)):
                angles = np.linspace(0, 360, num=10, endpoint=False)[1:]   # 10-way, skip 0 °
                for angle in angles:

                    (aug_pressure, aug_resistance, aug_current, aug_voltage,
                    aug_stage_pos, aug_pipette_pos, aug_cam) = \
                        self.get_attempt_observations(
                            attempt_graph_values, attempt_movement_values,
                            rig_recorder_data_folder,
                            include_camera=include_camera, rotation_angle=angle
                        )

                    aug_actions = self._rotate_actions(actions, angle)

                    aug_next_obs = self.get_attempt_next_observations(
                        attempt_graph_values, current_values, voltage_values,
                        aug_stage_pos, aug_pipette_pos, aug_cam,
                        include_next_obs=include_next_obs, include_camera=include_camera
                    ) if include_next_obs else (None,) * 7

                    aug_key = self.add_attempt_demo_to_dataset(
                        num_samples=attempt_graph_values.shape[0],
                        actions=aug_actions,
                        dones=dones,
                        pressure_values=pressure_values,
                        resistance_values=resistance_values,
                        current_values=current_values,
                        voltage_values=voltage_values,
                        stage_positions=aug_stage_pos,
                        pipette_positions=aug_pipette_pos,
                        camera_frames=aug_cam,
                        next_pressure_values=aug_next_obs[0],
                        next_resistance_values=aug_next_obs[1],
                        next_current_values=aug_next_obs[2],
                        next_voltage_values=aug_next_obs[3],
                        next_stage_positions=aug_next_obs[4],
                        next_pipette_positions=aug_next_obs[5],
                        next_camera_frames=aug_next_obs[6],
                        include_next_obs=include_next_obs,
                        include_camera=include_camera,
                        split_label=split_lbl            # ← train *or* valid
                    )
                    self._split_keys[split_lbl].append(aug_key)
                    print(f"  added augmented {split_lbl} demo (angle {angle}°)")

if __name__ == '__main__':
    # dataset_name = '2025_03_20-15_19_dataset.hdf5'
    dataset_name = 'HEK_dataset_v0_024.df5'  # For initial training dataset, uncomment this line to overwrite the existing dataset


    # rig_recorder_data_folder_set =  [
    #     "2025_03_11-16_01",
    #     "2025_03_11-16_32",
    #     "2025_03_11-16_49"
    # ]

    # rig_recorder_data_folder_set =  [
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    #     "2025_04_10-12_21",
    #     "2025_04_10-12_30",
    #     "2025_04_10-15_01",
    #     "2025_04_10-17_31",
    #     "2025_04_07-14_32", 
    #     "2025_04_07-14_50", 
    #     "2025_04_07-15_50", 
    #     "2025_04_07-18_04"
    #  ] # completely manual HEK data with no overlays. (4/10/2025)

    # rig_recorder_data_folder_set =  ["2025_03_11-16_32"] # inference test data (3/11/2025)
        
    rig_recorder_data_folder_set =  [
        "2025_05_20-15_50",
        "2025_05_20-15_16",
        "2025_05_20-14_05",
        "2025_04_10-11_57",
        "2025_04_10-12_16",
        "2025_04_10-12_21",
        "2025_04_10-12_30",
        "2025_04_10-15_01",
        "2025_04_10-17_31",
        "2025_04_07-14_32", 
        "2025_04_07-14_50", 
        "2025_04_07-15_50", 
        "2025_04_07-18_04"
     ] # completely manual HEK data with no overlays. (5/20/2025) v16, including more random start positions

    datasetBuilder = DatasetBuilder(
        dataset_name=dataset_name,
        val_ratio=0.1,              # 1-in-6 validation demos
        omit_stage_movement=True,   # skip demos with stage XYZ motion
        random_seed=0               # change to alter the split
    )


    for folder in rig_recorder_data_folder_set:
        print(f"Processing folder: {folder}")
        datasetBuilder.add_demo(
            rig_recorder_data_folder=folder,
            record_to_file=True
        )


    datasetBuilder._write_split_masks()