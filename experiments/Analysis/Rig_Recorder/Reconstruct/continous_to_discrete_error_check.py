import pandas as pd
import numpy as np
import re

def parse_file_to_dataframe(file_path):
    # Regex pattern to match the format of each line
    pattern = re.compile(r'timestamp:(?P<timestamp>[\d.]+)\s+st_x:(?P<st_x>[-\d.]+)\s+st_y:(?P<st_y>[-\d.]+)\s+st_z:(?P<st_z>[-\d.]+)\s+pi_x:(?P<pi_x>[-\d.]+)\s+pi_y:(?P<pi_y>[-\d.]+)\s+pi_z:(?P<pi_z>[-\d.]+)')

    data = {
        'timestamp': [],
        'st_x': [],
        'st_y': [],
        'st_z': [],
        'pi_x': [],
        'pi_y': [],
        'pi_z': []
    }

    # Read file line by line and apply regex
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                data['timestamp'].append(float(match.group('timestamp')))
                data['st_x'].append(float(match.group('st_x')))
                data['st_y'].append(float(match.group('st_y')))
                data['st_z'].append(float(match.group('st_z')))
                data['pi_x'].append(float(match.group('pi_x')))
                data['pi_y'].append(float(match.group('pi_y')))
                data['pi_z'].append(float(match.group('pi_z')))
    
    # Create and return a DataFrame
    return pd.DataFrame(data)

def align_based_on_coordinates(original_path, replay_path, output_path):
    # Parse both files to create DataFrames
    original_df = parse_file_to_dataframe(original_path)
    replay_df = parse_file_to_dataframe(replay_path)

    # Extract the initial coordinates from the original data
    initial_coords = original_df.iloc[0][['st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']].values

    # Find the index in replay_df where the coordinates match the initial coordinates from original_df
    tolerance = 0.1
    matched_index = None

    for i in range(len(replay_df)):
        replay_coords = replay_df.iloc[i][['st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']].values
        if np.allclose(replay_coords, initial_coords, atol=tolerance):
            matched_index = i
            break

    if matched_index is None:
        raise ValueError("No matching coordinates found in the replay data.")

    # Align replay data from the matching index onward
    aligned_replay_df = replay_df.iloc[matched_index:].reset_index(drop=True)

    # Ensure the replay data is of the same length as the original for comparison
    min_length = min(len(original_df), len(aligned_replay_df))
    original_aligned_df = original_df.iloc[:min_length]
    replay_aligned_df = aligned_replay_df.iloc[:min_length]

    # Calculate errors for each positional value
    error_data = {
        'timestamp': original_aligned_df['timestamp']
    }

    for axis in ['st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']:
        error_data[f'{axis}_error'] = original_aligned_df[axis] - replay_aligned_df[axis]

    # Create a DataFrame from the error data
    error_df = pd.DataFrame(error_data)

    # Save the error DataFrame to a CSV file
    error_df.to_csv(output_path, index=False)
    print(f"Error data has been successfully saved to {output_path}")


# Example usage
original_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_11-16_52\movement_recording.csv"  # Replace with the original file path
replay_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_11-16_54\movement_recording.csv" # Replace with the replay file path
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder\Reconstruct\ctodecanalysis\output.csv"

align_based_on_coordinates(original_path, replay_path, output_path)
