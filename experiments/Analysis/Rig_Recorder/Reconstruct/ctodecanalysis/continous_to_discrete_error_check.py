import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load both CSV files into separate DataFrames
file_path_1 = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2024\ForestLab\ML\Rig_Replay\error_comparison\movement_recording.csv"
file_path_2 = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2024\ForestLab\ML\Rig_Replay\error_comparison\movement_recording_truncated.csv"

# Read data from both files
data1 = pd.read_csv(file_path_1, sep='\s+', header=None)
data2 = pd.read_csv(file_path_2, sep='\s+', header=None)

# Rename columns based on expected key:value format
columns = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']
data1.columns = columns
data2.columns = columns

# Extract numeric values from each column
for col in columns:
    data1[col] = data1[col].str.split(':').str[1].astype(float)
    data2[col] = data2[col].str.split(':').str[1].astype(float)

# zero the time for both datasets
initial_timestamp_1 = data1['timestamp'].iloc[0]
data1['timestamp'] -= initial_timestamp_1
initial_timestamp_2 = data2['timestamp'].iloc[0]
data2['timestamp'] -= initial_timestamp_2

# Zero pipette movement to initial position for both datasets
# Dataset 1
initial_coords_1 = data1.iloc[0][['pi_x', 'pi_y', 'pi_z']].values
data1['pi_x'] -= initial_coords_1[0]
data1['pi_y'] -= initial_coords_1[1]
data1['pi_z'] -= initial_coords_1[2]
# Calculate resultant displacement for dataset 1
data1['pipette_resultant'] = np.sqrt(data1['pi_x']**2 + data1['pi_y']**2 + data1['pi_z']**2)

# Dataset 2
initial_coords_2 = data2.iloc[0][['pi_x', 'pi_y', 'pi_z']].values
data2['pi_x'] -= initial_coords_2[0]
data2['pi_y'] -= initial_coords_2[1]
data2['pi_z'] -= initial_coords_2[2]
# Calculate resultant displacement for dataset 2
data2['pipette_resultant'] = np.sqrt(data2['pi_x']**2 + data2['pi_y']**2 + data2['pi_z']**2)

# Truncate the longer dataset to match the length of the shorter one
min_length = min(len(data1), len(data2))
data1_truncated = data1.iloc[:min_length]
data2_truncated = data2.iloc[:min_length]

# Plot both resultant displacement curves
plt.figure(figsize=(12, 6))
plt.plot(data1_truncated['timestamp'], data1_truncated['pipette_resultant'], label='Dataset 1 - Pipette Resultant')
plt.plot(data2_truncated['timestamp'], data2_truncated['pipette_resultant'], label='Dataset 2 - Pipette Resultant')
plt.xlabel('Time (s)')
plt.ylabel('Pipette Resultant Displacement')
plt.title('Pipette Resultant Displacement Comparison')
plt.legend()
plt.grid()
plt.show()
