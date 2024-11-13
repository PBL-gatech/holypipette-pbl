import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_11_11-13_51\movement_recording.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-14_18\movement_recording.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-16_23\movement_recording.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-16_30\movement_recording_truncated.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-16_52\movement_recording_truncated.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-17_17\movement_recording_truncated.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-17_41\movement_recording_truncated.csv"
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_12-17_46\movement_recording_truncated.csv"

# Use pandas to read the data directly, specifying the delimiter as whitespace
data = pd.read_csv(file_path, sep='\s+', header=None)

# Rename columns based on the expected key:value format
data.columns = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']

# Split each column by ':' to extract the numeric value
for col in data.columns:
    data[col] = data[col].str.split(':').str[1].astype(float)

# get initial timestamp
initial_timestamp = data['timestamp'].iloc[0]
# subtract the initial timestamp from all subsequent timestamps
data['timestamp'] -= initial_timestamp
# Calculate time differences between consecutive rows
data['time_diff'] = data['timestamp'].diff()

# Calculate sampling frequency (inverse of time difference)
data['sampling_frequency'] = 1 / data['time_diff']

# Calculate overall average sampling frequency and standard deviation
avg_sampling_frequency = data['sampling_frequency'].mean()
std_sampling_frequency = data['sampling_frequency'].std()

# Plot sampling frequency over time
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'], data['sampling_frequency'], label='Sampling Frequency (Hz)')
plt.axhline(y=avg_sampling_frequency, color='r', linestyle='--', label=f'Average Sampling Frequency: {avg_sampling_frequency:.2f} Hz')
plt.fill_between(data['timestamp'], avg_sampling_frequency - std_sampling_frequency, avg_sampling_frequency + std_sampling_frequency, color='r', alpha=0.1, label=f'Standard Deviation: ±{std_sampling_frequency:.2f} Hz')

# Add labels and title
plt.xlabel('Timestamp (s)')
plt.ylabel('Sampling Frequency (Hz)')
plt.title('Sampling Frequency Over Time')
plt.legend()
plt.grid(True)
plt.show()

# zero pipette movement
#obtain the initial coordinates of the pipette
initial_coords = data.iloc[0][['pi_x', 'pi_y', 'pi_z']].values
# subtract the initial coordinates from all subsequent coordinates
data['pi_x'] -= initial_coords[0]
data['pi_y'] -= initial_coords[1]
data['pi_z'] -= initial_coords[2]

# Calculate the resultant displacement of the pipette
data['pipette_resultant'] = np.sqrt(data['pi_x']**2 + data['pi_y']**2 + data['pi_z']**2)

# Plot the pipette's resultant displacement over time
plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'], data['pipette_resultant'], label='Pipette Resultant Movement', color='k')
plt.plot(data['timestamp'], data['pi_x'], label='Pipette X movement', color='r')
plt.plot(data['timestamp'], data['pi_y'], label='Pipette Y movement', color='g')
plt.plot(data['timestamp'], data['pi_z'], label='Pipette Z movement', color='b')

# Add labels and title
plt.xlabel('Timestamp (s)')
plt.ylabel('Resultant Displacement')
plt.title('Pipette Resultant Displacement Over Time')
plt.legend()
plt.grid(True)
plt.show()
