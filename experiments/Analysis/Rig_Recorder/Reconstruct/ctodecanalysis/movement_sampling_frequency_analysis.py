import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter  # Import savgol_filter

# Load the CSV data
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_14-15_55\movement_recording.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_15-16_19\movement_recording.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_15-16_25\movement_recording.csv"
# file_path  = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_15-17_20\movement_recording_truncated.csv"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_18-17_18\movement_recording.csv"
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_18-17_35\movement_recording.csv"



# Use pandas to read the data directly, specifying the delimiter as whitespace
data = pd.read_csv(file_path, sep='\s+', header=None)

# Rename columns based on the expected key:value format
data.columns = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']

# Split each column by ':' to extract the numeric value
for col in data.columns:
    data[col] = data[col].str.split(':').str[1].astype(float)

# Get initial timestamp
initial_timestamp = data['timestamp'].iloc[0]
data['timestamp'] -= initial_timestamp

# Calculate time differences between consecutive rows
data['time_diff'] = data['timestamp'].diff().fillna(method='bfill')

# Calculate sampling frequency (inverse of time difference)
data['sampling_frequency'] = 1 / data['time_diff']

# Calculate overall average sampling frequency and standard deviation
avg_sampling_frequency = data['sampling_frequency'].mean()
std_sampling_frequency = data['sampling_frequency'].std()

# Zero pipette movement
initial_coords = data.iloc[0][['pi_x', 'pi_y', 'pi_z']].values
data['pi_x'] -= initial_coords[0]
data['pi_y'] -= initial_coords[1]
data['pi_z'] -= initial_coords[2]

# Calculate the resultant displacement of the pipette
data['pipette_resultant'] = np.sqrt(data['pi_x']**2 + data['pi_y']**2 + data['pi_z']**2)

# Define a function to apply a low-pass Butterworth filter
def apply_low_pass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Set Butterworth filter parameters
butter_cutoff_frequency = 15  # Hz
sampling_rate = 1 / data['time_diff'].mean()  # Average sampling rate in Hz

# Apply the Butterworth low-pass filter to the pipette resultant displacement
butter_filtered_resultant = apply_low_pass_filter(data['pipette_resultant'].values, butter_cutoff_frequency, sampling_rate)

# Calculate the numerical derivative of the pipette resultant displacement (velocity)
data['pipette_resultant_derivative'] = data['pipette_resultant'].diff() / data['time_diff']

# Apply the Butterworth low-pass filter to the velocity derivative
butter_filtered_velocity = apply_low_pass_filter(data['pipette_resultant_derivative'].fillna(0).values, butter_cutoff_frequency, sampling_rate)

# Calculate the numerical derivative of the velocity (acceleration)
data['pipette_resultant_derivative_derivative'] = data['pipette_resultant_derivative'].diff() / data['time_diff']

# Apply the Butterworth low-pass filter to the acceleration derivative
butter_filtered_acceleration = apply_low_pass_filter(data['pipette_resultant_derivative_derivative'].fillna(0).values, butter_cutoff_frequency, sampling_rate)

# Define Savitzky-Golay filter parameters
# Choose window_length based on the sampling rate and desired smoothing
# It must be a positive odd integer
window_length = 10  # Example value; adjust as needed
polyorder = 3       # Polynomial order; typically 2 or 3

# Apply the Savitzky-Golay filter to the pipette resultant displacement
savgol_filtered_resultant = savgol_filter(data['pipette_resultant'].values, window_length=window_length, polyorder=polyorder)

# Calculate velocity using numerical derivative on SG filtered displacement (optional)
# Alternatively, compute velocity first and then apply SG filter
# Here, we follow the latter approach for consistency
savgol_velocity = savgol_filter(data['pipette_resultant_derivative'].fillna(0).values, window_length=window_length, polyorder=polyorder)

# Calculate acceleration using numerical derivative on SG filtered velocity
savgol_acceleration = savgol_filter(data['pipette_resultant_derivative_derivative'].fillna(0).values, window_length=window_length, polyorder=polyorder)

# Plotting the original and filtered data for comparison

# Plot the pipette's resultant displacement over time
plt.figure(figsize=(12, 7))
plt.plot(data['timestamp'], data['pipette_resultant'], label='Original Pipette Resultant Movement', color='k', alpha=0.5)
plt.plot(data['timestamp'], butter_filtered_resultant, label='Butterworth Filtered (15 Hz)', color='r')
plt.plot(data['timestamp'], savgol_filtered_resultant, label='Savitzky-Golay Filtered', color='b')
plt.xlabel('Timestamp (s)')
plt.ylabel('Resultant Displacement')
plt.title('Pipette Resultant Displacement: Original vs Butterworth vs Savitzky-Golay Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the numerical derivative of the pipette's resultant displacement (velocity)
plt.figure(figsize=(12, 7))
plt.plot(data['timestamp'], data['pipette_resultant_derivative'], label='Original Velocity (Derivative)', color='k', alpha=0.5)
plt.plot(data['timestamp'], butter_filtered_velocity, label='Butterworth Filtered Velocity (15 Hz)', color='r')
plt.plot(data['timestamp'], savgol_velocity, label='Savitzky-Golay Filtered Velocity', color='b')
plt.xlabel('Timestamp (s)')
plt.ylabel('Velocity (Derivative of Displacement)')
plt.title('Pipette Velocity: Original vs Butterworth vs Savitzky-Golay Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the numerical derivative of the velocity (acceleration)
plt.figure(figsize=(12, 7))
plt.plot(data['timestamp'], data['pipette_resultant_derivative_derivative'], label='Original Acceleration (Derivative of Velocity)', color='k', alpha=0.5)
plt.plot(data['timestamp'], butter_filtered_acceleration, label='Butterworth Filtered Acceleration (15 Hz)', color='r')
plt.plot(data['timestamp'], savgol_acceleration, label='Savitzky-Golay Filtered Acceleration', color='b')
plt.xlabel('Timestamp (s)')
plt.ylabel('Acceleration (Derivative of Velocity)')
plt.title('Pipette Acceleration: Original vs Butterworth vs Savitzky-Golay Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

