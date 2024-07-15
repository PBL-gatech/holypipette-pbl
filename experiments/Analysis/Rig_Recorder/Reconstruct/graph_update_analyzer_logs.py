# This script assesses how accurate the update method in EPhysGraph is in the graph.py file. to use you need to uncomment #logging.debug('graph updated') in the update method in graph.py on approximately line 417
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder'
new_file_path = directory + '\logs.csv'
new_data = pd.read_csv(new_file_path, on_bad_lines='skip')

# Convert the 'Time(HH:MM:SS)' column to datetime and add the milliseconds from 'Time(ms)'
    # Ensure columns are treated as strings
new_data['Time(HH:MM:SS)'] = new_data['Time(HH:MM:SS)'].astype(str)
new_data['Time(ms)'] = new_data['Time(ms)'].astype(str)
new_data['Timestamp'] = pd.to_datetime(new_data['Time(HH:MM:SS)'] + '.' + new_data['Time(ms)'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

# Find the latest event with INFO and 'Program Started'
latest_program_started_idx = new_data[(new_data['Level'] == 'INFO') & (new_data['Message'].str.contains('Program Started'))]['Timestamp'].idxmax()

# Filter the data to only include events after the latest 'Program Started'
filtered_data = new_data.loc[latest_program_started_idx + 1:]

# Filter the rows where "Message" contains "graph updated"
graph_updated_rows_filtered = filtered_data[filtered_data['Message'].str.contains("graph updated", na=False)]

# Sort the dataframe by the timestamp to ensure the points are ordered properly
graph_updated_rows_sorted_filtered = graph_updated_rows_filtered.sort_values(by='Timestamp')

# Remove duplicate "graph updated" events with the exact same timestamp
graph_updated_unique_filtered = graph_updated_rows_sorted_filtered.drop_duplicates(subset='Timestamp', keep='first')

# Extract the unique times in milliseconds
unique_times_ms_filtered = graph_updated_unique_filtered['Timestamp'].astype('int64') // 10**6

# Calculate the intervals between successive unique "graph updated" times
unique_intervals_ms_filtered = [unique_times_ms_filtered.iloc[i] - unique_times_ms_filtered.iloc[i - 1] for i in range(1, len(unique_times_ms_filtered))]

# Calculate the running average of the unique intervals
window_size = 10
unique_running_average_filtered = np.convolve(unique_intervals_ms_filtered, np.ones(window_size) / window_size, mode='valid')

# Calculate the average frame rate (average of the unique intervals)
average_framerate_ms_filtered = sum(unique_intervals_ms_filtered) / len(unique_intervals_ms_filtered) if unique_intervals_ms_filtered else None

# Find the closest index for the "closing GUI" event in the unique times list
closing_gui_timestamp = filtered_data[filtered_data['Message'].str.contains('closing GUI', na=False)]['Timestamp'].min()
closest_index = (unique_times_ms_filtered - int(closing_gui_timestamp.timestamp() * 1000)).abs().argmin()

# Plot the unique intervals between "graph updated" events with running average, average frame rate, and "closin
plt.figure(figsize=(10, 6))
plt.plot(unique_intervals_ms_filtered, marker='o', linestyle='-', label='Intervals')
plt.plot(range(window_size - 1, len(unique_intervals_ms_filtered)), unique_running_average_filtered, color='red', linestyle='-', label='Running Average')
plt.axhline(y=average_framerate_ms_filtered, color='green', linestyle='--', label=f'Average Interval: {average_framerate_ms_filtered:.2f} ms')
plt.axvline(x=closest_index, color='blue', linestyle='--', label='Closing GUI')
plt.title('Intervals between Unique "Graph Updated" Events with Running Average, Average Interval, and Closing GUI Event')
plt.xlabel('Event Number')
plt.ylabel('Interval (ms)')
plt.legend()
plt.grid(True)
plt.show()