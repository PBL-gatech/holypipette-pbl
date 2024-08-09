import csv
import matplotlib.pyplot as plt
from datetime import datetime

def extract_data_and_plot(file_path, output_path):
    print("Writing to:", output_path)
    i = 0
    previous_timestamp = None

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Join the row into a single string and then split by '],', which is the end of each array
            joined_row = ''.join(row)
            timestamp_str = joined_row.split("pressure:")[0].replace("timestamp:", "").strip()
            joined_row = joined_row.split('time:')[1]

            # Convert timestamp to float (UNIX timestamp)
            current_timestamp = float(timestamp_str)

            # Check if the difference between timestamps is less than 0.032 seconds
            if previous_timestamp and (current_timestamp - previous_timestamp) < 0.032:
                continue

            time_vals = joined_row.split('current:')[0].replace("[", "").replace("]", "")
            time_vals_list = [float(val) for val in time_vals.strip('[]').split()]

            current_vals = joined_row.split('current:')[1]
            current_vals_list = [float(val) for val in current_vals.strip('[]').split()]
            filename = f'{i}_{timestamp_str}.webp'

            # Plotting
            plt.figure()
            plt.plot(time_vals_list, current_vals_list)
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.title('Time vs Current Plot')
            plt.savefig(f'{output_path}/{filename}')
            plt.close()
            
            # Update the previous timestamp
            previous_timestamp = current_timestamp
            i += 1

    print("Done writing to:", output_path)

# Example usage:
file_path =r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_08_06-18_11\graph_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_08_06-18_11\current_frames"
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\graph_recording.csv"
# output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\current_frames"
extract_data_and_plot(file_path, output_path)
