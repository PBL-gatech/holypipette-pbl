# import csv
# import matplotlib.pyplot as plt
# from collections import deque

# def extract_data_and_plot(file_path,output_path):
#     print("writing to: ", output_path)
#     i = 0
#     pressure_vals = deque(maxlen=100)

#     with open(file_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             joined_row = ''.join(row)
#             timestamp = joined_row.split("pressure:")[0].replace("timestamp:","")
#             joined_row = joined_row.split('pressure:')[1]
#             pressure_val = joined_row.split('resistance', 1)[0]
#             pressure_vals.append(float(pressure_val))
#             indices = list(range(len(pressure_vals)))
#             filename = f'{i}_{timestamp}.webp'
#             plt.figure()
#             plt.plot(indices, list(pressure_vals))
#             plt.xlabel('Index')
#             plt.ylabel('Pressure')
#             plt.title('Pressure Plot')
#             plt.savefig(f'{output_path}/{filename}')
#             plt.close()
#             i += 1
#     print("Done writing to: ", output_path)

# # Example usage:
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\graph_recording.csv"
# output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\pressure_frames"
# extract_data_and_plot(file_path, output_path)

import csv
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

def extract_data_and_plot(file_path, output_path):
    print("Writing to:", output_path)
    i = 0
    previous_timestamp = None
    pressure_vals = deque(maxlen=100)

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            joined_row = ''.join(row)
            timestamp_str = joined_row.split("pressure:")[0].replace("timestamp:", "").strip()
            joined_row = joined_row.split('pressure:')[1]
            pressure_val = joined_row.split('resistance', 1)[0]
            
            # Convert timestamp to float (UNIX timestamp)
            current_timestamp = float(timestamp_str)
            
            # Check if the difference between timestamps is less than 0.032 seconds
            if previous_timestamp and (current_timestamp - previous_timestamp) < 0.032:
                continue

            pressure_vals.append(float(pressure_val))
            indices = list(range(len(pressure_vals)))
            filename = f'{i}_{timestamp_str}.webp'
            
            # Plotting
            plt.figure()
            plt.plot(indices, list(pressure_vals))
            plt.xlabel('Index')
            plt.ylabel('Pressure')
            plt.title('Pressure Plot')
            plt.savefig(f'{output_path}/{filename}')
            plt.close()
            
            # Update the previous timestamp
            previous_timestamp = current_timestamp
            i += 1

    print("Done writing to:", output_path)

# Example usage:
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\graph_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\pressure_frames"
extract_data_and_plot(file_path, output_path)
