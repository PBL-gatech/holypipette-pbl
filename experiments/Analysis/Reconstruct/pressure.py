import csv
import matplotlib.pyplot as plt
from collections import deque

def extract_data_and_plot(file_path):
    # i = 0
    pressure_vals = deque(maxlen=100)

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            joined_row = ''.join(row)
            timestamp = joined_row.split("pressure:")[0].replace("timestamp:","")
            joined_row = joined_row.split('pressure:')[1]
            pressure_val = joined_row.split('resistance', 1)[0]
            pressure_vals.append(float(pressure_val))
            indices = list(range(len(pressure_vals)))

            plt.figure()
            plt.plot(indices, list(pressure_vals))
            plt.xlabel('Index')
            plt.ylabel('Pressure')
            plt.title('Pressure Plot')
            plt.savefig(f'./data/pressure_frames/{timestamp}.png')
            plt.close()
            # i += 1

# Example usage:
extract_data_and_plot('./data/graph_recording.csv')
