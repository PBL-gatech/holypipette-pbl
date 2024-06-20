import csv
import matplotlib.pyplot as plt
from collections import deque

def extract_data_and_plot(file_path):
    i = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            joined_row = ''.join(row)
            timestamp = joined_row.split("pressure:")[0].replace("timestamp:","")
            joined_row = joined_row.split('resistance:')[1]
            resistance_vals = joined_row.split('time:')[0]
            resistance_vals_list = deque([float(val) for val in resistance_vals.strip('[]').split()], maxlen=100)
            # print(time_vals_list)
            # print(resistance_vals_list)
            resistance_length_with_indices = [(index, val) for index, val in enumerate(resistance_vals_list)]
            # print(resistance_length_with_indices)
            # Plotting
            plt.figure()
            plt.plot([index for index, _ in resistance_length_with_indices], [val for _, val in resistance_length_with_indices])
            plt.xlabel('')
            plt.ylabel('Resistance')
            plt.title('Resistance Plot')
            plt.savefig(f'./data/resistance_frames/{i}_{timestamp}.png')
            plt.close()

            i += 1

# Example usage:
extract_data_and_plot('./data/graph_recording.csv')
