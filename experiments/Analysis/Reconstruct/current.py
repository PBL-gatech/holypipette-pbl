import csv
import matplotlib.pyplot as plt

def extract_data_and_plot(file_path):
    i = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row)
            # Since the data is not properly split by commas due to array-like strings, we need to handle it manually
            # Join the row into a single string and then split by '],', which is the end of each array
            joined_row = ''.join(row)
            timestamp = joined_row.split("pressure:")[0].replace("timestamp:","")
            joined_row = joined_row.split('time:')[1]
            time_vals = joined_row.split('current:')[0]
            time_vals_list = [float(val) for val in time_vals.strip('[]').split()]
            # print(time_vals_list)

            current_vals = joined_row.split('current:')[1]
            current_vals_list = [float(val) for val in current_vals.strip('[]').split()]
            
            # Plotting
            plt.figure()
            plt.plot(time_vals_list, current_vals_list)
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.title('Time vs Current Plot')
            plt.savefig(f'./data/current_frames/{i}_{timestamp}.png')
            plt.close()

            i += 1

# Example usage:
extract_data_and_plot('./data/graph_recording.csv')
