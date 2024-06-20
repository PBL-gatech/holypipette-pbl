import csv
import matplotlib.pyplot as plt

def extract_data_and_plot(file_path, output_path):
    print("writing to: ", output_path)
    i = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row)
            # Since the data is not properly split by commas due to array-like strings, we need to handle it manually
            # Join the row into a single string and then split by '],', which is the end of each array
            joined_row = ''.join(row)
            timestamp = joined_row.split("pressure:")[0].replace("timestamp:","").strip()   
            joined_row = joined_row.split('time:')[1]
            # print(joined_row)
            time_vals = joined_row.split('current:')[0].replace("[","").replace("]","")
            time_vals_list = [float(val) for val in time_vals.strip('[]').split()]
            # print(time_vals_list)

            current_vals = joined_row.split('current:')[1]
            current_vals_list = [float(val) for val in current_vals.strip('[]').split()]
            filename = f'{i}_{timestamp}.webp'
            # Plotting
            plt.figure()
            plt.plot(time_vals_list, current_vals_list)
            plt.xlabel('Time')
            plt.ylabel('Current')
            plt.title('Time vs Current Plot')
            plt.savefig(f'{output_path}/{filename}')
            plt.close()
            i += 1
    print("Done writing to: ", output_path)

# Example usage:
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\graph_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\current_frames"
extract_data_and_plot(file_path, output_path)
