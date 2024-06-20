import csv
import matplotlib.pyplot as plt

def extract_data_and_plot(file_path,output_path):
    print("writing to: ", output_path)
    i = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            joined_row = ''.join(row)
            timestamp = joined_row.split("pressure:")[0].replace("timestamp:","").strip()
            joined_row = joined_row.split('resistance:')[1]
            resistance_vals = joined_row.split('time:')[0].replace("[","").replace("]","")
            resistance_vals_list = [float(val) for val in resistance_vals.strip('[]').split()]
            # print(time_vals_list)
            # print(resistance_vals_list)
            resistance_length_with_indices = [(index, val) for index, val in enumerate(resistance_vals_list)]
            # print(resistance_length_with_indices)
            # Plotting
            filename = f'{i}_{timestamp}.webp'
            plt.figure()
            plt.plot([index for index, _ in resistance_length_with_indices], [val for _, val in resistance_length_with_indices])
            plt.xlabel('')
            plt.ylabel('Resistance')
            plt.title('Resistance Plot')
            plt.savefig(f'{output_path}/{filename}')
            plt.close()

            i += 1
    print("Done writing to: ", output_path)

file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\graph_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\resistance_frames"
# Example usage:
extract_data_and_plot(file_path, output_path)
