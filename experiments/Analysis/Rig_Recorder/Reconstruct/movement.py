
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def extract_data_and_plot(file_path, output_path):
    #create output path if it doesn't exist
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    print("Writing to:", output_path)

    
    # Load and preprocess the data
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        line_data = {}
        try:
            parts = line.split()
            line_data['timestamp'] = float(parts[0].split(":")[1])
            line_data['st_x'] = float(parts[1].split(":")[1])
            line_data['st_y'] = float(parts[2].split(":")[1])
            line_data['st_z'] = float(parts[3].split(":")[1])
            line_data['pi_x'] = float(parts[4].split(":")[1])
            line_data['pi_y'] = float(parts[5].split(":")[1])
            line_data['pi_z'] = float(parts[6].split(":")[1])
            data.append(line_data)
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line} - {e}")

    df = pd.DataFrame(data)
    
    # Determine axis limits and ensure they are not identical
    def adjust_limits(min_val, max_val, epsilon=1e-6):
        if min_val == max_val:
            min_val -= epsilon
            max_val += epsilon
        return min_val, max_val

    x_min, x_max = adjust_limits(df['st_x'].min(), df['st_x'].max())
    y_min, y_max = adjust_limits(df['st_y'].min(), df['st_y'].max())
    z_min, z_max = adjust_limits(df['st_z'].min(), df['st_z'].max())
    pi_x_min, pi_x_max = adjust_limits(df['pi_x'].min(), df['pi_x'].max())
    pi_y_min, pi_y_max = adjust_limits(df['pi_y'].min(), df['pi_y'].max())
    pi_z_min, pi_z_max = adjust_limits(df['pi_z'].min(), df['pi_z'].max())

    x_min, x_max = min(x_min, pi_x_min), max(x_max, pi_x_max)
    y_min, y_max = min(y_min, pi_y_min), max(y_max, pi_y_max)
    z_min, z_max = min(z_min, pi_z_min), max(z_max, pi_z_max)

    # Prepare plots in batches
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    print("Second pass to plot data")
    prev_timestamp = None

    for i, row in df.iterrows():
        timestamp = row['timestamp']
        if prev_timestamp and (timestamp - prev_timestamp) < 0.032:
            continue
        
        ax.cla()  # Clear the previous scatter points but keep axis limits and labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        ax.scatter(row['st_x'], row['st_y'], row['st_z'], c='r', label='ST')  # Red for ST
        ax.scatter(row['pi_x'], row['pi_y'], row['pi_z'], c='b', label='PI')  # Blue for PI
        
        ax.set_title(f'Position Plot at {timestamp}')
        ax.legend()
        
        filename = f'{i}_{timestamp}.webp'
        #create output path if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(f'{output_path}/{filename}')
        
        prev_timestamp = timestamp

    plt.close(fig)
    print("Done writing to:", output_path)


# Example usage:
file_path =r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_08_06-18_11\movement_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_08_06-18_11\movement_frames"
extract_data_and_plot(file_path, output_path)
