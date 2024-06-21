# import csv
# import matplotlib.pyplot as plt

# def extract_data_and_plot(file_path, output_path):
#     print("writing to: ", output_path)

#     with open(file_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         i = 0
#         for row in reader:
#             joined_row = ''.join(row)
#             timestamp = joined_row.split("st_x:")[0].replace("timestamp:", "").strip()
#             st_x = float(joined_row.split("st_x:")[1].split()[0])
#             st_y = float(joined_row.split("st_y:")[1].split()[0])
#             st_z = float(joined_row.split("st_z:")[1].split()[0])
#             pi_x = float(joined_row.split("pi_x:")[1].split()[0])
#             pi_y = float(joined_row.split("pi_y:")[1].split()[0])
#             pi_z = float(joined_row.split("pi_z:")[1].split()[0])
#             filename = f'{i}_{timestamp}.webp'
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             st_xs, st_ys, st_zs = [st_x], [st_y], [st_z]
#             pi_xs, pi_ys, pi_zs = [pi_x], [pi_y], [pi_z]
#             ax.scatter(st_xs, st_ys, st_zs, c='r', label='ST')  # Red for ST
#             ax.scatter(pi_xs, pi_ys, pi_zs, c='b', label='PI')  # Blue for PI
#             ax.set_xlabel('X')
#             ax.set_ylabel('Y')
#             ax.set_zlabel('Z')
#             ax.set_title(f'Position Plot at {timestamp}')
#             ax.legend()
#             plt.savefig(f'{output_path}/{filename}')
#             plt.close()
#             i+=1
#         print("Done writing to: ", output_path)

# # Example usage:
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\movement_recording.csv"
# output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\movement_frames"
# extract_data_and_plot(file_path, output_path)


# import csv
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def extract_data_and_plot(file_path, output_path):
#     print("writing to: ", output_path)

#     timestamps = []
#     st_xs, st_ys, st_zs = [], [], []
#     pi_xs, pi_ys, pi_zs = [], [], []

#     # First pass to gather all data and determine axis limits
#     print("First pass to gather all data and determine axis limits")   
#     with open(file_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             joined_row = ''.join(row)
#             timestamp = joined_row.split("st_x:")[0].replace("timestamp:", "").strip()
#             st_x = float(joined_row.split("st_x:")[1].split()[0])
#             st_y = float(joined_row.split("st_y:")[1].split()[0])
#             st_z = float(joined_row.split("st_z:")[1].split()[0])
#             pi_x = float(joined_row.split("pi_x:")[1].split()[0])
#             pi_y = float(joined_row.split("pi_y:")[1].split()[0])
#             pi_z = float(joined_row.split("pi_z:")[1].split()[0])
            
#             timestamps.append(timestamp)
#             st_xs.append(st_x)
#             st_ys.append(st_y)
#             st_zs.append(st_z)
#             pi_xs.append(pi_x)
#             pi_ys.append(pi_y)
#             pi_zs.append(pi_z)

#     # Determine axis limits and ensure they are not identical
#     def adjust_limits(min_val, max_val, epsilon=1e-6):
#         if min_val == max_val:
#             min_val -= epsilon
#             max_val += epsilon
#         return min_val, max_val

#     x_min, x_max = adjust_limits(min(min(st_xs), min(pi_xs)), max(max(st_xs), max(pi_xs)))
#     y_min, y_max = adjust_limits(min(min(st_ys), min(pi_ys)), max(max(st_ys), max(pi_ys)))
#     z_min, z_max = adjust_limits(min(min(st_zs), min(pi_zs)), max(max(st_zs), max(pi_zs)))

#     # Prepare plots in batches
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_min, z_max)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     print ("Second pass to plot data")
#     for i, timestamp in enumerate(timestamps):
#         ax.cla()  # Clear the previous scatter points but keep axis limits and labels
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#         ax.set_zlim(z_min, z_max)
        
#         ax.scatter([st_xs[i]], [st_ys[i]], [st_zs[i]], c='r', label='ST')  # Red for ST
#         ax.scatter([pi_xs[i]], [pi_ys[i]], [pi_zs[i]], c='b', label='PI')  # Blue for PI
        
#         ax.set_title(f'Position Plot at {timestamp}')
#         ax.legend()
        
#         filename = f'{i}_{timestamp}.webp'
#         plt.savefig(f'{output_path}/{filename}')

#     plt.close(fig)
#     print("Done writing to: ", output_path)

# # Example usage:
# file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\movement_recording.csv"
# output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\movement_frames"
# extract_data_and_plot(file_path, output_path)




import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def extract_data_and_plot(file_path, output_path, batch_size=10):
    print("writing to:", output_path)

    timestamps = []
    st_xs, st_ys, st_zs = [], [], []
    pi_xs, pi_ys, pi_zs = [], [], []

    # First pass to gather all data and determine axis limits
    print("First pass to gather all data and determine axis limits")
    previous_timestamp = None
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            joined_row = ''.join(row)
            timestamp = float(joined_row.split("st_x:")[0].replace("timestamp:", "").strip())
            if previous_timestamp is not None and (timestamp - previous_timestamp) < 0.032:
                continue
            previous_timestamp = timestamp
            
            st_x = float(joined_row.split("st_x:")[1].split()[0])
            st_y = float(joined_row.split("st_y:")[1].split()[0])
            st_z = float(joined_row.split("st_z:")[1].split()[0])
            pi_x = float(joined_row.split("pi_x:")[1].split()[0])
            pi_y = float(joined_row.split("pi_y:")[1].split()[0])
            pi_z = float(joined_row.split("pi_z:")[1].split()[0])
            
            timestamps.append(timestamp)
            st_xs.append(st_x)
            st_ys.append(st_y)
            st_zs.append(st_z)
            pi_xs.append(pi_x)
            pi_ys.append(pi_y)
            pi_zs.append(pi_z)  # Corrected this line

    # Determine axis limits and ensure they are not identical
    def adjust_limits(min_val, max_val, epsilon=1e-6):
        if min_val == max_val:
            min_val -= epsilon
            max_val += epsilon
        return min_val, max_val

    x_min, x_max = adjust_limits(min(min(st_xs), min(pi_xs)), max(max(st_xs), max(pi_xs)))
    y_min, y_max = adjust_limits(min(min(st_ys), min(pi_ys)), max(max(st_ys), max(pi_ys)))
    z_min, z_max = adjust_limits(min(min(st_zs), min(pi_zs)), max(max(st_zs), max(pi_zs)))

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    #  print number of plots to make
    print(f"Number of plots to make: {len(timestamps)}")
    # Prepare plots in batches
    print("Second pass to plot data")
    num_plots = len(timestamps)
    for batch_start in range(0, num_plots, batch_size):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for i in range(batch_start, min(batch_start + batch_size, num_plots)):
            timestamp = timestamps[i]
            ax.cla()  # Clear the previous scatter points but keep axis limits and labels
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            ax.scatter([st_xs[i]], [st_ys[i]], [st_zs[i]], c='r', label='ST')  # Red for ST
            ax.scatter([pi_xs[i]], [pi_ys[i]], [pi_zs[i]], c='b', label='PI')  # Blue for PI
            
            ax.set_title(f'Position Plot at {timestamp}')
            ax.legend()
            
            filename = f'{i}_{timestamp}.webp'
            plt.savefig(f'{output_path}/{filename}')

        plt.close(fig)
    print("Done writing to:", output_path)

# Example usage:
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\movement_recording.csv"
output_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-18_45\movement_frames"
extract_data_and_plot(file_path, output_path)

