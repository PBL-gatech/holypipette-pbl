import csv
import matplotlib.pyplot as plt
from collections import deque

def extract_data_and_plot(file_path):

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            joined_row = ''.join(row)
            timestamp = joined_row.split("st_x:")[0].replace("timestamp:", "").strip()
            st_x = float(joined_row.split("st_x:")[1].split()[0])
            st_y = float(joined_row.split("st_y:")[1].split()[0])
            st_z = float(joined_row.split("st_z:")[1].split()[0])
            pi_x = float(joined_row.split("pi_x:")[1].split()[0])
            pi_y = float(joined_row.split("pi_y:")[1].split()[0])
            pi_z = float(joined_row.split("pi_z:")[1].split()[0])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            st_xs, st_ys, st_zs = [st_x], [st_y], [st_z]
            pi_xs, pi_ys, pi_zs = [pi_x], [pi_y], [pi_z]
            ax.scatter(st_xs, st_ys, st_zs, c='r', label='ST')  # Red for ST
            ax.scatter(pi_xs, pi_ys, pi_zs, c='b', label='PI')  # Blue for PI
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Position Plot at {timestamp}')
            ax.legend()
            plt.savefig(f'./data/position_frames/{i}_{timestamp.strip()}.png')
            plt.close()
            i+=1

# Example usage:
extract_data_and_plot('./data/movement_recording.csv')

