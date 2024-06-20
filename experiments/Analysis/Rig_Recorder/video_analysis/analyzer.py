# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
# from datetime import datetime
# import re
# import concurrent.futures

# def find_latest_camera_frames_dir(parent_dir):
#     latest_time = None
#     latest_dir = None

#     for root, dirs, files in os.walk(parent_dir):
#         if 'camera_frames' in dirs:
#             dir_path = os.path.join(root, 'camera_frames')
#             parent_dir_name = os.path.basename(root)

#             try:
#                 dir_time = datetime.strptime(parent_dir_name, '%Y_%m_%d-%H_%M')
#                 if latest_time is None or dir_time > latest_time:
#                     latest_time = dir_time
#                     latest_dir = dir_path
#             except ValueError:
#                 continue
#     # print(f"Latest camera frames directory found: {latest_dir}")
#     # print(f"Latest camera frames directory time: {latest_time}")
#     # print(f"parent directory:{parent_dir_name}")
#     # print(f"intended save directory:{root}")
#     output_dir = latest_dir.split('camera_frames')[0] + 'output_videos'
#     print(f"output directory:{output_dir}")
#     return latest_dir, output_dir

# def load_image(file):
#     return cv2.imread(file)

# def create_videos_from_directory(parent_directory):
#     # Create output directory if it doesn't exist
#     input_dir, output_dir = find_latest_camera_frames_dir(parent_directory)
#     os.makedirs(output_dir, exist_ok=True)

#     # Define the pattern to extract frame index and timestamp from file names
#     pattern = re.compile(r'(\d+)_(\d+\.\d+)\.webp')

#     # Find all webp files in the directory and extract frame info
#     frames = []
#     image_files = []
#     for filename in os.listdir(input_dir):
#         match = pattern.match(filename)
#         if match:
#             frame_index = int(match.group(1))
#             timestamp = float(match.group(2))
#             frames.append((frame_index, timestamp, filename))

#     # Sort frames and image files by timestamp
#     frames.sort(key=lambda x: x[1])
#     image_files = [os.path.join(input_dir, frame[2]) for frame in frames]
#     print("Image files collected")

#     # Load the images using multi-threading for faster performance
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         images = list(executor.map(load_image, image_files))
#     print("Images loaded")

#     # Define video names (initially without frame counts)
#     video_name_variable_fps = os.path.join(output_dir, "output_video_variable_fps_temp.mp4")
#     video_name_constant_fps = os.path.join(output_dir, "output_video_constant_fps_temp.mp4")

#     # Base frame rate for the constant FPS video
#     base_frame_rate = 30

#     # Define the video writer
#     height, width, _ = images[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     print("Starting to write videos")

#     print("starting to write variable fps video")
#     # Video writer for variable FPS
#     video_variable_fps = cv2.VideoWriter(video_name_variable_fps, fourcc, base_frame_rate, (width, height))

#     # Determine the duration each frame should be displayed for variable FPS video
#     durations = []
#     previous_time = 0
#     for frame in frames:
#         current_time = frame[1]
#         durations.append(current_time - previous_time)
#         previous_time = current_time

#     # Add frames to the variable FPS video
#     variable_frame_count = 0
#     for img, duration in zip(images, durations):
#         frame_count = int(duration * base_frame_rate)
#         if frame_count == 0:
#             frame_count = 1  # Ensure at least one frame is written
#         for _ in range(frame_count):
#             video_variable_fps.write(img)
#             variable_frame_count += 1

#     # Release the video writer for variable FPS
#     video_variable_fps.release()
#     print(f"Variable FPS video saved with {variable_frame_count} frames")
#     print("starting to write constant fps video")
#     # Video writer for constant FPS
#     video_constant_fps = cv2.VideoWriter(video_name_constant_fps, fourcc, base_frame_rate, (width, height))

#     # Add frames to the constant FPS video (each frame is displayed for the same duration)
#     constant_frame_count = len(images)
#     for img in images:
#         video_constant_fps.write(img)

#     # Release the video writer for constant FPS
#     video_constant_fps.release()
#     print(f"Constant FPS video saved with {constant_frame_count} frames")

#     # Define final video names with frame counts
#     final_video_name_variable_fps = os.path.join(output_dir, f"output_video_variable_fps_{variable_frame_count}_frames.mp4")
#     final_video_name_constant_fps = os.path.join(output_dir, f"output_video_constant_fps_{constant_frame_count}_frames.mp4")

#     # Rename temporary files to final names
#     os.rename(video_name_variable_fps, final_video_name_variable_fps)
#     os.rename(video_name_constant_fps, final_video_name_constant_fps)

#     print("Videos saved at:")
#     print(final_video_name_variable_fps)
#     print(final_video_name_constant_fps)

# def process_images_and_plot(parent_directory, output_filename='framerate_plot.png'):
#     input_dir, output_dir = find_latest_camera_frames_dir(parent_directory)
#     os.makedirs(output_dir, exist_ok=True)
#     # Use a generator to avoid loading all filenames into memory at once
#     def get_file_names(directory):
#         for filename in os.listdir(directory):
#             if filename.endswith('.webp'):
#                 yield filename
    
#     # Extract frame numbers and timestamps from filenames using the generator
#     frames_timestamps = [
#         (int(name.split('_')[0]), float(name.split('_')[1].split('.webp')[0]))
#         for name in get_file_names(input_dir)
#     ]
    
#     # Convert to a DataFrame
#     df = pd.DataFrame(frames_timestamps, columns=['frame', 'timestamp'])
    
#     # Sort by frame number to ensure correct order
#     df = df.sort_values(by='frame').reset_index(drop=True)
    
#     # Calculate time intervals (differences between consecutive timestamps)
#     df['time_interval'] = df['timestamp'].diff()
    
#     # Calculate framerate (frames per second)
#     df['framerate'] = 1 / df['time_interval']
    
#     # Remove the first row as it will have NaN for time_interval and framerate
#     df = df.dropna()
    
#     # Calculate the average framerate
#     average_framerate = df['framerate'].mean()

#     # Calculate the running average of the framerate
#     window_size = 20  # You can adjust the window size for the running average
#     df['running_avg_framerate'] = df['framerate'].rolling(window=window_size).mean()

#     # Calculate the cumulative average framerate
#     df['cumulative_avg_framerate'] = df['framerate'].expanding().mean()

#     # Plot framerate over cumulative time with horizontal lines at 60fps and 30fps
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['timestamp'], df['framerate'], marker='o', label='Framerate')
#     plt.plot(df['timestamp'], df['running_avg_framerate'], color='orange', linestyle='-', label='Running Average Framerate')
#     plt.plot(df['timestamp'], df['cumulative_avg_framerate'], color='purple', linestyle='-', label='Cumulative Average Framerate')
#     plt.axhline(y=60, color='r', linestyle='--', label='60 fps')
#     plt.axhline(y=30, color='g', linestyle='--', label='30 fps')
#     plt.axhline(y=average_framerate, color='b', linestyle='--', label=f'Overall Average Framerate: {average_framerate:.2f} fps')
#     plt.title('Framerate Over Time with Horizontal Lines at 60fps, 30fps, and Average Framerates')
#     plt.xlabel('Timestamp (s)')
#     plt.ylabel('Framerate (frames per second)')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot to the output directory
#     output_path = os.path.join(output_dir, output_filename)
#     plt.savefig(output_path)
#     plt.close()

#     # Print the average framerate
#     print(f'Overall Average Framerate: {average_framerate:.2f} fps')


# # Example usage
# input_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data'
# # create_videos_from_directory(input_directory)
# process_images_and_plot(input_directory)




# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# def process_images_and_plot(input_directory, output_directory, output_filename):
#     # Use a generator to avoid loading all filenames into memory at once
#     def get_file_names(directory):
#         for filename in os.listdir(directory):
#             if filename.endswith('.webp'):
#                 yield filename
    
#     # Extract frame numbers and timestamps from filenames using the generator
#     frames_timestamps = [
#         (int(name.split('_')[0]), float(name.split('_')[1].split('.webp')[0]))
#         for name in get_file_names(input_directory)
#     ]
    
#     # Convert to a DataFrame
#     df = pd.DataFrame(frames_timestamps, columns=['frame', 'timestamp'])
    
#     # Sort by frame number to ensure correct order
#     df = df.sort_values(by='frame').reset_index(drop=True)
    
#     # Calculate time intervals (differences between consecutive timestamps)
#     df['time_interval'] = df['timestamp'].diff()
    
#     # Calculate framerate (frames per second)
#     df['framerate'] = 1 / df['time_interval']
    
#     # Remove the first row as it will have NaN for time_interval and framerate
#     df = df.dropna()
    
#     # Calculate the average framerate
#     average_framerate = df['framerate'].mean()

#     # Calculate the running average of the framerate
#     window_size = 20  # You can adjust the window size for the running average
#     df['running_avg_framerate'] = df['framerate'].rolling(window=window_size).mean()

#     # Calculate the cumulative average framerate
#     df['cumulative_avg_framerate'] = df['framerate'].expanding().mean()

#     # Plot framerate over cumulative time with horizontal lines at 60fps and 30fps
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['timestamp'], df['framerate'], marker='o', label='Framerate')
#     plt.plot(df['timestamp'], df['running_avg_framerate'], color='orange', linestyle='-', label='Running Average Framerate')
#     plt.plot(df['timestamp'], df['cumulative_avg_framerate'], color='purple', linestyle='-', label='Cumulative Average Framerate')
#     plt.axhline(y=60, color='r', linestyle='--', label='60 fps')
#     plt.axhline(y=30, color='g', linestyle='--', label='30 fps')
#     plt.axhline(y=average_framerate, color='b', linestyle='--', label=f'Overall Average Framerate: {average_framerate:.2f} fps')
#     plt.title('Framerate Over Time with Horizontal Lines at 60fps, 30fps, and Average Framerates')
#     plt.xlabel('Timestamp (s)')
#     plt.ylabel('Framerate (frames per second)')
#     plt.legend()
#     plt.grid(True)

#     # Save the plot to the output directory
#     output_path = os.path.join(output_directory, output_filename)
#     plt.savefig(output_path)
#     plt.close()

#     # Print the average framerate
#     print(f'Average Framerate: {average_framerate:.2f} fps')




# # Example usage:
# input_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\camera_frames"
# output_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_videos"
# output_filename = 'framerate_plot.png'
# process_images_and_plot(input_directory, output_directory, output_filename)






# import os

# # Directory containing the .webp images
# image_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\camera_frames"

# # Output video file
# output_video = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_videos\output_video_constant_frame_rate_1.mp4"

# # Frame rate for the video
# frame_rate = 30

# # Function to extract the order number from the filename
# def get_order_number(filename):
#     return int(filename.split('_')[0])

# # Get all .webp files in the directory and sort them by the order number
# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.webp')],
#                      key=get_order_number)

# # Create an input file list for ffmpeg
# input_list_path = os.path.join(image_directory, 'input_images.txt')
# with open(input_list_path, 'w') as f:
#     for image_file in image_files:
#         f.write(f"file '{os.path.join(image_directory, image_file)}'\n")

# # Construct the ffmpeg command to create the video
# ffmpeg_command = [
    
#     'C:\\ffmpeg\\bin\\ffmpeg.exe', '-r', str(frame_rate), '-f', 'concat', '-safe', '0', '-i', input_list_path,
#     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video
# ]

# # Execute the ffmpeg command
# import subprocess
# subprocess.run(ffmpeg_command)



# import os
# import subprocess

# # # Directory containing the .webp images
# image_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\camera_frames"

# # # Output video file
# output_video = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_videos\output_video_variable_frame_rate.mp4"

# # Function to extract the order number and timestamp from the filename
# def get_order_and_timestamp(filename):
#     parts = filename.split('_')
#     order_number = int(parts[0])
#     timestamp = int(parts[1].split('.')[0])
#     return order_number, timestamp

# # Get all .webp files in the directory and sort them by the order number
# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.webp')],
#                      key=lambda x: get_order_and_timestamp(x)[0])

# # Calculate durations based on timestamps
# durations = []
# previous_timestamp = None
# for filename in image_files:
#     _, timestamp = get_order_and_timestamp(filename)
#     if previous_timestamp is not None:
#         duration = (timestamp - previous_timestamp) / 1000.0  # Convert to seconds
#         durations.append(duration)
#     previous_timestamp = timestamp

# # Ensure at least one duration to prevent errors
# if not durations:
#     durations.append(1 / 30.0)  # Default to 1/30th of a second if only one frame

# # Create a temporary file list for ffmpeg with frame durations
# input_list_path = os.path.join(image_directory, 'input_images.txt')
# with open(input_list_path, 'w') as f:
#     for filename, duration in zip(image_files, durations):
#         f.write(f"file '{os.path.join(image_directory, filename)}'\n")
#         f.write(f"duration {duration}\n")
#     # Write the last file again to ensure the last frame duration is used
#     f.write(f"file '{os.path.join(image_directory, image_files[-1])}'\n")


# # Construct the ffmpeg command to create the video
# ffmpeg_command = [
#     'C:\\ffmpeg\\bin\\ffmpeg.exe', '-f', 'concat', '-safe', '0', '-i', input_list_path,
#     '-fps_mode', 'vfr', '-pix_fmt', 'yuv420p', output_video
# ]

# # Execute the ffmpeg command
# subprocess.run(ffmpeg_command)



# import os
# import pandas as pd
# import shutil
# import subprocess


# # # # Directory containing the .webp images
# image_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\camera_frames"

# # # # Output video file
# output_video = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_videos\output_video_variable_frame_rate.mp4"


# # Frame rate for the video (used for the maximum frame rate)
# frame_rate = 30

# # Extract order numbers and timestamps from filenames, including the fractional part of the timestamp
# def get_order_number(filename):
#     return int(filename.split('_')[0])

# # Get all .webp files in the directory and sort them by the order number
# uploaded_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.webp')],
#                         key=get_order_number)

# file_info = [{'filename': f, 'timestamp': float(f.split('_')[1].replace('.webp', ''))} for f in uploaded_files]
# print("files sorted")
# # Convert to DataFrame for easier manipulation
# df_files = pd.DataFrame(file_info)

# # Calculate time intervals between consecutive frames
# df_files['interval'] = df_files['timestamp'].diff().fillna(0)  # First frame interval set to 0
# print("intervals calculated")
# # Create the intermediate images and durations for FFmpeg
# intermediate_dir = os.path.join(image_directory, 'intermediate')
# os.makedirs(intermediate_dir, exist_ok=True)

# # Create intermediate images and write them with proper durations
# frame_index = 0
# image_sequence = []
# for idx, row in df_files.iterrows():
#     # Copy the image to the intermediate directory multiple times based on the interval
#     interval_ms = int(row['interval'] * 1000)  # Convert interval to milliseconds
#     duration = max(interval_ms, 1)  # Ensure a minimum duration of 1ms
    
#     for i in range(duration):
#         intermediate_image_path = os.path.join(intermediate_dir, f"frame_{frame_index:04d}.webp")
#         shutil.copyfile(os.path.join(image_directory, row['filename']), intermediate_image_path)
#         image_sequence.append(intermediate_image_path)
#         frame_index += 1
# print("intermediate images created")
# # Write the image sequence to a text file for FFmpeg
# intermediate_list_path = os.path.join(intermediate_dir, 'input_images.txt')
# with open(intermediate_list_path, 'w') as f:
#     for image_path in image_sequence:
#         f.write(f"file '{image_path}'\n")

# # Construct the ffmpeg command to create the variable fps video
# ffmpeg_command = [
#     'C:\\ffmpeg\\bin\\ffmpeg.exe', '-f', 'concat', '-safe', '0', '-i', intermediate_list_path,
#     '-fps_mode', 'vfr', '-pix_fmt', 'yuv420p', output_video
# ]

# # Execute the ffmpeg command
# subprocess.run(ffmpeg_command)

# print(f"Variable FPS video created: {output_video}")


import os
import pandas as pd
import subprocess

# # # # Directory containing the .webp images
image_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\resistance_frames"
# image_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\camera_frames"


# Output video file directory
output_video_directory = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_current_videos"
os.makedirs(output_video_directory, exist_ok=True)

# # # # # Output video file
# output_video = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_19-19_31\output_videos\output_video_variable_frame_rate.mp4"

# Extract order numbers and timestamps from filenames, including the fractional part of the timestamp
def get_order_number(filename):
    return int(filename.split('_')[0])

# Get all .webp files in the directory and sort them by the order number
uploaded_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.webp')],
                        key=get_order_number)

file_info = [{'filename': f, 'timestamp': float(f.split('_')[1].replace('.webp', ''))} for f in uploaded_files]

# Convert to DataFrame for easier manipulation
df_files = pd.DataFrame(file_info)

# Calculate time intervals between consecutive frames
df_files['interval'] = df_files['timestamp'].diff().fillna(0)  # First frame interval set to 0

# Calculate the number of frames and total duration
num_frames = len(df_files)
total_duration = df_files['interval'].sum()

# Create the output filename with the number of frames and total duration
output_filename = f"output_resistance_{num_frames}frames_{total_duration:.2f}seconds.mp4"
output_video = os.path.join(output_video_directory, output_filename)

# Write the image sequence with durations to a text file for FFmpeg
input_list_path = os.path.join(image_directory, 'input_images.txt')
with open(input_list_path, 'w') as f:
    for idx, row in df_files.iterrows():
        image_path = os.path.join(image_directory, row['filename'])
        duration = row['interval']
        f.write(f"file '{image_path}'\n")
        f.write(f"duration {duration}\n")

# Add the last image again for FFmpeg to process it correctly
last_image_path = os.path.join(image_directory, df_files.iloc[-1]['filename'])
with open(input_list_path, 'a') as f:
    f.write(f"file '{last_image_path}'\n")

# Construct the ffmpeg command to create the variable fps video
ffmpeg_command = [
    'C:\\ffmpeg\\bin\\ffmpeg.exe', '-f', 'concat', '-safe', '0', '-i', input_list_path,
    '-vsync', 'vfr', '-pix_fmt', 'yuv420p', output_video
]

# Execute the ffmpeg command
subprocess.run(ffmpeg_command)

print(f"Variable FPS video created: {output_video}")
