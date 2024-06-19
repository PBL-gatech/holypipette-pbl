import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import re
import concurrent.futures

def find_latest_camera_frames_dir(parent_dir):
    latest_time = None
    latest_dir = None

    for root, dirs, files in os.walk(parent_dir):
        if 'camera_frames' in dirs:
            dir_path = os.path.join(root, 'camera_frames')
            parent_dir_name = os.path.basename(root)

            try:
                dir_time = datetime.strptime(parent_dir_name, '%Y_%m_%d-%H_%M')
                if latest_time is None or dir_time > latest_time:
                    latest_time = dir_time
                    latest_dir = dir_path
            except ValueError:
                continue
    # print(f"Latest camera frames directory found: {latest_dir}")
    # print(f"Latest camera frames directory time: {latest_time}")
    # print(f"parent directory:{parent_dir_name}")
    # print(f"intended save directory:{root}")
    output_dir = latest_dir.split('camera_frames')[0] + 'output_videos'
    print(f"output directory:{output_dir}")
    return latest_dir, output_dir

def load_image(file):
    return cv2.imread(file)

def create_videos_from_directory(parent_directory):
    # Create output directory if it doesn't exist
    input_dir, output_dir = find_latest_camera_frames_dir(parent_directory)
    os.makedirs(output_dir, exist_ok=True)

    # Define the pattern to extract frame index and timestamp from file names
    pattern = re.compile(r'(\d+)_(\d+\.\d+)\.webp')

    # Find all webp files in the directory and extract frame info
    frames = []
    image_files = []
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            frame_index = int(match.group(1))
            timestamp = float(match.group(2))
            frames.append((frame_index, timestamp, filename))

    # Sort frames and image files by timestamp
    frames.sort(key=lambda x: x[1])
    image_files = [os.path.join(input_dir, frame[2]) for frame in frames]
    print("Image files collected")

    # Load the images using multi-threading for faster performance
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_files))
    print("Images loaded")

    # Define video names (initially without frame counts)
    video_name_variable_fps = os.path.join(output_dir, "output_video_variable_fps_temp.mp4")
    video_name_constant_fps = os.path.join(output_dir, "output_video_constant_fps_temp.mp4")

    # Base frame rate for the constant FPS video
    base_frame_rate = 30

    # Define the video writer
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print("Starting to write videos")

    # Video writer for variable FPS
    video_variable_fps = cv2.VideoWriter(video_name_variable_fps, fourcc, base_frame_rate, (width, height))

    # Determine the duration each frame should be displayed for variable FPS video
    durations = []
    previous_time = 0
    for frame in frames:
        current_time = frame[1]
        durations.append(current_time - previous_time)
        previous_time = current_time

    # Add frames to the variable FPS video
    variable_frame_count = 0
    for img, duration in zip(images, durations):
        frame_count = int(duration * base_frame_rate)
        if frame_count == 0:
            frame_count = 1  # Ensure at least one frame is written
        for _ in range(frame_count):
            video_variable_fps.write(img)
            variable_frame_count += 1

    # Release the video writer for variable FPS
    video_variable_fps.release()

    # Video writer for constant FPS
    video_constant_fps = cv2.VideoWriter(video_name_constant_fps, fourcc, base_frame_rate, (width, height))

    # Add frames to the constant FPS video (each frame is displayed for the same duration)
    constant_frame_count = len(images)
    for img in images:
        video_constant_fps.write(img)

    # Release the video writer for constant FPS
    video_constant_fps.release()

    # Define final video names with frame counts
    final_video_name_variable_fps = os.path.join(output_dir, f"output_video_variable_fps_{variable_frame_count}_frames.mp4")
    final_video_name_constant_fps = os.path.join(output_dir, f"output_video_constant_fps_{constant_frame_count}_frames.mp4")

    # Rename temporary files to final names
    os.rename(video_name_variable_fps, final_video_name_variable_fps)
    os.rename(video_name_constant_fps, final_video_name_constant_fps)

    print("Videos saved at:")
    print(final_video_name_variable_fps)
    print(final_video_name_constant_fps)

def process_images_and_plot(parent_directory, output_filename='framerate_plot.png'):
    input_dir, output_dir = find_latest_camera_frames_dir(parent_directory)
    os.makedirs(output_dir, exist_ok=True)
    # Use a generator to avoid loading all filenames into memory at once
    def get_file_names(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.webp'):
                yield filename
    
    # Extract frame numbers and timestamps from filenames using the generator
    frames_timestamps = [
        (int(name.split('_')[0]), float(name.split('_')[1].split('.webp')[0]))
        for name in get_file_names(input_dir)
    ]
    
    # Convert to a DataFrame
    df = pd.DataFrame(frames_timestamps, columns=['frame', 'timestamp'])
    
    # Sort by frame number to ensure correct order
    df = df.sort_values(by='frame').reset_index(drop=True)
    
    # Calculate time intervals (differences between consecutive timestamps)
    df['time_interval'] = df['timestamp'].diff()
    
    # Calculate framerate (frames per second)
    df['framerate'] = 1 / df['time_interval']
    
    # Remove the first row as it will have NaN for time_interval and framerate
    df = df.dropna()
    
    # Calculate the average framerate
    average_framerate = df['framerate'].mean()

    # Calculate the running average of the framerate
    window_size = 20  # You can adjust the window size for the running average
    df['running_avg_framerate'] = df['framerate'].rolling(window=window_size).mean()

    # Calculate the cumulative average framerate
    df['cumulative_avg_framerate'] = df['framerate'].expanding().mean()

    # Plot framerate over cumulative time with horizontal lines at 60fps and 30fps
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['framerate'], marker='o', label='Framerate')
    plt.plot(df['timestamp'], df['running_avg_framerate'], color='orange', linestyle='-', label='Running Average Framerate')
    plt.plot(df['timestamp'], df['cumulative_avg_framerate'], color='purple', linestyle='-', label='Cumulative Average Framerate')
    plt.axhline(y=60, color='r', linestyle='--', label='60 fps')
    plt.axhline(y=30, color='g', linestyle='--', label='30 fps')
    plt.axhline(y=average_framerate, color='b', linestyle='--', label=f'Overall Average Framerate: {average_framerate:.2f} fps')
    plt.title('Framerate Over Time with Horizontal Lines at 60fps, 30fps, and Average Framerates')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Framerate (frames per second)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the output directory
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()

    # Print the average framerate
    print(f'Overall Average Framerate: {average_framerate:.2f} fps')


# Example usage
input_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data'
# create_videos_from_directory(input_directory)
process_images_and_plot(input_directory)


