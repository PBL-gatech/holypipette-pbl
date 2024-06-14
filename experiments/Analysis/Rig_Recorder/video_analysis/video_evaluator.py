import cv2
import os
import re
import concurrent.futures

def load_image(file):
    return cv2.imread(file)

def create_videos_from_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
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
    final_video_name_variable_fps = os.path.join(output_dir, f"output_video_variable_fps_{variable_frame_count}frames.mp4")
    final_video_name_constant_fps = os.path.join(output_dir, f"output_video_constant_fps_{constant_frame_count}frames.mp4")

    # Rename temporary files to final names
    os.rename(video_name_variable_fps, final_video_name_variable_fps)
    os.rename(video_name_constant_fps, final_video_name_constant_fps)

    print("Videos saved at:")
    print(final_video_name_variable_fps)
    print(final_video_name_constant_fps)

# Example usage
input_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\rig_recorder_data\2024_06_14-16_50\camera_frames'
output_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\rig_recorder_data\2024_06_14-16_50\output_videos'
create_videos_from_directory(input_directory, output_directory)
