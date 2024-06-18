import os
import pandas as pd
import matplotlib.pyplot as plt

def process_images_and_plot(input_directory, output_directory, output_filename):
    # Use a generator to avoid loading all filenames into memory at once
    def get_file_names(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.webp'):
                yield filename
    
    # Extract frame numbers and timestamps from filenames using the generator
    frames_timestamps = [
        (int(name.split('_')[0]), float(name.split('_')[1].split('.webp')[0]))
        for name in get_file_names(input_directory)
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
    
    # Plot framerate over cumulative time with horizontal lines at 60fps and 30fps
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['framerate'], marker='o', label='Framerate')
    plt.axhline(y=60, color='r', linestyle='--', label='60 fps')
    plt.axhline(y=30, color='g', linestyle='--', label='30 fps')
    plt.axhline(y=average_framerate, color='b', linestyle='--', label=f'Average Framerate: {average_framerate:.2f} fps')
    plt.title('Framerate Over Time with Horizontal Lines at 60fps, 30fps, and Average Framerate')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Framerate (frames per second)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the output directory
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)
    plt.close()

    # Print the average framerate
    print(f'Average Framerate: {average_framerate:.2f} fps')




# Example usage:
input_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_18-16_15\camera_frames'
output_directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2024_06_18-16_15\output_videos'
output_filename = 'framerate_plot.png'
process_images_and_plot(input_directory, output_directory, output_filename)
