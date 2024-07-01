


import os
import pandas as pd

def get_dir_size(directory):
    """ Calculate the total size of files in a directory in megabytes. """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert bytes to megabytes

def add_folder_sizes(base_directory, csv_file_path):
    """ Add folder sizes to the CSV data. """
    data = pd.read_csv(csv_file_path, delimiter='\t')
    folder_sizes = []
    
    for index, row in data.iterrows():
        # Construct the directory path using both package and format
        dir_name = f"{row['Package']}_{row['Format']}"
        dir_path = os.path.join(base_directory, dir_name)
        # Get the directory size
        try:
            size_mb = get_dir_size(dir_path)
        except Exception as e:
            print(f"Error accessing {dir_path}: {e}")
            size_mb = 0  # Assume size is 0 if any error occurs
        folder_sizes.append(size_mb)
    
    # Add the folder sizes as a new column
    data['FolderSize (MB)'] = folder_sizes
    # Save the modified DataFrame to a new CSV file
    new_csv_path = csv_file_path.replace('.csv', '_SIMD_with_sizes.csv')
    data.to_csv(new_csv_path, index=False, sep='\t')
    print(f"Updated CSV saved to {new_csv_path}")

# Usage example:
save_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder\Reconstruct\image_saver_test"

csv_file_path = save_folder + r"\average_durations.csv"
add_folder_sizes(save_folder, csv_file_path)

