# will test the saving performance of differnt image formats wwith differnt packages
import numpy as np
import time
import cv2
from PIL import Image
import imageio
import os

# Generate random images of size 1280x1280
images = [np.random.randint(0, 256, (1280, 1280, 3), dtype=np.uint8) for _ in range(100)]

# Define the formats to save the images in
formats = {
    'Pillow': {
        'formats': ['jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp', 'ppm'],
        'extension': {'jpeg': 'jpg', 'png': 'png', 'bmp': 'bmp', 'gif': 'gif', 'tiff': 'tiff', 'webp': 'webp', 'ppm': 'ppm'}
    },
    'OpenCV': {
        'formats': ['jpeg', 'png', 'bmp', 'tiff', 'webp'],
        'extension': {'jpeg': 'jpg', 'png': 'png', 'bmp': 'bmp', 'tiff': 'tiff', 'webp': 'webp'}
    },
    'imageio': {
        'formats': ['jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp', 'ppm'],
        'extension': {'jpeg': 'jpg', 'png': 'png', 'bmp': 'bmp', 'gif': 'gif', 'tiff': 'tiff', 'webp': 'webp', 'ppm': 'ppm'}
    }
}

# Function definitions for saving images with different libraries
def save_with_pillow(images, format, save_folder):
    times = []
    specific_save_folder = os.path.join(save_folder, f"Pillow_{format}")
    os.makedirs(specific_save_folder, exist_ok=True)
    for idx, img in enumerate(images):
        pil_img = Image.fromarray(img)
        timestamp = int(time.time())
        filename = f"{specific_save_folder}\\frame-{idx}_{timestamp}.{formats['Pillow']['extension'][format]}"
        start_time = time.time()
        pil_img.save(filename, format=format)
        times.append(time.time() - start_time)
    return np.mean(times)

def save_with_opencv(images, format, save_folder):
    times = []
    specific_save_folder = os.path.join(save_folder, f"OpenCV_{format}")
    os.makedirs(specific_save_folder, exist_ok=True)
    for idx, img in enumerate(images):
        timestamp = int(time.time())
        filename = f"{specific_save_folder}\\frame-{idx}_{timestamp}.{formats['OpenCV']['extension'][format]}"
        start_time = time.time()
        cv2.imwrite(filename, img)
        times.append(time.time() - start_time)
    return np.mean(times)

def save_with_imageio(images, format, save_folder):
    times = []
    specific_save_folder = os.path.join(save_folder, f"imageio_{format}")
    os.makedirs(specific_save_folder, exist_ok=True)
    for idx, img in enumerate(images):
        timestamp = int(time.time())
        filename = f"{specific_save_folder}\\frame-{idx}_{timestamp}.{formats['imageio']['extension'][format]}"
        start_time = time.time()
        imageio.imwrite(filename, img, format=format)
        times.append(time.time() - start_time)
    return np.mean(times)


# Specify the folder to save the images
save_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder\Reconstruct\image_saver_test"

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)
print("Saving images to different formats using different packages...")
# Compute average durations for all packages and formats
average_durations = {}
for package, info in formats.items():
    average_durations[package] = {}
    for format in info['formats']:
        if package == 'Pillow':
            avg_time = save_with_pillow(images, format, save_folder)
        elif package == 'OpenCV':
            avg_time = save_with_opencv(images, format, save_folder)
        elif package == 'imageio':
            avg_time = save_with_imageio(images, format, save_folder)
        average_durations[package][format] = avg_time

average_durations

import csv
print("Saving average durations to a CSV file...")
# Existing code to calculate average_durations...

# Specify the CSV file path
csv_file_path = save_folder + r"\average_durations.csv"

# Ensure the save folder exists for the CSV file
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Write the average durations to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    # Write the header
    writer.writerow(['Package', 'Format', 'Average Duration (s)'])
    # Write the data
    for package, formats in average_durations.items():
        for format, avg_time in formats.items():
            writer.writerow([package, format, avg_time])