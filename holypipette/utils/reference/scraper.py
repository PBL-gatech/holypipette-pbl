# used to grab images from Dr. Matt Gaidica website, assistant professor in the Department of Neuroscience at Washington University in St. Louis
# https://labs.gaidi.ca/mouse-brain-atlas/
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import itertools
from urllib.parse import urljoin

# Base URL of the website
base_url = "https://labs.gaidi.ca/mouse-brain-atlas/"

# Define the folder path
folder_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\reference\images"

# Create a directory to save the images
os.makedirs(folder_path, exist_ok=True)

# Generate combinations of coordinates with increments of 0.12
ml_values = np.arange(-8, 4.12, 0.12)
ap_values = np.arange(0.12, 3.12, 0.12)
coordinate_combinations = itertools.product(ml_values, ap_values)

def download_image(img_url, ml, ap, view):
    img_data = requests.get(img_url).content
    img_name = os.path.join(folder_path, f"ML_{ml}_AP_{ap}_{view}.jpg")
    with open(img_name, 'wb') as handler:
        handler.write(img_data)
    print(f"Downloaded {img_name}")

def process_page(url, ml, ap):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        
        if len(img_tags) >= 2:
            coronal_img_url = urljoin(base_url, img_tags[0]['src'])
            sagittal_img_url = urljoin(base_url, img_tags[1]['src'])
            download_image(coronal_img_url, ml, ap, 'coronal')
            download_image(sagittal_img_url, ml, ap, 'sagittal')
        else:
            print(f"Not enough images found for ML={ml}, AP={ap}")
    else:
        print(f"Failed to load page for ML={ml}, AP={ap} - Status Code: {response.status_code}")

for ml, ap in coordinate_combinations:
    ml_str = f"{ml:.2f}"
    ap_str = f"{ap:.2f}"
    page_url = f"{base_url}?ml={ml_str}&ap={ap_str}"
    process_page(page_url, ml_str, ap_str)

print("All images have been downloaded.")
