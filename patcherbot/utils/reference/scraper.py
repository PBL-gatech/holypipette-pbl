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
folder_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\reference\images"

# Create a directory to save the images
os.makedirs(folder_path, exist_ok=True)

# Generate combinations of coordinates with increments of 0.12
ml_values = np.arange(-8, 4.12, 0.12)
ap_values = np.arange(0.12, 3.12, 0.12)
coordinate_combinations = itertools.product(ml_values, ap_values)

def download_image(img_url, ml, ap, view):
    """
    Download an image from a given URL and save it locally with a structured filename.

    Args:
        img_url (str): URL of the image to download.
        ml (float or str): Mediolateral coordinate associated with the image.
        ap (float or str): Anteroposterior coordinate associated with the image.
        view (str): View type of the image (e.g., 'coronal', 'sagittal').
    """
    img_data = requests.get(img_url).content
    img_name = os.path.join(folder_path, f"ML_{ml}_AP_{ap}_{view}.jpg")
    with open(img_name, 'wb') as handler:
        handler.write(img_data)
    print(f"Downloaded {img_name}")

def process_page(url, ml, ap):
    """
    Fetch a webpage corresponding to specific brain coordinates, extract image URLs,
    and download the coronal and sagittal views if available.

    Args:
        url (str): The URL of the webpage to process.
        ml (float or str): Mediolateral coordinate used for labeling downloaded images.
        ap (float or str): Anteroposterior coordinate used for labeling downloaded images.

    Raises:
        None: Errors are handled internally via status checks and printed messages.
    """
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
