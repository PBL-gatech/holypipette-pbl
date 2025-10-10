from pathlib import Path
import numpy as np
from PIL import Image
import csv

# --- USER CONFIGURATION ---
input_folder = Path(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\AgentPath3")  # change this to your actual folder
output_csv = Path(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\AgentPath3\red_dot_coordinates.csv")

# --- SCRIPT START ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def find_red_centroid(img_path: Path, r_min=150, g_max=100, b_max=100):
    """Return (x, y) centroid of strong red pixels or None if not found."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
    except Exception:
        return None

    red_mask = (arr[:, :, 0] >= r_min) & (arr[:, :, 1] <= g_max) & (arr[:, :, 2] <= b_max)
    coords = np.argwhere(red_mask)

    if coords.size == 0:
        return None

    y, x = coords.mean(axis=0)
    return int(round(x)), int(round(y))

rows = []
for img_file in sorted(input_folder.glob("*")):
    if img_file.suffix.lower() in IMAGE_EXTS:
        xy = find_red_centroid(img_file)
        if xy:
            x, y = xy
            rows.append([img_file.name, x, y])
        else:
            rows.append([img_file.name, "", ""])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x", "y"])
    writer.writerows(rows)

print(f"Saved CSV with coordinates to: {output_csv}")
