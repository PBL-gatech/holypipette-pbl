import json
import os
from pathlib import Path

# Define class mappings (e.g., "tip" -> 0)
class_mapping = {
    "tip": 0  # You can add other classes if needed
}

def convert_json_to_yolo(json_folder, image_folder, output_folder):
    """
    Convert JSON annotations (with normalized coordinates) to YOLO format and save .txt files in the output folder.
    json_folder: Path to folder with JSON files
    image_folder: Path to folder with corresponding images
    output_folder: Path to save YOLO annotations
    """
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get the list of all image filenames without extensions
    image_filenames = {os.path.splitext(img)[0]: img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.webp'))}

    # Loop over all JSON files in the folder
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)

            # Read the JSON file
            with open(json_path) as f:
                data = json.load(f)

            # Extract the base filename (without extension) of the JSON file
            json_base = os.path.splitext(json_file)[0]

            # Check if there's a corresponding image file with the same base name
            if json_base in image_filenames:
                image_filename = image_filenames[json_base]
            else:
                print(f"No matching image found for {json_file}")
                continue  # Skip if no corresponding image is found

            # Create corresponding YOLO format .txt file
            txt_filename = json_base + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)

            with open(txt_path, 'w') as txt_file:
                # Iterate over all bounding boxes in the JSON annotation
                for obj in data['bounding_boxes']:
                    class_name = obj['label']
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        
                        # Write the bounding box to the YOLO .txt file in the correct format
                        x_center = obj['x_center']
                        y_center = obj['y_center']
                        width = obj['width']
                        height = obj['height']
                        
                        txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Annotations converted to YOLO format and saved in {output_folder}")

if __name__ == '__main__':
    # Paths to the folders
    json_folder = 'convert/P_DET_LABELS'  # JSON annotations folder
    image_folder = 'convert/P_DET_IMAGES'  # Images folder
    output_folder = 'convert/output'  # Output folder for YOLO labels

    # Run the conversion
    convert_json_to_yolo(json_folder, image_folder, output_folder)
