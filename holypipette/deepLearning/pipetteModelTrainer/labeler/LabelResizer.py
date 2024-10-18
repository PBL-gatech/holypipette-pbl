import os

# Parameters to adjust the bounding box size
ADJUST_AMOUNT = (5/3)  # Modify this value to control the adjustment

def adjust_bounding_box_size(folder_path, adjust_amount):
    # Loop through all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Store adjusted bounding boxes
            updated_lines = []

            for line in lines:
                components = line.strip().split()
                
                if len(components) != 5:
                    # Skip invalid lines
                    continue
                
                class_id = components[0]
                x_center = float(components[1])
                y_center = float(components[2])
                width = float(components[3])
                height = float(components[4])

                # Adjust width and height
                new_width = max(0, min(1, width*adjust_amount))
                new_height = max(0, min(1, height*adjust_amount))

                # Format and add the updated line
                updated_line = f"{class_id} {x_center:.6f} {y_center:.6f} {new_width:.6f} {new_height:.6f}\n"
                updated_lines.append(updated_line)

            # Write the updated bounding boxes back to the file
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)

# Specify the folder containing YOLO label files
folder_path = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\classified_images\focus_set-2\P_DET_LABELS"  # Replace with your folder path

# Call the function to adjust bounding box sizes
adjust_bounding_box_size(folder_path, ADJUST_AMOUNT)
