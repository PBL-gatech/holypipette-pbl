import os
import shutil

def main():
    # ===========================
    # Configuration
    # ===========================

    # Set this variable to True to enable condensing into 'focus_training_set'
    CONDENSE = True  # Set to False to disable condensing

    # Define the main directory containing the classified images
    main_dir = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\classified_images"

    # Define the class folders
    class_folders = ['focused', 'above', 'below']

    # Define subdirectories for images and labels
    p_det_images = 'P_DET_IMAGES'
    p_det_labels = 'P_DET_LABELS'

    # Define label mappings
    label_ID = {
        "in_focus": 0,
        "above_plane": 1,
        "below_plane": 2,
        "not_detected": 3
    }

    # Initialize counters
    total_images = 0
    labeled_images = 0
    not_detected_images = 0
    focus_training_set_images = 0  # Counter for condensed images

    # Dictionary to store labels for each labeled image
    images_with_labels = {}  # Key: image filename, Value: list of labels

    # If condensing is enabled, set up the destination directories
    if CONDENSE:
        focus_training_set_dir = os.path.join(main_dir, 'focus_training_set')
        focus_images_dir = os.path.join(focus_training_set_dir, p_det_images)
        focus_labels_dir = os.path.join(focus_training_set_dir, p_det_labels)

        # Create the focus_training_set directories if they don't exist
        os.makedirs(focus_images_dir, exist_ok=True)
        os.makedirs(focus_labels_dir, exist_ok=True)

        print(f"Condensing is enabled. 'focus_training_set' directories created at:")
        print(f"Images: {focus_images_dir}")
        print(f"Labels: {focus_labels_dir}\n")

    # Traverse each class folder
    for class_folder in class_folders:
        images_dir = os.path.join(main_dir, class_folder, p_det_images)
        labels_dir = os.path.join(main_dir, class_folder, p_det_labels)

        # Check if directories exist
        if not os.path.exists(images_dir):
            print(f"Images directory does not exist: {images_dir}")
            continue
        if not os.path.exists(labels_dir):
            print(f"Labels directory does not exist: {labels_dir}")
            continue

        # Iterate over each image in the images directory
        for filename in os.listdir(images_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue  # Skip non-image files

            total_images += 1
            image_base, _ = os.path.splitext(filename)
            label_file = os.path.join(labels_dir, image_base + '.txt')

            if os.path.exists(label_file):
                labeled_images += 1
                labels = set()
                try:
                    with open(label_file, 'r') as f:
                        for line_number, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) < 1:
                                print(f"Warning: Line {line_number} in {label_file} is empty or malformed.")
                                continue
                            label_num_str = parts[0]
                            if label_num_str.isdigit():
                                label_num = int(label_num_str)
                                if label_num in label_ID.values():
                                    labels.add(label_num)
                                else:
                                    print(f"Warning: Unknown label ID {label_num} in {label_file} on line {line_number}.")
                            else:
                                print(f"Warning: Non-integer label '{label_num_str}' in {label_file} on line {line_number}.")
                    images_with_labels[filename] = sorted(labels)

                    # If condensing is enabled, copy the image and label file
                    if CONDENSE:
                        # Check if at least one label is 0, 1, or 2
                        if any(label in [0, 1, 2] for label in labels):
                            src_image_path = os.path.join(images_dir, filename)
                            src_label_path = label_file

                            dest_image_path = os.path.join(focus_images_dir, filename)
                            dest_label_path = os.path.join(focus_labels_dir, image_base + '.txt')

                            # Copy image
                            try:
                                shutil.copy2(src_image_path, dest_image_path)
                                print(f"Copied image: {src_image_path} to {dest_image_path}")
                            except Exception as e:
                                print(f"Error copying image {src_image_path} to {dest_image_path}: {e}")
                                continue  # Skip label copying if image copy fails

                            # Copy label file without modifying
                            try:
                                shutil.copy2(src_label_path, dest_label_path)
                                print(f"Copied label: {src_label_path} to {dest_label_path}\n")
                                focus_training_set_images += 1
                            except Exception as e:
                                print(f"Error copying label {src_label_path} to {dest_label_path}: {e}")
                except Exception as e:
                    print(f"Error reading label file {label_file}: {e}")
            else:
                not_detected_images += 1

    # Summary Report
    print("\n=== Image Labeling Summary ===")
    print(f"Total images processed: {total_images}")
    print(f"Images with labels: {labeled_images}")
    print(f"Images without labels (assumed 'not_detected'): {not_detected_images}\n")

    if images_with_labels:
        print("=== Labeled Images and Their Labels ===")
        for img, labels in sorted(images_with_labels.items()):
            label_names = [name for name, id_ in label_ID.items() if id_ in labels]
            print(f"{img}: {labels} ({', '.join(label_names)})")
    else:
        print("No labeled images found.")

    # Summary of label distribution
    label_distribution = {id_: 0 for id_ in label_ID.values()}
    for labels in images_with_labels.values():
        for label in labels:
            label_distribution[label] += 1

    print("\n=== Label Distribution ===")
    for label_name, label_id in label_ID.items():
        count = label_distribution.get(label_id, 0)
        print(f"Label '{label_name}' (ID {label_id}): {count} image(s)")

    # Condense Summary
    if CONDENSE:
        print("\n=== Focus Training Set Summary ===")
        print(f"Total images added to 'focus_training_set': {focus_training_set_images}")
        print(f"Images copied to '{focus_images_dir}'")
        print(f"Labels copied to '{focus_labels_dir}'\n")
    else:
        print("\nCondensing was not performed. To condense labeled images into 'focus_training_set', set CONDENSE = True in the script and rerun.")

if __name__ == "__main__":
    main()
