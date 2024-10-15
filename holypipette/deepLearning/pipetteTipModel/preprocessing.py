import cv2
import numpy as np
import os
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("image_processing_debug.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

def apply_gaussian_blur(image):
    logging.debug("Applying Gaussian Blur.")
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_clahe(image):
    logging.debug("Applying CLAHE.")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_canny_edge_detection(image):
    logging.debug("Applying Canny Edge Detection.")
    edges = cv2.Canny(image, 50, 150)
    return edges

def apply_morphological_transformation(image):
    logging.debug("Applying Morphological Transformation (Closing).")
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resize_image(image, target_size=(640, 640)):
    logging.debug(f"Resizing image to {target_size}.")
    return cv2.resize(image, target_size)

def flip_image(image, labels, flip_code):
    """Flip the image and adjust bounding box labels."""
    logging.debug(f"Flipping image with flip_code={flip_code}.")
    flipped_image = cv2.flip(image, flip_code)
    updated_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        original_x, original_y = x, y
        if flip_code == 1:  # Horizontal flip
            x = 1 - x
        elif flip_code == 0:  # Vertical flip
            y = 1 - y
        elif flip_code == -1:  # Both horizontal and vertical flip
            x = 1 - x
            y = 1 - y
        updated_labels.append((class_id, x, y, w, h))
        logging.debug(f"Adjusted label from ({class_id}, {original_x}, {original_y}, {w}, {h}) to ({class_id}, {x}, {y}, {w}, {h}).")
    return flipped_image, updated_labels

def shear_image(image, labels):
    """Apply a shearing transformation to the image and adjust labels."""
    logging.debug("Applying Shear Transformation.")
    rows, cols = image.shape[:2]
    shear_factor = random.uniform(0.1, 0.3)
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    logging.debug(f"Shear factor applied: {shear_factor}")

    updated_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        original_x = x
        # Calculate new x position after shearing
        x = x + shear_factor * y
        updated_labels.append((class_id, x, y, w, h))
        logging.debug(f"Adjusted label x from {original_x} to {x} after shearing.")
    return sheared_image, updated_labels

def read_labels(label_path):
    """Read label file and return a list of labels."""
    logging.debug(f"Reading labels from {label_path}.")
    labels = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logging.warning(f"Incorrect label format in {label_path}: {line.strip()}")
                        continue
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    labels.append((class_id, x, y, w, h))
            logging.debug(f"Total labels read: {len(labels)}")
        except Exception as e:
            logging.error(f"Error reading labels from {label_path}: {e}")
    else:
        logging.warning(f"Label file does not exist: {label_path}")
    return labels

def save_labels(label_path, labels):
    """Save updated labels to a file."""
    logging.debug(f"Saving labels to {label_path}.")
    try:
        with open(label_path, 'w') as file:
            for label in labels:
                file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        logging.debug("Labels saved successfully.")
    except Exception as e:
        logging.error(f"Error saving labels to {label_path}: {e}")

def process_and_save_images(input_folder, label_folder, output_folder, output_label_folder):
    logging.info("Starting image processing...")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    augmentations = ["flip", "shear", "gaussian_blur", "clahe", "canny_edge", "morphological"]

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            filepath = os.path.join(input_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
            logging.info(f"Processing file: {filename}")

            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logging.error(f"Failed to load image: {filename}")
                continue
            else:
                logging.debug(f"Image loaded successfully: {filename}")

            # Read labels
            labels = read_labels(label_path)

            # Resize the original image
            try:
                resized = resize_image(image)
                logging.debug("Image resized successfully.")
            except Exception as e:
                logging.error(f"Error resizing image {filename}: {e}")
                continue

            # Save the original resized image and labels
            try:
                original_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_original.jpg")
                cv2.imwrite(original_output_path, resized)
                logging.debug(f"Original image saved to {original_output_path}")

                original_label_output_path = os.path.join(output_label_folder, f"{os.path.splitext(filename)[0]}_original.txt")
                save_labels(original_label_output_path, labels)
            except Exception as e:
                logging.error(f"Error saving original image or labels for {filename}: {e}")
                continue

            # Apply each augmentation to the image
            for aug in augmentations:
                try:
                    logging.info(f"Applying augmentation: {aug} on {filename}")
                    if aug == "flip":
                        flip_code = random.choice([0, 1, -1])
                        augmented_image, updated_labels = flip_image(resized, labels, flip_code)
                    elif aug == "shear":
                        augmented_image, updated_labels = shear_image(resized, labels)
                    elif aug == "gaussian_blur":
                        augmented_image = apply_gaussian_blur(resized)
                        updated_labels = labels
                    elif aug == "clahe":
                        augmented_image = apply_clahe(resized)
                        updated_labels = labels
                    elif aug == "canny_edge":
                        augmented_image = apply_canny_edge_detection(resized)
                        updated_labels = labels
                    elif aug == "morphological":
                        augmented_image = apply_morphological_transformation(resized)
                        updated_labels = labels
                    else:
                        logging.warning(f"Unknown augmentation type: {aug}")
                        continue

                    # Save the augmented image and updated labels
                    augmented_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{aug}.webp")
                    cv2.imwrite(augmented_output_path, augmented_image)
                    logging.debug(f"Augmented image saved to {augmented_output_path}")

                    augmented_label_output_path = os.path.join(output_label_folder, f"{os.path.splitext(filename)[0]}_{aug}.txt")
                    save_labels(augmented_label_output_path, updated_labels)
                    logging.debug(f"Augmented labels saved to {augmented_label_output_path}")

                except Exception as e:
                    logging.error(f"Error applying augmentation {aug} on {filename}: {e}")
                    continue

    logging.info("Image processing completed.")

if __name__ == "__main__":
    try:
        # Define your paths here
        input_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\focus_dataset\focus_training_set\images"
        label_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\focus_dataset\focus_training_set\labels"
        output_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\focus_dataset\focus_training_set\P_DET_IMAGES"
        output_label_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\focus_dataset\focus_training_set\P_DET_LABELS"

        logging.debug(f"Input folder: {input_folder}")
        logging.debug(f"Label folder: {label_folder}")
        logging.debug(f"Output folder: {output_folder}")
        logging.debug(f"Output label folder: {output_label_folder}")

        process_and_save_images(input_folder, label_folder, output_folder, output_label_folder)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
