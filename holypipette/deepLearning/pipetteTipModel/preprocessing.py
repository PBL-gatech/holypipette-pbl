import cv2
import numpy as np
import os
import random

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_canny_edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def apply_morphological_transformation(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resize_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)

def flip_image(image, labels, flip_code):
    """Flip the image and adjust bounding box labels."""
    flipped_image = cv2.flip(image, flip_code)
    # Adjust labels
    updated_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        if flip_code == 1:  # Horizontal flip
            x = 1 - x
        elif flip_code == 0:  # Vertical flip
            y = 1 - y
        elif flip_code == -1:  # Both horizontal and vertical flip
            x = 1 - x
            y = 1 - y
        updated_labels.append((class_id, x, y, w, h))
    return flipped_image, updated_labels

def shear_image(image, labels):
    """Apply a shearing transformation to the image and adjust labels."""
    rows, cols = image.shape[:2]
    shear_factor = random.uniform(0.1, 0.3)
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols, rows))
    
    # Adjust labels
    updated_labels = []
    for label in labels:
        class_id, x, y, w, h = label
        # Calculate new x position after shearing
        x = x + shear_factor * y
        updated_labels.append((class_id, x, y, w, h))
    return sheared_image, updated_labels

def read_labels(label_path):
    """Read label file and return a list of labels."""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                labels.append((class_id, x, y, w, h))
    return labels

def save_labels(label_path, labels):
    """Save updated labels to a file."""
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def process_and_save_images(input_folder, label_folder, output_folder, output_label_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    augmentations = ["flip", "shear", "gaussian_blur", "clahe", "canny_edge", "morphological"]

    for filename in os.listdir(input_folder):
        if filename.endswith(('.webp')):
            filepath = os.path.join(input_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Read labels
                labels = read_labels(label_path)

                # Resize the original image
                resized = resize_image(image)

                # Save the original resized image and labels
                original_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_original.webp")
                cv2.imwrite(original_output_path, resized)
                original_label_output_path = os.path.join(output_label_folder, f"{os.path.splitext(filename)[0]}_original.txt")
                save_labels(original_label_output_path, labels)
                print(f"Processed and saved: {original_output_path}")

                # Apply each augmentation to the image
                for aug in augmentations:
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

                    # Save the augmented image and updated labels
                    augmented_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{aug}.webp")
                    cv2.imwrite(augmented_output_path, augmented_image)
                    augmented_label_output_path = os.path.join(output_label_folder, f"{os.path.splitext(filename)[0]}_{aug}.txt")
                    save_labels(augmented_label_output_path, updated_labels)
                    print(f"Processed and saved: {augmented_output_path}")

            else:
                print(f"Failed to load image: {filename}")

if __name__ == "__main__":
    input_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\Dataset\newdataset\images"
    label_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\Dataset\newdataset\labels"
    output_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\Dataset\newdataset\P_DET_IMAGES"
    output_label_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\Dataset\newdataset\P_DET_LABELS"
    process_and_save_images(input_folder, label_folder, output_folder, output_label_folder)