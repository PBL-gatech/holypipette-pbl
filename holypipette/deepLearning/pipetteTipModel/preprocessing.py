import cv2
import numpy as np
import os

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

def process_and_save_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(( '.webp')): 
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is not None:
            
                blurred = apply_gaussian_blur(image)

                contrast_enhanced = apply_clahe(blurred)
                morph_cleaned = apply_morphological_transformation(contrast_enhanced)


                edges = apply_canny_edge_detection(morph_cleaned)

                resized = resize_image(edges)

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, resized)

                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to load image: {filename}")

if __name__ == "__main__":
    input_folder = "P_DET_IMAGES" 
    output_folder = "processed_images"  
    process_and_save_images(input_folder, output_folder)
